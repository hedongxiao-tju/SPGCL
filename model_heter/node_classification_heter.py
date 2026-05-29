import sys
sys.path.append("..")
import utils
import seaborn as sb
import torch
from model import Model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from layers import LogReg
from torch.utils.data import Dataset
import torch.nn.functional as F
import statistics
import argparse
import random
import warnings
import dgl

def build_similarity_drop_matrix(x: torch.Tensor, topk_idx: torch.Tensor, scale: float) -> torch.Tensor:
    n = x.size(0)
    drop_prob = torch.zeros((n, n), device=x.device)
    if topk_idx.numel() == 0:
        return drop_prob
    x_top = x[:, topk_idx]
    x_norm = F.normalize(x_top, p=2, dim=1)
    sim = torch.mm(x_norm, x_norm.t())  
    drop_prob = sim.clamp(min=0.0, max=1.0) * scale  
    drop_prob = drop_prob.clamp(max=1.0)
    drop_prob.diagonal().fill_(1.0)
    return drop_prob

def sample_edges_with_probs(edge_index: torch.Tensor, drop_prob: torch.Tensor) -> torch.Tensor:
    """
    Sample edges with per-edge drop probability from a precomputed matrix.
    """
    row, col = edge_index
    p = drop_prob[row, col]
    keep_mask = torch.bernoulli(p).to(torch.bool)
    return edge_index[:, keep_mask], keep_mask



class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    



# Training
def train(args,g, feats, lr, epoch, device, lambda_loss, hidden_dim, sample_size=10,moving_average_decay=0.0, drop_prob=None):
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
    in_dim = feats.shape[1]
    print(feats.shape[0],feats.shape[1])
    GNNModel = Model(args,in_dim, hidden_dim, 2, sample_size, lambda_loss=lambda_loss,  moving_average_decay=moving_average_decay)
    GNNModel.to(device)
    opt = torch.optim.Adam([{'params': GNNModel.parameters()}], lr=lr, weight_decay=0.0003)
    with tqdm(total=epoch, desc="(T)") as pbar:
        for epocha in range(1, epoch + 1):
            feats = feats.to(device)
            if drop_prob is not None:
                edge_index = torch.stack(g.edges())
                sampled_edge_index, keep_mask = sample_edges_with_probs(edge_index, drop_prob)
                if keep_mask.any():
                    g_epoch = dgl.graph((sampled_edge_index[0], sampled_edge_index[1]), num_nodes=g.num_nodes())
                    g_epoch = g_epoch.to(device)
                else:
                    g_epoch = g
            else:
                g_epoch = g
            loss, node_embeddings = GNNModel(g_epoch, feats)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": loss.item()})
            pbar.update()
    return node_embeddings.cpu().detach(), loss.item()


def evaluate(model, loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    return (correct / total).item()

def write_results(acc, best_epoch):
    best_epoch = [str(tmp) for tmp in best_epoch]
    f = open("log_classification_heter/" + args.dataset, 'a+')
    f.write(args.dataset + ' --epochs ' + str(args.epoch_num) + ' --seed ' + str(args.seed) + ' --lr ' + str(args.lr) + ' --lambda_loss ' + str(args.lambda_loss) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(args.dimension) + ' --sample_size ' + str(args.sample_size) + ' --wd2 ' + str(args.wd2) + ' --num_MLP ' + str(args.num_MLP) + ' --tau ' + str(args.tau) + ' --best_epochs ' + " ".join(best_epoch) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f}\n')
    f.close()

def train_new_datasets(dataset_str,args, epoch_num = 10, lr=0, lambda_loss=1, sample_size=10, hidden_dim=None,moving_average_decay=0.0, edgedropscale=1.0):
    g, labels, split_lists = utils.read_real_datasets(dataset_str,args.root)
    g = g.to(device)
    node_features = g.ndata['attr']
    node_labels = labels
    # attr, feat
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim


    g,node_features,topk_idx, rest_idx = utils.select_topk_dims(g, node_features, args.dim_keep, args.choice)
    topk_idx = topk_idx.to(device)
    rest_idx = rest_idx.to(device)
    args.topk_idx = topk_idx
    args.rest_idx = rest_idx
    drop_prob = build_similarity_drop_matrix(node_features, rest_idx, scale=args.edgedropscale)
    acc = []
    epochs = []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    #pre-training
    node_embeddings, loss = train(args,g, node_features, lr=lr, epoch=epoch_num, device=device,
                                  lambda_loss=lambda_loss, sample_size=sample_size, hidden_dim=hidden_dim,
                                  moving_average_decay=moving_average_decay, drop_prob=drop_prob)
    #evaluation
    for index in range(10):
        input_dims = node_embeddings.shape
        # print(input_dims[1])
        class_number = int(max(node_labels)) + 1
        FNN = LogReg(input_dims[1], class_number).to(device)
        FNN = FNN.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(FNN.parameters(), lr =1e-2, weight_decay=args.wd2)
        dataset = NodeClassificationDataset(node_embeddings, node_labels)
        split = utils.DataSplit(dataset, split_lists[index]['train_idx'], split_lists[index]['valid_idx'], split_lists[index]['test_idx'], shuffle=True)
        train_loader, val_loader, test_loader = split.get_split(batch_size=100000, num_workers=0)
        best = -float('inf')
        best_epoch = 0
        test_acc = 0
        for epoch in range(3000):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_pred = FNN(inputs)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = FNN(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        total += labels.size(0)
                        correct += torch.sum(predicted == labels)
                if correct / total > best:
                    best = correct / total
                    test_acc = evaluate(FNN, test_loader)
                    best_epoch = epoch
       
        print(test_acc)
        acc.append(test_acc)
        epochs.append(best_epoch)
    print("mean:")
    print(statistics.mean(acc))
    print("std:")
    print(statistics.stdev(acc))
    write_results(acc, epochs)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="crocodile")
    parser.add_argument('--root', type=str, default="./datasets")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch_num', type=int, default=60)
    parser.add_argument('--lambda_loss', type=float, default=1)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=4096)
    parser.add_argument('--moving_average_decay', type=float, default=0.97)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--wd2', type=float, default=1e-05)
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2014)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of encoders")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of encoder layers")
    parser.add_argument("--use_ln", type=bool, default=True, help="Enable layer norm in encoders")
    parser.add_argument("--activation", type=str, default="relu", help="Activation type for encoders") 
    parser.add_argument("--embed_dropout_prob", type=float, default=0.1, help="Dropout rate in encoder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    #propogate_dim
    parser.add_argument("--dim_keep", type=int, default=256, help="Number of top feature dimensions to route to GCN")
    parser.add_argument("--choice", type=str, default="high", help="Feature selection mode: high/low/random")
    parser.add_argument("--edgedropscale", type=float, default=0.4)

    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    dataset_str = args.dataset
    train_new_datasets(dataset_str=dataset_str, args=args,lr=args.lr, epoch_num=args.epoch_num, lambda_loss=args.lambda_loss, sample_size=args.sample_size, hidden_dim=args.dimension,moving_average_decay=args.moving_average_decay,edgedropscale=args.edgedropscale)
