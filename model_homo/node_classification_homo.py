import argparse
import os
import random
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from torch_geometric.datasets import (Amazon, CitationFull, Coauthor, Planetoid,
                                      WebKB, WikiCS, WikipediaNetwork, PPI)
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, dropout_edge, to_torch_coo_tensor, mask_feature
from tqdm import tqdm
from itertools import product

def load_dataset(dataset_name: str, dataset_dir: str):
    print("Dataloader: Loading Dataset", dataset_name)
    assert dataset_name in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "dblp",
        "Photo",
        "Computers",
        "CS",
        "Physics",
        "ogbn-products",
        "ogbn-arxiv",
        "Wiki",
        "ppi",
        "Cornell",
        "Texas",
        "Wisconsin",
        "chameleon",
        "crocodile",
        "squirrel",
    ]

    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == "dblp":
        dataset = CitationFull(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["Photo", "Computers"]:
        dataset = Amazon(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["CS", "Physics"]:
        dataset = Coauthor(dataset_dir, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["Wiki"]:
        dataset = WikiCS(dataset_dir, transform=T.NormalizeFeatures())
    elif dataset_name in ["ppi"]:
        train = PPI(root=dataset_dir, split="train", transform=T.NormalizeFeatures())
        val = PPI(root=dataset_dir, split="val", transform=T.NormalizeFeatures())
        test = PPI(root=dataset_dir, split="test", transform=T.NormalizeFeatures())
        dataset = [train, val, test]
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        return WebKB(dataset_dir, dataset_name)
    elif dataset_name in ["chameleon", "crocodile"]:
        return WikipediaNetwork(dataset_dir, dataset_name)
    elif dataset_name in ["squirrel"]:
        return WikipediaNetwork(dataset_dir, dataset_name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    print("Dataloader: Loading success.")
    print(dataset[0])
    return dataset

def edgeindex2adj(edge_index, num_nodes):
    adj_shape = (num_nodes, num_nodes)
    edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
    adj = to_torch_coo_tensor(edge_index, size=adj_shape)
    return adj


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, data_random_seed: int = 1, repeat: int = 3):
    one_hot_encoder = OneHotEncoder(categories="auto")

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    X = normalize(X, norm="l2")
    rng = np.random.RandomState(data_random_seed)
    accuracies: List[float] = []
    for _ in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        logreg = LogisticRegression(solver="liblinear")
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=5, cv=cv, verbose=0
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies

def select_topk_dims(data, k: int, choice: str) -> Tuple[torch_geometric.data.Data, torch.Tensor, torch.Tensor]:
    x = data.x
    edge_index = data.edge_index
    num_nodes, num_features = x.size()

    # Compute node degrees
    deg = torch.zeros(num_nodes, device=x.device)
    deg[edge_index[0]] += 1
    deg[edge_index[1]] += 1
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes

    # Extract edge endpoints
    row, col = edge_index
    x_i = x[row]  # (E, F)
    x_j = x[col]  # (E, F)

    # Degree-normalized features
    x_i_norm = x_i * deg_inv_sqrt[row].unsqueeze(1)  # (E, F)
    x_j_norm = x_j * deg_inv_sqrt[col].unsqueeze(1)  # (E, F)

    # Normalized squared difference per edge per feature
    diff_sq_norm = (x_i_norm - x_j_norm) ** 2  # (E, F)

    # Total variation approximation: mean over edges (unbiased estimator of x^T L x)
    scores = diff_sq_norm.mean(dim=0)  # (F,)

    # Select top-k
    k_select = min(k, num_features)
    if choice == "high":
        topk_idx = torch.topk(scores, k=k_select, largest=True).indices
    elif choice == "low":
        topk_idx = torch.topk(scores, k=k_select, largest=False).indices
    else:
        topk_idx = torch.randperm(num_features, device=x.device)[:k_select]
    topk_idx, _ = torch.sort(topk_idx)

    all_idx = torch.arange(num_features, device=x.device)
    rest_idx = all_idx[~torch.isin(all_idx, topk_idx)]

    return data, topk_idx, rest_idx

def build_similarity_drop_matrix(x: torch.Tensor, topk_idx: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.size(0)
    drop_prob = torch.zeros((n, n), device=x.device)
    if topk_idx.numel() == 0:
        return drop_prob
    x_top = x[:, topk_idx]
    x_norm = F.normalize(x_top, p=2, dim=1)
    sim = torch.mm(x_norm, x_norm.t())  # cosine similarity in [-1, 1]
    drop_prob = sim.clamp(min=0.0, max=1.0) * scale  # keep in [0, 1]
    drop_prob = drop_prob.clamp(max=1.0)
    drop_prob.diagonal().fill_(1.0)
    return drop_prob

def sample_edges_with_probs(edge_index: torch.Tensor, drop_prob: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    p = drop_prob[row, col]
    keep_mask = torch.bernoulli(p).to(torch.bool)
    return edge_index[:, keep_mask], keep_mask

class Encoder_Dropout(torch.nn.Module):
    def __init__(self, in_channels: int, args):
        super(Encoder_Dropout, self).__init__()

        base_layer = GCNConv
        out_channels = args.hidden_dim
        p_drop = args.embed_dropout_prob
        self.num_layer = args.num_layers
        self.args = args
        self.activation = {"relu": F.relu, "prelu": nn.PReLU(), "rrelu": F.rrelu, "elu": F.elu}[args.activation]
        self.factor = 2
        self.convs = torch.nn.ModuleList()
        self.drops = nn.ModuleList()
        if args.use_ln:
            self.lns = nn.ModuleList()

        if self.num_layer >= 2:
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(base_layer(in_channels, self.factor * out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(self.factor * out_channels))
            for _ in range(1, self.num_layer - 1):
                self.drops.append(nn.Dropout(p=p_drop))
                self.convs.append(base_layer(self.factor * out_channels, self.factor * out_channels))
                if args.use_ln:
                    self.lns.append(nn.LayerNorm(self.factor * out_channels))
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(base_layer(self.factor * out_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))
        else:
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(base_layer(in_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.num_layer):
            x = self.drops[i](x)
            x = self.activation(self.convs[i](x, edge_index))
            if self.args.use_ln:
                x = self.lns[i](x)
        return x

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layer):
            x = self.drops[i](x)
            conv = self.convs[i]
            x = self.activation(F.linear(x, conv.lin.weight, conv.bias))
            if self.args.use_ln:
                x = self.lns[i](x)
        return x

class Model(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        topk_idx: torch.Tensor,
        rest_idx: torch.Tensor,
        args,
    ):
        super(Model, self).__init__()
        self.device = torch.device(args.device)
        self.args = args
        self.encoder = encoder
        self.register_buffer("topk_idx", topk_idx)
        self.register_buffer("rest_idx", rest_idx)

        self.fc1 = torch.nn.Linear(args.hidden_dim, args.proj_dim)
        self.activation = F.elu
        self.fc2 = torch.nn.Linear(args.proj_dim, args.hidden_dim)
        self.register_buffer("eps", torch.tensor(1e-6, device=self.device))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_top = torch.zeros_like(x)
        x_top[:, self.topk_idx] = x[:, self.topk_idx]
        z_top = self.encoder.forward(x_top, edge_index)
        if self.rest_idx.numel() > 0:
            x_rest = torch.zeros_like(x)
            x_rest[:, self.rest_idx] = x[:, self.rest_idx]
            z_rest = self.encoder.forward_linear(x_rest)
            z = z_top + z_rest
        else:
            z = z_top
        return z, z_top

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(z)))

    def loss(self, z, edge_index, batch_size: int = 0, epoch: int = 0):
        h = F.normalize(self.projection(z))
        adj = edgeindex2adj(edge_index, h.shape[0]).to_dense().bool()
        eye_mask = torch.eye(h.size(0), device=h.device, dtype=torch.bool)
        pos_mask = adj
        sim = torch.mm(h, h.t()) / self.args.temperature
        exp_sim = torch.exp(sim)
        pos = (exp_sim * pos_mask).sum(dim=1) + self.eps
        neg = (exp_sim * (~pos_mask) * (~eye_mask)).sum(dim=1) + self.eps
        denom = pos + neg
        loss = -torch.log(pos / denom)
        return loss.mean()

def build_masked_view(x: torch.Tensor, topk_idx: torch.Tensor, pf: float) -> torch.Tensor:
    x_masked = x.clone()
    masked_slice, _ = mask_feature(x_masked[:, topk_idx], p=pf, mode="all")
    x_masked[:, topk_idx] = masked_slice
    return x_masked

def train(model: Model, data, optimizer, edge_index, topk_idx, args):
    model.train()
    optimizer.zero_grad()
    x = build_masked_view(data.x, topk_idx, args.pf)
    z, z_top = model(x, edge_index)
    loss = model.loss(z, edge_index, args.batch_size)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_embeddings(model: Model, data, num_hop):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        z = out[0]
        if num_hop != 0:
            a = DAD_edge_index(data.edge_index, (z.size()[0], z.size()[0]))
            for i in range(num_hop):
                z = a @ z
    scores = fit_logistic_regression(z.detach().cpu().numpy(), data.y.cpu().numpy())
    return float(np.mean(scores)), float(np.std(scores))

def DAD_edge_index(edge_index, size):
    a = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index)[0], size).to_dense().to(edge_index.device)
    a = a + torch.eye(n=a.size()[0]).to(edge_index.device)
    d = a.sum(dim=0)
    d_2 = torch.diag(d.pow(-0.5))
    a = d_2 @ a @ d_2
    return a

def main(args):
    device = torch.device(args.device)
    path = args.data_dir
    log_dir = args.log_dir
    with open(log_dir, "a") as f:
        f.write(str(args))
        f.write("\n")

    torch_geometric.seed.seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_means = []
    for i in range(args.runs):
        dataset = load_dataset(args.dataset_name, path)
        data = dataset[0].to(device)
        
        data, topk_idx, rest_idx = select_topk_dims(data, args.dim_keep, args.choice)
        topk_idx = topk_idx.to(device)
        rest_idx = rest_idx.to(device)
        drop_prob_matrix = build_similarity_drop_matrix(data.x, rest_idx, scale=args.neighbor_masking_prob)
    
        encoder = Encoder_Dropout(dataset.num_features, args).to(device)
        model = Model(encoder, topk_idx, rest_idx, args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        with tqdm(total=args.epochs, desc="(T)") as pbar:
            for epoch in range(1, args.epochs + 1):
                if epoch > args.edepoch:
                    edge_ind, keep_mask = sample_edges_with_probs(data.edge_index, drop_prob_matrix)
                    loss = train(model, data, optimizer, edge_ind, topk_idx, args)
                    
                    # loss = train(model, data, optimizer, data.edge_index, topk_idx, args)
                else:
                    loss = train(model, data, optimizer, data.edge_index, topk_idx, args)
                pbar.set_postfix({"loss": loss})
                pbar.update()
        m, n = evaluate_embeddings(model, data, args.numhop)
        print(m, n)
        run_means.append(m)
        with open(log_dir, "a") as f:
            f.write("mean: " + str(m)[0:7] + "std: " + str(n))
            f.write("\n")
            
    if len(run_means) > 1:
        avg_score = float(np.mean(run_means))
        print(f"Average score over {args.runs} runs: {avg_score}")
        with open(log_dir, "a") as f:
            f.write(f"Average score over {args.runs} runs: {avg_score}\n")
    
    with open(log_dir, "a") as f:
        f.write("\n")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser("FINAL_HOMO")
    parser.add_argument("--dataset_name", type=str, default="Photo", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="../dataset", help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="./log/random", help="Path to log")
  
    parser.add_argument("--dim_keep", type=int, default=256, help="Number of top feature dimensions to route to GCN")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of encoders")
    parser.add_argument("--proj_dim", type=int, default=256, help="Hidden dimension of projector head")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of encoder layers")
    
    parser.add_argument("--embed_dropout_prob", type=float, default=0.1, help="Dropout rate in encoder")
    parser.add_argument("--neighbor_masking_prob", type=float, default=0.4, help="Edge dropout probability")
    parser.add_argument("--pf", type=float, default=0.4, help="Feature mask probability")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    
    parser.add_argument("--use_ln", type=bool, default=True, help="Enable layer norm in encoders")
    parser.add_argument("--activation", type=str, default="prelu", help="Activation type for encoders") 
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for loss; 0 means full graph")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:6", help="Device")
    parser.add_argument("--choice", type=str, default="high", help="Feature selection mode: high/low/random")
    parser.add_argument("--numhop", type=int, default=0, help="num of hops")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for InfoNCE")
    parser.add_argument("--runs", type=int, default=1, help="run")
    parser.add_argument("--edepoch", type=int, default=300, help="epoch to start edge dropping")
    
    args = parser.parse_args()
    main(args)
