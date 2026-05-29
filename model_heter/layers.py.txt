import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP with linear outputs (without softmax)

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(MLP_generator, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers
        # self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        neighbor_embedding = self.linears[self.num_layers - 1](h)
        # neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding



class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=10):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale


    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x

# FNN
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        return x


import dgl
import dgl.nn as dglnn

class Encoder_Dropout(nn.Module):
    def __init__(self, in_channels: int, args):
        super(Encoder_Dropout, self).__init__()

        hidden_dim = args.hidden_dim
        out_channels = args.dimension
        p_drop = args.embed_dropout_prob
        self.num_layer = args.num_layers
        self.args = args
        self.activation = {"relu": F.relu, "prelu": nn.PReLU(), "rrelu": F.rrelu, "elu": F.elu}[args.activation]
        self.factor = 2

        self.convs = nn.ModuleList()
        self.drops = nn.ModuleList()
        if getattr(args, 'use_ln', False):
            self.lns = nn.ModuleList()

        if self.num_layer >= 2:
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(dglnn.GraphConv(in_channels, self.factor * hidden_dim, activation=None, norm='both', allow_zero_in_degree=True))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(self.factor * hidden_dim))
            for _ in range(1, self.num_layer - 1):
                self.drops.append(nn.Dropout(p=p_drop))
                self.convs.append(dglnn.GraphConv(self.factor * hidden_dim, self.factor * hidden_dim, activation=None, norm='both', allow_zero_in_degree=True))
                if args.use_ln:
                    self.lns.append(nn.LayerNorm(self.factor * hidden_dim))
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(dglnn.GraphConv(self.factor * hidden_dim, out_channels, activation=None, norm='both', allow_zero_in_degree=True))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))
        else:
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(dglnn.GraphConv(in_channels, out_channels, activation=None, norm='both', allow_zero_in_degree=True))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        for i in range(self.num_layer):
            x = self.drops[i](x)
            x = self.convs[i](g, x)
            x = self.activation(x)
            if self.args.use_ln:
                x = self.lns[i](x)
        return x

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layer):
            x = self.drops[i](x)
            weight = self.convs[i].weight.T  # (out_feats, in_feats)
            bias = self.convs[i].bias      # (out_feats,)
            x = F.linear(x, weight, bias)
            x = self.activation(x)
            if self.args.use_ln:
                x = self.lns[i](x)
        return x


