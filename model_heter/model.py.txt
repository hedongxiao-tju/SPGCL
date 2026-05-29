from layers import  MLP_generator, PairNorm, Encoder_Dropout
import torch
import torch.nn as nn
from dgl.nn import  GraphConv
import random
import copy
from torch.nn.functional import normalize
import numpy as np
import torch.nn.functional as F


def contrastive_loss(projected_emd, v_emd, sampled_embeddings_u, sampled_embeddings_neg_v, tau, lambda_loss):
    pos = torch.exp(torch.bmm(projected_emd, sampled_embeddings_u.transpose(-1, -2)).squeeze()/tau)
    neg_score = torch.log(pos + torch.sum(torch.exp(torch.bmm(v_emd, sampled_embeddings_neg_v.transpose(-1, -2)).squeeze()/tau), dim=1).unsqueeze(-1))
    neg_score = torch.sum(neg_score, dim=1)
    pos_socre = torch.sum(torch.log(pos), dim=1)
    total_loss = torch.sum(lambda_loss * neg_score - pos_socre)
    total_loss = total_loss/sampled_embeddings_u.shape[0]/sampled_embeddings_u.shape[1]
    return total_loss

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Model(nn.Module):
    def __init__(self, args,in_dim, hidden_dim, layer_num, sample_size, norm_mode="PN-SCS", norm_scale=20, lambda_loss=1,moving_average_decay=0.0):
        super(Model, self).__init__()
        self.device = torch.device(args.device)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss = lambda_loss
        self.tau = args.tau
        self.register_buffer("topk_idx", args.topk_idx)
        self.register_buffer("rest_idx", args.rest_idx)
        self.register_buffer("eps", torch.tensor(1e-6, device=self.device))
        
        # GNN Encoder
        self.graphconv = Encoder_Dropout(in_dim, args)
        self.target_graphconv = copy.deepcopy(self.graphconv)
        set_requires_grad(self.target_graphconv, True)

        

        self.target_ema_updater = EMA(moving_average_decay)
        self.num_MLP = args.num_MLP
        self.projector = MLP_generator(hidden_dim, hidden_dim, self.num_MLP)

        self.in_dim = in_dim
        self.sample_size = sample_size

    def update_moving_average(self):
        assert self.target_graphconv is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_graphconv, self.graphconv)

    def encode(self,g,x,target_encoder=False):
        x_top = torch.zeros_like(x)
        x_top[:, self.topk_idx] = x[:, self.topk_idx]
        if target_encoder:
            z_top = self.graphconv(g, x_top) 
            if self.rest_idx.numel() > 0:
                x_rest = torch.zeros_like(x)
                x_rest[:, self.rest_idx] = x[:, self.rest_idx]
                z_rest = self.graphconv.forward_linear(x_rest)
                z = z_top + z_rest
            else:
                z = z_top
        else:
            z_top = self.target_graphconv(g, x_top) 
            if self.rest_idx.numel() > 0:
                x_rest = torch.zeros_like(x)
                x_rest[:, self.rest_idx] = x[:, self.rest_idx]
                z_rest = self.target_graphconv.forward_linear(x_rest)
                z = z_top + z_rest
            else:
                z = z_top
        return z
    
    
    # def forward(self, g, h, neighbor_dict, device):
    #     v_emd, u_emd, projected_emd = self.forward_encoder(g, h)
    #     loss = self.neighbor_decoder(v_emd, u_emd, projected_emd,neighbor_dict)
    #     return loss, v_emd

    def forward(self,g,h):
        z = self.encode(g,h)
        loss = self.loss(z,g)
        return loss,z

    def loss(self, z, g):
        h = F.normalize(self.projector(z), dim=-1)
        N = h.size(0)
        device = h.device
    
        adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        src, dst = g.edges()
        adj[src, dst] = True
    
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = adj
    
        sim = torch.mm(h, h.t()) / self.tau
        exp_sim = torch.exp(sim)
    
        pos = (exp_sim * pos_mask).sum(dim=1) + self.eps
        neg = (exp_sim * (~pos_mask) * (~eye_mask)).sum(dim=1) + self.eps
        denom = pos + neg
    
        loss = -torch.log(pos / denom)
        return loss.mean()