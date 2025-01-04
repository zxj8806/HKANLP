# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform

import args

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias


def positional_encoding(t, T, d_model, device):
    position = torch.arange(0, d_model, dtype=torch.float, device=device).unsqueeze(0)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(1, d_model, device=device)
    pe[:, 0::2] = torch.sin(position[:, 0::2] * div_term)
    pe[:, 1::2] = torch.cos(position[:, 1::2] * div_term)
    return pe


class GraphKGNN(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphKGNN, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = x + torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class SingleVarFunction(nn.Module):
    def __init__(self, hidden_dim=16):
        super(SingleVarFunction, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Use ReLU instead of tanh
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        out = out.squeeze(-1)
        return out


class GraphHKAN(nn.Module):
    def __init__(self, adj, device):
        super(GraphHKAN, self).__init__()
        self.n = args.n + 1
        self.gcns_a1 = nn.ModuleList([
            GraphKGNN(args.input_dim, args.hidden1_dim, adj) for _ in range(self.n)
        ])
        self.gcns_a2 = nn.ModuleList([
            GraphKGNN(args.hidden1_dim, args.hidden2_dim, adj) for _ in range(self.n)
        ])
        self.gcn_mean = GraphKGNN(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_concentration = GraphKGNN(args.hidden1_dim, 1, adj, activation=lambda x: F.softplus(x) + 1)
        self.gcns_concentration = nn.ModuleList([
            GraphKGNN(args.hidden1_dim, 1, adj, activation=lambda x: F.softplus(x) + 1)
            for _ in range(self.n)
        ])

        self.d = args.hidden2_dim

        self.psi_functions = nn.ModuleList([
            nn.ModuleList([
                SingleVarFunction(hidden_dim=16) for _ in range(self.d)
            ])
            for _ in range(self.n)
        ])

        self.varphi_functions = nn.ModuleList([
            SingleVarFunction(hidden_dim=16) for _ in range(self.n)
        ])

        self.device = device

    def encode(self, X, t):
        X = X.to(self.device)  # Ensure X is on the correct device before processing
        hidden = self.gcns_a1[t](X)
        mean = self.gcns_a2[t](hidden)
        mean = F.normalize(mean, p=2, dim=1)  # Normalize to ensure it lies on the unit sphere
        concentration = self.gcns_concentration[t](hidden)
        return mean.to(self.device), concentration.to(self.device)

    def reparameterize(self, mean, concentration):
        mean = mean.to(self.device)
        concentration = concentration.to(self.device)
        q_z = VonMisesFisher(mean, concentration)  # Create distribution with parameters on the correct device
        p_z = HypersphericalUniform(mean.size(1) - 1,
                                    device=self.device)  # Explicitly set the device for the distribution
        return q_z, p_z

    def forward(self, X):
        if X.is_sparse:
            X_global_sparse = X.sum(dim=0)
            X_global = X_global_sparse.to_dense()
            X_global = X_global / X.size(0)  # Compute mean
        else:
            X_global = X.mean(dim=0)

        predicted_adj = torch.zeros(X.size(0), X.size(0), device=X.device)

        q_z_list = []
        p_z_list = []

        for t in range(self.n):
            mean, concentration = self.encode(X, t)
            q_z, p_z = self.reparameterize(mean, concentration)

            sampled_z = q_z.rsample()
            mean = F.normalize(mean, dim=1)
            perturbed_mean = mean + args.sample * sampled_z
            perturbed_mean = F.normalize(perturbed_mean, dim=1)

            adj_t = torch.mm(perturbed_mean, perturbed_mean.t())

            # S_t(X) = sum_{s=1}^{d} psi_{t,s}(X_global_s)
            S_t = 0
            for s in range(self.d):
                S_t += self.psi_functions[t][s](X_global[s])

            f_t = self.varphi_functions[t](S_t) + 1
            predicted_adj += f_t * adj_t

            q_z_list.append(q_z)
            p_z_list.append(p_z)

        predicted_adj /= self.n
        return predicted_adj, q_z_list, p_z_list
