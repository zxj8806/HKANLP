
# set CUDA_LAUNCH_BLOCKING=1
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import torch.nn as nn

import math
from input_data import load_data
from preprocessing import *
import args

from tqdm.auto import tqdm
from model import GraphHKAN

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
import sys
import os
# Disable buffering of stdout
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
#import model

# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

adj, features = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)

adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)


adj = adj_train


# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                    torch.FloatTensor(adj_norm[1]),
                                    torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                     torch.FloatTensor(adj_label[1]),
                                     torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                    torch.FloatTensor(features[1]),
                                    torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight




REDUCED_DIM = 1


def D_losses(denoise_model, adj_label, features, noise=None, loss_type="huber"):
    predicted_adj_t, q_z_list, p_z_list = denoise_model(features.to(device))

    if predicted_adj_t.device != device:
        predicted_adj_t = predicted_adj_t.to(device)
    
    kl_divergence = 0
    for q_z, p_z in zip(q_z_list, p_z_list):
        kl_divergence += torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    kl_divergence /= args.n 

    # Calculate total loss
    if loss_type == 'L1':
        total_loss = F.l1_loss(adj_label.to(device), predicted_adj_t)
    elif loss_type == 'L2':
        total_loss = F.mse_loss(adj_label.to(device), predicted_adj_t)
    elif loss_type == 'huber':
        predicted_adj_t = torch.clamp(predicted_adj_t, 0, 1)
        total_loss = F.binary_cross_entropy(predicted_adj_t.view(-1), adj_label.to(device).to_dense().view(-1), weight=weight_tensor.to(device))
    else:
        raise NotImplementedError("Loss type not implemented")
    
    total_loss += args.sample*kl_divergence  # Include the averaged KL divergence in the total loss

    return total_loss, predicted_adj_t


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)



def positional_encoding(t, T, d_model, device):
    position = torch.arange(0, d_model, dtype=torch.float, device=device).unsqueeze(0)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(1, d_model, device=device)
    pe[:, 0::2] = torch.sin(position[:, 0::2] * div_term)
    pe[:, 1::2] = torch.cos(position[:, 1::2] * div_term)
    return pe

    
def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


@torch.no_grad()
def sample(model, features, adj_label, device):

    predicted_adj_t, q_z_list, p_z_list = model(features)


    if predicted_adj_t.device != device:
        predicted_adj_t = predicted_adj_t.to(device)

    if predicted_adj_t.is_sparse:
        predicted_adj_t = predicted_adj_t.to_dense()
    predicted_adj_t = torch.clamp(predicted_adj_t, 0, 1)

    return predicted_adj_t

print("torch.cuda.is_available()", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

adj_norm = adj_norm.to(device)

model = GraphHKAN(adj_norm, device)
model.to(device)
features = features.to(device)
adj_label = adj_label.to(device)
weight_tensor = weight_tensor.to(device)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
epochs = args.num_epoch

for epoch in range(epochs):
    optimizer.zero_grad()


    loss, A_pred = D_losses(model, adj_label, features, loss_type="huber")

    loss.backward()
    optimizer.step()
    if epoch < 10000:

        train_acc = get_acc(A_pred, adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap))

    else:
        print(f"Epoch: {epoch + 1}, Time step: {t.item()}, Loss: {loss.item()}")

A_pred = sample(model, features, adj_label, device)
test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

#torch.save(model, './model.pth')






