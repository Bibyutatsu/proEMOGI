
import torch

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import io
import matplotlib.pyplot as plt
import math
import numpy as np


class proEMOGI(torch.nn.Module):
    """EMOGI model. A GCN with 3D graph convolutions and weighted loss.

    This class implements the EMOGI model. It is derived from the GCN
    model but contains some different metrics for logging (AUPR and AUROC
    for binary classification settings), a weighted loss function for
    imbalanced class sizes (eg. more negatives than positives) and
    the support for 3D graph convolutions (third dimension is treated
    similarly to channels in rgb images).
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_hidden_layers=2,
                 dropout_rate=0.5,
                 hidden_dims=[20, 40],
                 pos_loss_multiplier=1,
                 weight_decay=5e-4,
                 **kwargs):
        super(proEMOGI, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # model params
        self.weight_decay = weight_decay
        self.pos_loss_multiplier = pos_loss_multiplier
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        self.layers = []

        # add intermediate layers
        inp_dim = self.input_dim
        for l in range(self.num_hidden_layers):
            self.layers.append(GCNConv(inp_dim,
                                       self.hidden_dims[l]))
            inp_dim = self.hidden_dims[l]
            
        self.layers.append(GCNConv(self.hidden_dims[-1],
                                    self.output_dim))
        
    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            if self.dropout_rate is not None:
                x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Model saved in file: %s" % path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, x, edge_index, edge_weight=None):
        logits = self(x, edge_index, edge_weight)
        pred = logits.max(1)[1]
        return pred
