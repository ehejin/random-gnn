import torch
import torch_geometric
from torch_geometric.nn import GINConv, GCNConv, Linear, global_mean_pool, SAGEConv, JumpingKnowledge, global_add_pool, PNAConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Optional
from utils import label
from torch.nn import Linear, Sigmoid, Dropout, Module, Sequential, BatchNorm1d, ReLU
from torch_geometric.utils import dropout_adj

from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from models.gnn import new_global_mean_pool

class Linear2(Linear):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input features.
        """
        # x is shape Nx1xM
        N, B, M = x.shape
        x = x.view(N*M, B)
        c = F.linear(x, self.weight, self.bias)
        c = c.view(N, -1, M)
        return c


class PNA(torch.nn.Module):
    def __init__(self, random, deg):
        super(PNA, self).__init__()
        self.random = random
        aggregators = ['mean', 'min', 'max', 'std']  # Define aggregators
        scalers = ['identity', 'amplification', 'attenuation']  # Define scalers

        if self.random:
            self.conv = PNAConv(1, 50, aggregators, scalers, deg=deg, edge_dim=1)
            self.conv2 = PNAConv(50, 50, aggregators, scalers, deg=deg, edge_dim=1)
        else:
            self.conv = PNAConv(1, 50, aggregators, scalers, deg=deg)
            self.conv2 = PNAConv(50, 50, aggregators, scalers, deg=deg)

        self.lin = Linear(50, 1)
        self.pred = Sigmoid()
        self.drop = Dropout(0.5)
        self.relu = ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.conv(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        x = new_global_mean_pool(x, batch)

        x = self.drop(x)
        x = self.lin(x)
        x = self.pred(x).flatten()

        return x
    
    def evaluate(self, x, edge_index, batch):
        preds = self.forward(x=x, edge_index=edge_index, batch=batch)
        labels = label(preds)  # Define 'label' function or replace with appropriate logic

        return preds, labels

class PNAGIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_dim):
        super(PNAGIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        deg = torch.tensor([...])  # appropriate degree statistics
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        for i in range(num_layers):
            in_channels = num_features if i == 0 else hidden_dim
            self.layers.append(
                PNAConv(in_channels, hidden_dim, aggregators, scalers, deg=deg)
            )
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear2(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear2(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.nn.functional.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x
