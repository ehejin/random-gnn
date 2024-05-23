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
        super().__init__()
        self.random = random
        aggregators = ['mean', 'min', 'max', 'std']  
        scalers = ['identity', 'amplification', 'attenuation']  
        
        #self.node_emb = Embedding(max_degree + 1, 75)
        #self.edge_emb = Embedding(4, 50)  
        #self.feature_emb = torch.nn.Linear(1, 75)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=1, out_channels=50,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=None, towers=2, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(torch.nn.BatchNorm1d(50)) 
            self.dropouts.append(torch.nn.Dropout(0.5))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.ReLU(), 
            torch.nn.Linear(50, 25), torch.nn.ReLU(),
            torch.nn.Linear(25, 1))
    
    def forward(self, x, edge_index, batch):
        #x = self.feature_emb(x)  
        #edge_attr = self.edge_emb(edge_attr)  

        for conv, batch_norm, dropout in zip(self.convs, self.batch_norms, self.dropouts):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = dropout(x)

        x = global_add_pool(x, batch)
        return self.mlp(x)

    def evaluate(self, x, edge_index, batch):
        preds = self.forward(x=x, edge_index=edge_index, batch=batch)
        #labels = label(preds)  # Define 'label' function or replace with appropriate logic

        return preds

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
