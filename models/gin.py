import torch
import torch_geometric
from torch_geometric.nn import GINConv, GCNConv, Linear, global_mean_pool, SAGEConv, JumpingKnowledge, global_add_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Optional
from utility import label
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

class GINModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=5):
        super(GINModel, self).__init__()
        self.gin_convs = torch.nn.ModuleList()

        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.gin_convs.append(GINConv(
                torch.nn.Sequential(
                    Linear2(layer_input_dim, hidden_dim),
                    ReLU(),
                    Linear2(hidden_dim, hidden_dim),
                    ReLU(),
                    BatchNorm1d(hidden_dim),
                    torch.nn.Dropout(0.5)
                )
            , node_dim=0))

        self.out_proj = Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch):
        # input is Nx1xM
        # output should be NxMx1
        import pdb; pdb.set_trace()
        for gin_conv in self.gin_convs:
            x = gin_conv(x, edge_index)
        x = new_global_mean_pool(x, torch.zeros(len(x), dtype=torch.int64).to('cuda'))  
        # x is 1x64x10 (1xHxM) -->   
        x = x.permute(0,2,1)
        # Input should be 10x1x64?
        x = self.out_proj(x)
        x = self.sigmoid(x)

        return x
    
    def evaluate(self, x, edge_index, batch):
        with torch.no_grad():
            self.eval()
            preds = self.forward(x=x, edge_index=edge_index, batch=batch)
            labels = label(preds)  # Assuming binary classification
            return preds, labels