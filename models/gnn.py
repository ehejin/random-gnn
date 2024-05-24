import torch
import torch_geometric
from torch_geometric.nn import GINConv, GCNConv, Linear, global_mean_pool, SAGEConv, JumpingKnowledge, global_add_pool
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

def new_global_mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor],
                     size: Optional[int] = None) -> torch.Tensor:
    #dim = -1 if x.dim() == 1 else -2
    dim = 0

    if batch is None:
        return x.mean(dim=dim, keepdim=x.dim() <= 2)
    size = int(batch.max().item() + 1) if size is None else size
    return torch_geometric.utils.scatter(x, batch, dim=dim, dim_size=size, reduce='mean')

class GCNConv2(GCNConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
    def forward(self, x, edge_index, edge_weight: Optional[torch.Tensor] = None, batch=None):
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, torch.sparse.Tensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # EDIT: reshape to x
        M,N = x.shape[2], x.shape[0]
        in_x = x.view(x.shape[0] * x.shape[2], x.shape[1])
        x = self.lin(in_x)
        x = x.view(N, x.shape[1], M)
        #x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias[None,:,None]

        return out
    
    def message(self, x_j: torch.Tensor, edge_weight) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1, 1) * x_j

class GCN(torch.nn.Module):
    # output should be NxFxM
    # input is Nx1xM
    # h should be 1 x F? where F is hidden size
    def __init__(self, random):
        super().__init__()
        self.random = random
        if self.random:
            self.conv = GCNConv2(1, 50, node_dim=0)
            self.conv2 = GCNConv2(50, 50, node_dim=0)
        else:
            self.conv = GCNConv(1, 50) # Usually 136
            self.conv2 = GCNConv(50, 50)
        self.lin = Linear(50, 1)
        self.pred = Sigmoid()
        self.drop = Dropout(0.5)

    def forward(self, x, edge_index, batch=None):
        x = self.conv(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        if batch is not None:
            if self.random:
                x = new_global_mean_pool(x, batch)
            else:
                x = global_mean_pool(x, batch) 
        else:
            if self.random:
                x = new_global_mean_pool(x, torch.zeros(len(x), dtype=torch.int64))
            x = global_mean_pool(x, torch.zeros(len(x), dtype=torch.int64))

        x = self.drop(x) 
        # x is FxM
        if self.random:
            N, F, M = x.shape
            new_x = x.view(N * M, F)
            x = self.lin(new_x)
            x = x.view(N, M)
            x = self.pred(x).mean(axis=1)
        else:
            x = self.lin(x)
            x = self.pred(x).flatten()

        return x
    
    def evaluate(self, x, edge_index, batch):
        preds = self.forward(x=x, edge_index=edge_index, batch=batch) # this should be length N
        #labels = label(preds)

        return preds #, labels

class SAGEConv2(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        # self.lin_l is 1x64
        # out is 297x1x10

        M,N = out.shape[2], out.shape[0]
        in_x = out.view(out.shape[0] * out.shape[2], out.shape[1])
        out = self.lin_l(in_x)
        out = out.view(N, out.shape[1], M)

        #out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            M,N = x_r.shape[2], x_r.shape[0]
            in_xr = x_r.view(x_r.shape[0] * x_r.shape[2], x_r.shape[1])
            out = out + self.lin_r(in_xr).view(N, -1, M)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

class GraphSAGE(torch.nn.Module):
    def __init__(self, random):
        super().__init__()
        self.random = random
        in_channels = 1
        hidden_channels = 64
        out_channels = 1
        if self.random:
            self.conv1 = SAGEConv2(in_channels, hidden_channels, node_dim=0, aggr='mean')
            self.conv2 = SAGEConv2(hidden_channels, hidden_channels, node_dim=0, aggr='mean')
            self.conv3 = SAGEConv2(hidden_channels, hidden_channels, node_dim=0, aggr='mean')
        else:
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.lin = Linear(hidden_channels, out_channels)
        self.pred = Sigmoid()
        self.drop = Dropout(0.5)

    def forward(self, x, edge_index, batch=None):
        #edge_index, _ = dropout_adj(edge_index, p=0.1, force_undirected=True, num_nodes=len(x), training=self.training)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        if batch is not None:
            if self.random:
                x = new_global_mean_pool(x, batch)
            else:
                x = global_mean_pool(x, batch) 
        else:
            if self.random:
                x = new_global_mean_pool(x, torch.zeros(len(x), dtype=torch.int64))
            x = global_mean_pool(x, torch.zeros(len(x), dtype=torch.int64))

        '''x = self.drop(x)
        x = self.lin(x)
        x = self.pred(x).squeeze()

        return x'''
        x = self.drop(x) 
        if self.random:
            N, F, M = x.shape
            new_x = x.view(N * M, F)
            x = self.lin(new_x)
            x = x.view(N, M)
            x = self.pred(x).mean(axis=1)
        else:
            x = self.lin(x)
            x = self.pred(x).flatten()

        return x

    def evaluate(self, x, edge_index, batch):
        with torch.no_grad():
            self.eval()
            preds = self.forward(x=x, edge_index=edge_index, batch=batch)
            labels = label(preds)  # Assuming binary classification
            return preds, labels
