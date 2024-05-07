import torch
from torch.nn import Sigmoid, Dropout
import torch_geometric
from torch_geometric.nn import GCNConv, Linear, global_mean_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Optional
from utils import label


def new_global_mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor],
                     size: Optional[int] = None) -> torch.Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.MeanAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
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
        #preds = torch.sum(preds)#, axis=2)
        labels = label(preds)

        return preds, labels
