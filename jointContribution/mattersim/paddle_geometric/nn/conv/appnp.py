from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from paddle_geometric.utils import is_paddle_sparse_tensor, spmm, to_edge_index
from paddle_geometric.utils.sparse import set_sparse_value


class APPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper.
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.shape[self.node_dim], False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.shape[self.node_dim], False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    if is_paddle_sparse_tensor(edge_index):
                        _, edge_weight = to_edge_index(edge_index)
                        edge_weight = F.dropout(edge_weight, p=self.dropout)
                        edge_index = set_sparse_value(edge_index, edge_weight)
                    else:
                        assert edge_weight is not None
                        edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.unsqueeze(-1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'
