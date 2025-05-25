import math
from typing import Optional

import paddle
from paddle import Tensor
from paddle.nn import Layer, Linear
from paddle.nn.initializer import XavierUniform

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from paddle_geometric.utils import spmm


class GCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper.
    """

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = math.log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = self.create_parameter(
            shape=[channels, channels],
            default_initializer=XavierUniform()
        )

        if shared_weights:
            self.weight2 = None
        else:
            self.weight2 = self.create_parameter(
                shape=[channels, channels],
                default_initializer=XavierUniform()
            )

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.weight1)
        if self.weight2 is not None:
            paddle.nn.initializer.XavierUniform()(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.shape[0], False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.shape[0], False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        x *= (1 - self.alpha)
        x_0 = self.alpha * x_0[:x.shape[0]]

        if self.weight2 is None:
            out = x + x_0
            out = paddle.addmm(out, out, self.weight1, beta=1. - self.beta,
                               alpha=self.beta)
        else:
            out = paddle.addmm(x, x, self.weight1, beta=1. - self.beta,
                               alpha=self.beta)
            out += paddle.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                                alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.unsqueeze(-1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'alpha={self.alpha}, beta={self.beta})')
