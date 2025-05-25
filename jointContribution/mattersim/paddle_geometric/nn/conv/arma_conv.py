from typing import Callable, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, ReLU
from paddle.nn.initializer import XavierUniform

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.utils import spmm
from paddle_geometric.typing import Adj, OptTensor, SparseTensor


class ARMAConv(MessagePassing):
    r"""The ARMA graph convolutional operator from the "Graph Neural Networks
    with Convolutional ARMA Filters" paper.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_stacks: int = 1, num_layers: int = 1,
                 shared_weights: bool = False,
                 act: Optional[Callable] = ReLU(), dropout: float = 0.,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.act = act
        self.shared_weights = shared_weights
        self.dropout = dropout

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        T = 1 if self.shared_weights else T

        self.weight = self.create_parameter(
            [max(1, T - 1), K, F_out, F_out], default_initializer=XavierUniform())

        if in_channels > 0:
            self.init_weight = self.create_parameter(
                [K, F_in, F_out], default_initializer=XavierUniform())
            self.root_weight = self.create_parameter(
                [T, K, F_in, F_out], default_initializer=XavierUniform())
        else:
            raise ValueError("in_channels must be greater than 0.")

        if bias:
            self.bias = self.create_parameter([T, K, 1, F_out], is_bias=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        XavierUniform()(self.weight)
        XavierUniform()(self.init_weight)
        XavierUniform()(self.root_weight)
        if self.bias is not None:
            paddle.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: Optional[Tensor] = None) -> Tensor:

        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.shape[self.node_dim],
            add_self_loops=False, dtype=x.dtype)

        x = x.unsqueeze(-3)
        out = x
        for t in range(self.num_layers):
            if t == 0:
                out = paddle.matmul(out, self.init_weight)
            else:
                out = paddle.matmul(out, self.weight[0 if self.shared_weights else t - 1])

            out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

            root = F.dropout(x, p=self.dropout, training=self.training)
            root = paddle.matmul(root, self.root_weight[0 if self.shared_weights else t])
            out = out + root

            if self.bias is not None:
                out = out + self.bias[0 if self.shared_weights else t]

            if self.act is not None:
                out = self.act(out)

        return out.mean(axis=-3)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.unsqueeze(-1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_stacks={self.num_stacks}, '
                f'num_layers={self.num_layers})')
