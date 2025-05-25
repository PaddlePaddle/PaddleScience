from typing import Optional

from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, BatchNorm1D, Linear

from paddle_geometric.nn import SimpleConv
from paddle_geometric.nn.dense.linear import Linear as PaddleLinear


class PMLP(Layer):
    r"""The P(ropagational)MLP model from the `"Graph Neural Networks are
    Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"
    <https://arxiv.org/abs/2212.09034>`_ paper.
    :class:`PMLP` is identical to a standard MLP during training, but then
    adopts a GNN architecture during testing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): The number of layers.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        norm (bool, optional): If set to :obj:`False`, will not apply batch
            normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the module
            will not learn additive biases. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.,
        norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias

        self.lins = paddle.nn.LayerList()
        self.lins.append(PaddleLinear(in_channels, hidden_channels, self.bias))
        for _ in range(self.num_layers - 2):
            lin = PaddleLinear(hidden_channels, hidden_channels, self.bias)
            self.lins.append(lin)
        self.lins.append(PaddleLinear(hidden_channels, out_channels, self.bias))

        self.norm = None
        if norm:
            self.norm = BatchNorm1D(
                hidden_channels,
                weight_attr=False,
                bias_attr=False,
            )

        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            paddle.nn.initializer.XavierUniform()(lin.weight)
            if self.bias:
                paddle.nn.initializer.Constant(0.)(lin.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """"""  # noqa: D419
        if not self.training and edge_index is None:
            raise ValueError(f"'edge_index' needs to be present during "
                             f"inference in '{self.__class__.__name__}'")

        for i in range(self.num_layers):
            x = paddle.matmul(x, self.lins[i].weight.T)
            if not self.training:
                x = self.conv(x, edge_index)
            if self.bias:
                x = x + self.lins[i].bias
            if i != self.num_layers - 1:
                if self.norm is not None:
                    x = self.norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
