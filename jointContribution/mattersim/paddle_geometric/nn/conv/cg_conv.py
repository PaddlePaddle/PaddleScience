from typing import Tuple, Union, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import BatchNorm1D, Linear

from paddle_geometric.nn.conv import MessagePassing


class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    "Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties" paper.
    """

    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias_attr=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias_attr=bias)

        if batch_norm:
            self.bn = BatchNorm1D(channels[1])
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: Tuple[Tensor, Tensor], edge_attr: Optional[Tensor])
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.bn is not None:
            out = self.bn(out)

        out = out + x[1]

        return out

    def message(self, x_i, x_j, edge_attr: Optional[Tensor]) -> Tensor:
        if edge_attr is None:
            z = paddle.concat([x_i, x_j], axis=-1)
        else:
            z = paddle.concat([x_i, x_j, edge_attr], axis=-1)

        return F.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'
