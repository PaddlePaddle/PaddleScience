from typing import Optional, Tuple

import paddle
from paddle import Tensor
from paddle.nn import Layer
from paddle.nn import ParameterList, Linear
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.utils import get_laplacian


class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = ParameterList([
            Linear(in_channels, out_channels, bias_attr=False)
            for _ in range(K)
        ])

        if bias:
            self.bias = self.create_parameter(shape=[out_channels], is_bias=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bias is not None:
            paddle.nn.initializer.Constant(0.0)(self.bias)

    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: Optional[Tensor],
        normalization: Optional[str],
        lambda_max: Optional[Tensor] = None,
        dtype: Optional[str] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * paddle.max(edge_weight)
        elif not isinstance(lambda_max, Tensor):
            lambda_max = paddle.to_tensor(lambda_max, dtype=dtype)

        if batch is not None and lambda_max.shape[0] > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight = paddle.where(edge_weight == float('inf'), paddle.zeros_like(edge_weight), edge_weight)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight = paddle.where(loop_mask, edge_weight - 1, edge_weight)

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        lambda_max: Optional[Tensor] = None,
    ) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.shape[0],
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
