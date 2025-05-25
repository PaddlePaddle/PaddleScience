import math
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, Linear

class DenseGATConv(Layer):
    r"""See :class:`paddle_geometric.nn.conv.GATConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * out_channels, bias_attr=False)

        # Learnable parameters for attention coefficients:
        self.att_src = self.create_parameter([1, 1, heads, out_channels])
        self.att_dst = self.create_parameter([1, 1, heads, out_channels])

        if bias and concat:
            self.bias = self.create_parameter([heads * out_channels])
        elif bias and not concat:
            self.bias = self.create_parameter([out_channels])
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Glorot (Xavier) initialization:
        fan_in, fan_out = self.in_channels, self.heads * self.out_channels
        bound = math.sqrt(6 / (fan_in + fan_out))
        paddle.nn.initializer.Uniform(-bound, bound)(self.lin.weight)
        paddle.nn.initializer.Uniform(-bound, bound)(self.att_src)
        paddle.nn.initializer.Uniform(-bound, bound)(self.att_dst)
        if self.bias is not None:
            paddle.nn.initializer.Constant(0)(self.bias)

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None,
                add_loop: bool = True) -> Tensor:
        x = x.unsqueeze(0) if x.ndim == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.ndim == 2 else adj  # [B, N, N]

        H, C = self.heads, self.out_channels
        B, N, _ = x.shape

        if add_loop:
            adj = adj.clone()
            idx = paddle.arange(N, dtype='int64')
            adj[:, idx, idx] = 1.0

        x = self.lin(x).reshape([B, N, H, C])  # [B, N, H, C]

        alpha_src = paddle.sum(x * self.att_src, axis=-1)  # [B, N, H]
        alpha_dst = paddle.sum(x * self.att_dst, axis=-1)  # [B, N, H]

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H]

        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = paddle.where(adj.unsqueeze(-1) == 0, float('-inf'), alpha)
        alpha = F.softmax(alpha, axis=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = paddle.matmul(alpha.transpose([0, 3, 1, 2]), x.transpose([0, 2, 1, 3]))
        out = out.transpose([0, 2, 1, 3])  # [B, N, H, C]

        if self.concat:
            out = out.reshape([B, N, H * C])
        else:
            out = out.mean(axis=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.reshape([-1, N, 1]).astype(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
