import math
from typing import Callable, Optional
import paddle
from paddle import Tensor
import paddle.nn.functional as F

def _orthogonal_matrix(dim: int) -> Tensor:
    r"""Get an orthogonal matrix by applying QR decomposition."""
    mat = paddle.randn((dim, dim))
    q, _ = paddle.linalg.qr(mat, mode='reduced')
    return q.t()


def orthogonal_matrix(num_rows: int, num_cols: int) -> Tensor:
    r"""Generate an orthogonal matrix with `num_rows` rows and `num_cols` columns."""
    num_full_blocks = int(num_rows / num_cols)
    blocks = []
    for _ in range(num_full_blocks):
        q = _orthogonal_matrix(num_cols)
        blocks.append(q)
    remain_rows = num_rows - num_full_blocks * num_cols
    if remain_rows > 0:
        q = _orthogonal_matrix(num_cols)
        blocks.append(q[:remain_rows])
    mat = paddle.concat(blocks, axis=0)
    return mat


def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    r"""Efficient attention mechanism from the `"Rethinking Attention with Performers" <https://arxiv.org/abs/2009.14794>`_ paper."""
    D_inv = 1.0 / (q @ k.sum(axis=-2).unsqueeze(-1))
    kv = paddle.matmul(k.transpose([0, 1, 3, 2]), v)
    qkv = paddle.matmul(q, kv)
    out = D_inv.squeeze(-1) * qkv
    return out


def generalized_kernel(
    x: Tensor,
    mat: Tensor,
    kernel: Callable = F.relu,
    epsilon: float = 0.001,
) -> Tensor:
    batch_size, num_heads = x.shape[:2]
    projection = mat.t().expand([batch_size, num_heads, -1, -1])
    x = paddle.matmul(x, projection)
    out = kernel(x) + epsilon
    return out


class PerformerProjection(paddle.nn.Layer):
    r"""The fast attention that uses a projection matrix from the `"Rethinking Attention with Performers" <https://arxiv.org/abs/2009.14794>`_ paper.
    """
    def __init__(self, num_cols: int, kernel: Callable = F.relu):
        super().__init__()
        num_rows = int(num_cols * math.log(num_cols))
        self.num_rows = num_rows
        self.num_cols = num_cols
        projection_matrix = orthogonal_matrix(self.num_rows, self.num_cols)
        self.register_buffer('projection_matrix', projection_matrix)
        self.kernel = kernel

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = generalized_kernel(q, self.projection_matrix, self.kernel)
        k = generalized_kernel(k, self.projection_matrix, self.kernel)
        out = linear_attention(q, k, v)
        return out


class PerformerAttention(paddle.nn.Layer):
    r"""The linear scaled attention mechanism from the `"Rethinking Attention with Performers" <https://arxiv.org/abs/2009.14794>`_ paper.
    """
    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: int = 64,
        kernel: Callable = F.relu,
        qkv_bias: bool = False,
        attn_out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels
        self.kernel = kernel
        self.fast_attn = PerformerProjection(head_channels, kernel)

        inner_channels = head_channels * heads
        self.q = paddle.nn.Linear(channels, inner_channels, bias_attr=qkv_bias)
        self.k = paddle.nn.Linear(channels, inner_channels, bias_attr=qkv_bias)
        self.v = paddle.nn.Linear(channels, inner_channels, bias_attr=qkv_bias)
        self.attn_out = paddle.nn.Linear(inner_channels, channels, bias_attr=attn_out_bias)
        self.dropout = paddle.nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, N, *_ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = [t.reshape([B, N, self.heads, self.head_channels]).transpose([0, 2, 1, 3]) for t in (q, k, v)]
        if mask is not None:
            mask = mask[:, None, :, None]
            v = paddle.where(~mask, paddle.zeros_like(v), v)
        out = self.fast_attn(q, k, v)
        out = out.transpose([0, 2, 1, 3]).reshape([B, N, -1])
        out = self.attn_out(out)
        out = self.dropout(out)
        return out

    def redraw_projection_matrix(self):
        r"""As described in the paper, periodically redraw examples to improve overall approximation of attention."""
        num_rows = self.fast_attn.num_rows
        num_cols = self.fast_attn.num_cols
        projection_matrix = orthogonal_matrix(num_rows, num_cols)
        self.fast_attn.projection_matrix.set_value(projection_matrix)

    def _reset_parameters(self):
        self.q.weight.set_value(paddle.nn.initializer.KaimingUniform())
        self.k.weight.set_value(paddle.nn.initializer.KaimingUniform())
        self.v.weight.set_value(paddle.nn.initializer.KaimingUniform())
        self.attn_out.weight.set_value(paddle.nn.initializer.KaimingUniform())
        self.redraw_projection_matrix()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(heads={self.heads}, head_channels={self.head_channels}, kernel={self.kernel})'
