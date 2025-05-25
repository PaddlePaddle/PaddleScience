from typing import Optional

import paddle
from paddle import Tensor, nn


class MultiheadAttentionBlock(nn.Layer):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        layer_norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout

        self.attn = nn.MultiHeadAttention(
            channels,
            heads,
            dropout=dropout,
        )
        self.lin = nn.Linear(channels, channels)
        self.layer_norm1 = nn.LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = nn.LayerNorm(channels) if layer_norm else None

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.lin.reset_parameters()
        if self.layer_norm1 is not None:
            self.layer_norm1.reset_parameters()
        if self.layer_norm2 is not None:
            self.layer_norm2.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, x_mask: Optional[Tensor] = None,
                y_mask: Optional[Tensor] = None) -> Tensor:

        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, attn_mask=y_mask)

        if x_mask is not None:
            out = paddle.where(x_mask.unsqueeze(-1), out, paddle.zeros_like(out))

        out = out + x

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = out + self.lin(out).relu()

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads}, '
                f'layer_norm={self.layer_norm1 is not None}, '
                f'dropout={self.dropout})')


class SetAttentionBlock(nn.Layer):
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(x, x, mask, mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mab.channels}, '
                f'heads={self.mab.heads}, '
                f'layer_norm={self.mab.layer_norm1 is not None}, '
                f'dropout={self.mab.dropout})')


class InducedSetAttentionBlock(nn.Layer):
    def __init__(self, channels: int, num_induced_points: int, heads: int = 1,
                 layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.ind = self.create_parameter(shape=[1, num_induced_points, channels],
                                         default_initializer=nn.initializer.XavierUniform())
        self.mab1 = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)
        self.mab2 = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.ind)
        self.mab1.reset_parameters()
        self.mab2.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.mab1(self.ind.tile([x.shape[0], 1, 1]), x, y_mask=mask)
        return self.mab2(x, h, x_mask=mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.ind.shape[2]}, '
                f'num_induced_points={self.ind.shape[1]}, '
                f'heads={self.mab1.heads}, '
                f'layer_norm={self.mab1.layer_norm1 is not None}, '
                f'dropout={self.mab1.dropout})')


class PoolingByMultiheadAttention(nn.Layer):
    def __init__(self, channels: int, num_seed_points: int = 1, heads: int = 1,
                 layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.seed = self.create_parameter(shape=[1, num_seed_points, channels],
                                          default_initializer=nn.initializer.XavierUniform())
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.initializer.XavierUniform()(self.seed)
        self.mab.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.lin(x).relu()
        return self.mab(self.seed.tile([x.shape[0], 1, 1]), x, y_mask=mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.seed.shape[2]}, '
                f'num_seed_points={self.seed.shape[1]}, '
                f'heads={self.mab.heads}, '
                f'layer_norm={self.mab.layer_norm1 is not None}, '
                f'dropout={self.mab.dropout})')
