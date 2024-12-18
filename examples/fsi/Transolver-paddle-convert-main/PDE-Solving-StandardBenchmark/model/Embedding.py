import paddle
import math
from einops import rearrange


class RotaryEmbedding(paddle.nn.Layer):

    def __init__(self, dim, min_freq=1 / 2, scale=1.0):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (paddle.arange(start=0, end=dim, step=2).
                                   astype(dtype='float32') / dim)
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer(name='inv_freq', tensor=inv_freq)

    def forward(self, coordinates, device):
        t = coordinates.to(device).astype(dtype=self.inv_freq.dtype)
        t = t * (self.scale / self.min_freq)
        freqs = paddle.einsum('... i , j -> ... i j', t, self.inv_freq)
        return paddle.concat(x=(freqs, freqs), axis=-1)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(t, freqs):
    return t * freqs.cos() + rotate_half(t) * freqs.sin()


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    d = tuple(t.shape)[-1]
    t_x, t_y = t[..., :d // 2], t[..., d // 2:]
    return paddle.concat(x=(apply_rotary_pos_emb(t_x, freqs_x),
                            apply_rotary_pos_emb(t_y, freqs_y)), axis=-1)


class PositionalEncoding(paddle.nn.Layer):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=421 * 421):
        super(PositionalEncoding, self).__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        pe = paddle.zeros(shape=[max_len, d_model])
        position = paddle.arange(start=0, end=max_len).unsqueeze(axis=1)
        div_term = paddle.exp(x=paddle.arange(start=0, end=d_model, step=2) *
                                -(math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(x=position * div_term)
        pe[:, 1::2] = paddle.cos(x=position * div_term)
        pe = pe.unsqueeze(axis=0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        out_0 = self.pe[:, :x.shape[1]]
        out_0.stop_gradient = not False
        x = x + out_0
        return self.dropout(x)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = paddle.exp(x=-math.log(max_period) * paddle.arange(start=0, end
    =half, dtype='float32') / half)
    args = timesteps[:, None].astype(dtype='float32') * freqs[None]
    embedding = paddle.concat(x=[paddle.cos(x=args), paddle.sin(x=args)],
                              axis=-1)
    if dim % 2:
        embedding = paddle.concat(x=[embedding, paddle.zeros_like(x=
                                                                  embedding[:, :1])], axis=-1)
    return embedding
