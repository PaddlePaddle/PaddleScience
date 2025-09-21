import copy

import numpy as np
import paddle
import paddle.nn as nn
from src.backbones.positional_encoding import PositionalEncoder


class LTAE2d(nn.Layer):
    """
    Lightweight Temporal Attention Encoder (L-TAE) for image time series.
    Attention-based sequence encoding that maps a sequence of images to a single feature map.
    A shared L-TAE is applied to all pixel positions of the image sequence.
    Args:
        in_channels (int): Number of channels of the input embeddings.
        n_head (int): Number of attention heads.
        d_k (int): Dimension of the key and query vectors.
        mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
        dropout (float): dropout
        d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
            to project them into a feature space of dimension d_model.
        T (int): Period to use for the positional encoding.
        return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
        positional_encoding (bool): If False, no positional encoding is used (default True).
    """

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):

        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1D(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1D(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1).tile([1, 1, h]).unsqueeze(-1).tile([1, 1, 1, w])
            )  # BxTxHxW
            pad_mask = pad_mask.transpose([0, 2, 3, 1]).reshape([sz_b * h * w, seq_len])

        out = x.transpose([0, 3, 4, 1, 2]).reshape([sz_b * h * w, seq_len, d])
        out = self.in_norm(out.transpose([0, 2, 1])).transpose([0, 2, 1])

        if self.inconv is not None:
            out = self.inconv(out.transpose([0, 2, 1])).transpose([0, 2, 1])

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .tile([1, 1, h])
                .unsqueeze(-1)
                .tile([1, 1, 1, w])
            )  # BxTxHxW
            bp = bp.transpose([0, 2, 3, 1]).reshape([sz_b * h * w, seq_len])
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = out.transpose([1, 0, 2]).reshape([sz_b * h * w, -1])  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.reshape([sz_b, h, w, -1]).transpose([0, 3, 1, 2])

        attn = attn.reshape([self.n_head, sz_b, h, w, seq_len]).transpose(
            [0, 1, 4, 2, 3]
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Layer):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = self.create_parameter(
            shape=[n_head, d_k],
            dtype="float32",
            default_initializer=nn.initializer.Normal(mean=0.0, std=np.sqrt(2.0 / d_k)),
        )

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        # Initialize weights
        nn.initializer.Normal(mean=0.0, std=np.sqrt(2.0 / d_k))(self.fc1_k.weight)

        self.attention = ScaledDotProductAttention(
            temperature=float(np.power(d_k, 0.5))
        )

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.shape

        q = paddle.stack([self.Q for _ in range(sz_b)], axis=1).reshape(
            [-1, d_k]
        )  # (n*b) x d_k

        k = self.fc1_k(v).reshape([sz_b, seq_len, n_head, d_k])
        k = k.transpose([2, 0, 1, 3]).reshape([-1, seq_len, d_k])  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.tile(
                [n_head, 1]
            )  # replicate pad_mask for each head (nxb) x lk

        # Split v into n_head chunks
        chunk_size = v.shape[-1] // n_head
        v_chunks = []
        for i in range(n_head):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            v_chunks.append(v[:, :, start_idx:end_idx])
        v = paddle.stack(v_chunks).reshape([n_head * sz_b, seq_len, -1])

        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.reshape([n_head, sz_b, 1, seq_len])
        attn = attn.squeeze(axis=2)

        output = output.reshape([n_head, sz_b, 1, d_in // n_head])
        output = output.squeeze(axis=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Layer):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(axis=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = paddle.matmul(q.unsqueeze(1), k.transpose([0, 2, 1]))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = paddle.where(pad_mask.unsqueeze(1), paddle.to_tensor(-1e3), attn)
        if return_comp:
            comp = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = paddle.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn
