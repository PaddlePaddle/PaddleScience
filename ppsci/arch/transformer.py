# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://github.com/omron-sinicx/transformer4sr
"""

from __future__ import annotations

import math
from typing import Callable
from typing import Tuple

import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base


def transpose_aux_func(dims, dim0, dim1):
    perm = list(range(dims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class MultiHeadAttention(nn.Layer):
    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_O = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = paddle.matmul(
            x=Q, y=K.transpose(perm=transpose_aux_func(K.ndim, -1, -2))
        ) / math.sqrt(self.d_k)
        if mask is not None:
            scores = paddle.where(
                condition=mask,
                x=paddle.to_tensor(data=[-1e9], dtype="float32"),
                y=scores,
            )
        weights = nn.functional.softmax(x=scores, axis=-1)
        return paddle.matmul(x=weights, y=V)

    def forward(self, Q, K, V, mask=None):
        Q_temp = paddle.reshape(
            x=self.W_Q(Q),
            shape=[i for i in tuple(Q.shape)[:-1]] + [self.heads] + [self.d_k],
        ).transpose(
            perm=transpose_aux_func(
                paddle.reshape(
                    x=self.W_Q(Q),
                    shape=[i for i in tuple(Q.shape)[:-1]] + [self.heads] + [self.d_k],
                ).ndim,
                1,
                2,
            )
        )
        K_temp = paddle.reshape(
            x=self.W_K(K),
            shape=[i for i in tuple(K.shape)[:-1]] + [self.heads] + [self.d_k],
        ).transpose(
            perm=transpose_aux_func(
                paddle.reshape(
                    x=self.W_K(K),
                    shape=[i for i in tuple(K.shape)[:-1]] + [self.heads] + [self.d_k],
                ).ndim,
                1,
                2,
            )
        )
        V_temp = paddle.reshape(
            x=self.W_V(V),
            shape=[i for i in tuple(V.shape)[:-1]] + [self.heads] + [self.d_k],
        ).transpose(
            perm=transpose_aux_func(
                paddle.reshape(
                    x=self.W_V(V),
                    shape=[i for i in tuple(V.shape)[:-1]] + [self.heads] + [self.d_k],
                ).ndim,
                1,
                2,
            )
        )
        sdpa = self.scaled_dot_product_attention(
            Q_temp, K_temp, V_temp, mask
        ).transpose(
            perm=transpose_aux_func(
                self.scaled_dot_product_attention(Q_temp, K_temp, V_temp, mask).ndim,
                1,
                2,
            )
        )
        sdpa = paddle.reshape(
            x=sdpa, shape=[i for i in tuple(sdpa.shape)[:-2]] + [self.d_model]
        )
        y_mha = self.W_O(sdpa)
        return y_mha


class MLP(nn.Layer):
    def __init__(self, list_dims, act="relu", dropout=0.0):
        super().__init__()
        self.layers = nn.LayerList()
        for i in range(len(list_dims) - 1):
            self.layers.append(
                nn.Linear(in_features=list_dims[i], out_features=list_dims[i + 1])
            )
            self.layers.append(act_mod.get_activation(act) if act else None)
            self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class EncoderLayerMix(nn.Layer):
    def __init__(self, in_features, d_model, heads, act="relu", dropout=0.0):
        super().__init__()
        self.mlp = MLP([in_features, d_model, d_model], act="relu", dropout=dropout)
        self.multihead_attention = MultiHeadAttention(heads, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        y = x
        y = paddle.flatten(y, start_axis=2)
        y = self.mlp(y)
        y = self.multihead_attention(y, y, y, mask=None)
        y = self.dropout(y)
        y = paddle.unsqueeze(y, axis=2)
        y = x + y
        y = self.norm(y)
        return y


class Encoder(nn.Layer):
    def __init__(
        self, num_layers, num_var_max, d_model, heads, act="relu", dropout=0.0
    ):
        super().__init__()
        self.first_mlp = MLP([1, d_model, d_model], act="relu", dropout=dropout)
        self.layers = nn.LayerList(
            sublayers=[
                EncoderLayerMix(
                    d_model * num_var_max, d_model, heads, act="relu", dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.last_mlp = MLP([d_model, d_model], act="relu", dropout=dropout)

    def forward(self, x):
        y = x
        y = self.first_mlp(y)
        for layer in self.layers:
            y = layer(y)
        y = self.last_mlp(y)
        y = paddle.max(y, axis=1)
        return y


class TokenEmbeddings(nn.Layer):
    def __init__(self, vocab_size, seq_length, d_model, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.seq_length = seq_length
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.get_pe_num()

    def get_pe_num(self):
        self.pe = paddle.zeros(shape=[self.seq_length, self.d_model])
        numerator = paddle.arange(
            self.seq_length, dtype=paddle.get_default_dtype()
        ).unsqueeze(axis=1)
        denominator = paddle.pow(
            x=paddle.to_tensor(10e4, dtype=paddle.get_default_dtype()),
            y=paddle.arange(self.d_model, step=2) / self.d_model,
        ).unsqueeze(axis=0)
        self.pe[:, 0::2] = paddle.sin(x=numerator / denominator)
        self.pe[:, 1::2] = paddle.cos(x=numerator / denominator)
        self.pe.stop_gradient = True

    def forward(self, x):
        # embedding
        y = x
        y = self.embed(y) * math.sqrt(self.d_model)
        # position encoding
        y = self.dropout(y + self.pe)
        return y


class DecoderLayer(nn.Layer):
    def __init__(self, heads, d_model, act="relu", dropout=0.0):
        super().__init__()
        self.multihead_attention_1 = MultiHeadAttention(heads, d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.multihead_attention_2 = MultiHeadAttention(heads, d_model)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

        self.mlp = MLP([d_model, 2 * d_model, d_model], act="relu", dropout=dropout)
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, x_emb, x_enc, mask):
        y_mha_1 = self.multihead_attention_1(x_emb, x_emb, x_emb, mask=mask)
        y_mha_1 = self.dropout_1(y_mha_1)
        y = y_mha_1 + x_emb
        y = self.norm_1(y)
        y_mha_2 = self.multihead_attention_2(y, x_enc, x_enc, mask=None)
        y_mha_2 = self.dropout_2(y_mha_2)
        y = y + y_mha_2
        y = self.norm_2(y)
        y_mlp = self.mlp(y)
        y = y + y_mlp
        y = self.norm_3(y)
        return y


class Decoder(nn.Layer):
    def __init__(
        self,
        num_layers,
        vocab_size,
        seq_length,
        d_model,
        heads,
        act="relu",
        dropout=0.0,
    ):
        super().__init__()
        self.token_embeddings = TokenEmbeddings(
            vocab_size, seq_length, d_model, dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.LayerList(
            sublayers=[
                DecoderLayer(heads, d_model, act="relu", dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_target, x_enc, mask):
        y = x_target
        y = self.token_embeddings(y)
        y = self.dropout(y)
        for layer in self.layers:
            y = layer(y, x_enc, mask)
        return y


class Transformer(base.Arch):
    """A Kind of Transformer Model.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("x", "y", "z").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("u", "v", "w").
        num_var_max (int): Maximum number of variables.
        vocab_size (int): Size of vocab. Size of unary operators = 1, binary operators = 2.
        seq_length (int): Length of sequance.
        d_model (int, optional): The innermost dimension of model. Defaults to 256.
        heads (int, optional): The number of independent heads for the multi-head attention layers. Defaults to 4.
        num_layers_enc (int, optional): The number of encoders. Defaults to 4.
        num_layers_dec (int, optional): The number of decoders. Defaults to 8.
        dropout (float, optional): Dropout regularization. Defaults to 0.0.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.Transformer(
        ...     input_keys=("input", "target_seq"),
        ...     output_keys=("output",),
        ...     num_var_max=7,
        ...     vocab_size=20,
        ...     seq_length=30,
        ... )
        >>> input_dict = {"input": paddle.rand([512, 50, 7, 1]),
        ...               "target_seq": paddle.rand([512, 30])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["output"].shape)
        [512, 30, 20]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_var_max: int,
        vocab_size: int,
        seq_length: int,
        d_model: int = 256,
        heads: int = 4,
        num_layers_enc: int = 4,
        num_layers_dec: int = 8,
        act: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.num_var_max = num_var_max
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.d_model = d_model
        self.heads = heads
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.act = act
        self.dropout = dropout

        self.encoder = Encoder(
            num_layers_enc, num_var_max, d_model, heads, act="relu", dropout=dropout
        )
        self.decoder = Decoder(
            num_layers_dec,
            vocab_size,
            seq_length,
            d_model,
            heads,
            act="relu",
            dropout=dropout,
        )
        self.last_layer = paddle.nn.Linear(in_features=d_model, out_features=vocab_size)

    def get_mask(self, target_seq):
        padding_mask = paddle.equal(target_seq, 0).unsqueeze(axis=1).unsqueeze(axis=1)
        future_mask = paddle.triu(
            paddle.ones(shape=[target_seq.shape[1], target_seq.shape[1]]),
            diagonal=1,
        ).astype(dtype="bool")
        mask = paddle.logical_or(x=padding_mask, y=future_mask)
        return mask

    def forward_tensor(self, x_lst):
        y, target_seq = x_lst[0], x_lst[1]
        mask = self.get_mask(target_seq)
        y_enc = self.encoder(y)
        y = self.decoder(target_seq, y_enc, mask)
        y = self.last_layer(y)
        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        x_lst = [x[key] for key in self.input_keys]  # input, target_seq
        y = self.forward_tensor(x_lst)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y

    @paddle.no_grad()
    def decode_process(
        self, dataset: paddle.Tensor, complete_func: Callable
    ) -> paddle.Tensor:
        """Greedy decode with the Transformer model, decode until the equation tree is completed.

        Args:
            dataset (paddle.Tensor): Tabular dataset.
            complete_func (Callable): Function used to calculate whether inference is complete.
        """
        encoder_output = self.encoder(dataset)
        decoder_output = paddle.zeros(
            shape=(dataset.shape[0], self.seq_length + 1), dtype=paddle.int64
        )
        decoder_output[:, 0] = 1
        is_complete = paddle.zeros(shape=dataset.shape[0], dtype=paddle.bool)
        for n1 in range(self.seq_length):
            padding_mask = (
                paddle.equal(x=decoder_output[:, :-1], y=0)
                .unsqueeze(axis=1)
                .unsqueeze(axis=1)
            )
            future_mask = paddle.triu(
                x=paddle.ones(shape=[self.seq_length, self.seq_length]), diagonal=1
            ).astype(dtype=paddle.bool)
            mask_dec = paddle.logical_or(x=padding_mask, y=future_mask)
            y_dec = self.decoder(
                x_target=decoder_output[:, :-1],
                x_enc=encoder_output,
                mask=mask_dec,
            )
            y_mlp = self.last_layer(y_dec)
            # set value depending on complete condition
            decoder_output[:, n1 + 1] = paddle.where(
                is_complete, 0, paddle.argmax(y_mlp[:, n1], axis=-1)
            )
            # set complete condition
            for n2 in range(dataset.shape[0]):
                if complete_func(decoder_output[n2, 1:]):
                    is_complete[n2] = True
        return decoder_output
