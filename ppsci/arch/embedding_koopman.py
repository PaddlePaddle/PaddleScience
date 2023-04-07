# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code below is heavily based on:[transformer-physx](https://github.com/zabaras/transformer-physx)
"""

from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Uniform

from ppsci.arch import base

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class LorenzEmbedding(base.NetBase):
    """Embedding Koopman model for the Lorenz ODE system.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("states",).
        output_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        mean (Optional[Tuple[float, ...]], optional): Mean of training dataset. Defaults to None.
        std (Optional[Tuple[float, ...]], optional): Standard Deviation of training dataset. Defaults to None.
        input_size (int, optional): Size of input data. Defaults to 3.
        hidden_size (int, optional): Number of hidden size. Defaults to 500.
        embed_size (int, optional): Number of embedding size. Defaults to 32.
        drop (float, optional):  Probability of dropout the units. Defaults to 0.0.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        input_size: int = 3,
        hidden_size: int = 500,
        embed_size: int = 32,
        drop: float = 0.0,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        # build observable netowrk
        self.encoder_net = self.build_encoder(input_size, hidden_size, embed_size, drop)
        # build koopman operator
        self.k_diag, self.k_ut = self.build_koopman_operator(embed_size)
        # build recovery netowrk
        self.decoder_net = self.build_decoder(input_size, hidden_size, embed_size)

        mean = [0.0, 0.0, 0.0] if mean is None else mean
        std = [1.0, 1.0, 1.0] if std is None else std
        self.register_buffer("mean", paddle.to_tensor(mean).reshape([1, 3]))
        self.register_buffer("std", paddle.to_tensor(std).reshape([1, 3]))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Layer):
        if isinstance(m, nn.Linear):
            k = 1 / m.weight.shape[0]
            uniform = Uniform(-(k**0.5), k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def build_encoder(
        self, input_size: int, hidden_size: int, embed_size: int, drop: float = 0.0
    ):
        net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.Dropout(drop),
        )
        return net

    def build_decoder(self, input_size: int, hidden_size: int, embed_size: int):
        net = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        return net

    def build_koopman_operator(self, embed_size: int):
        # Learned Koopman operator
        data = paddle.linspace(1, 0, embed_size)
        k_diag = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data),
        )

        data = 0.1 * paddle.rand([2 * embed_size - 3])
        k_ut = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data),
        )
        return k_diag, k_ut

    def encoder(self, x: paddle.Tensor):
        x = self._normalize(x)
        g = self.encoder_net(x)
        return g

    def decoder(self, g: paddle.Tensor):
        out = self.decoder_net(g)
        x = self._unnormalize(out)
        return x

    def koopman_operation(self, embed_data: paddle.Tensor, k_matrix: paddle.Tensor):
        # Apply Koopman operation
        embed_pred_data = paddle.bmm(
            k_matrix.expand(
                [embed_data.shape[0], k_matrix.shape[0], k_matrix.shape[1]]
            ),
            embed_data.transpose([0, 2, 1]),
        ).transpose([0, 2, 1])
        return embed_pred_data

    def _normalize(self, x: paddle.Tensor):
        return (x - self.mean) / self.std

    def _unnormalize(self, x: paddle.Tensor):
        return self.std * x + self.mean

    def get_koopman_matrix(self):
        # # Koopman operator
        k_ut_tensor = self.k_ut * 1
        k_ut_tensor = paddle.diag(
            k_ut_tensor[0 : self.embed_size - 1], offset=1
        ) + paddle.diag(k_ut_tensor[self.embed_size - 1 :], offset=2)
        k_matrix = k_ut_tensor + (-1) * k_ut_tensor.t()
        k_matrix = k_matrix + paddle.diag(self.k_diag)
        return k_matrix

    def forward_tensor(self, x):
        k_matrix = self.get_koopman_matrix()
        embed_data = self.encoder(x)
        recover_data = self.decoder(embed_data)

        embed_pred_data = self.koopman_operation(embed_data, k_matrix)
        pred_data = self.decoder(embed_pred_data)

        return (pred_data[:, :-1, :], recover_data, k_matrix)

    def split_to_dict(
        self, data_tensors: Tuple[paddle.Tensor, ...], keys: Tuple[str, ...]
    ):
        return {key: data_tensors[i] for i, key in enumerate(keys)}

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        x = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(x)
        y = self.split_to_dict(y, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y
