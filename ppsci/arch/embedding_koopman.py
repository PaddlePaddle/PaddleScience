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
Code below is heavily based on [transformer-physx](https://github.com/zabaras/transformer-physx)
"""

from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Uniform

from ppsci.arch import base

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class LorenzEmbedding(base.Arch):
    """Embedding Koopman model for the Lorenz ODE system.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("states",).
        output_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        mean (Optional[Tuple[float, ...]]): Mean of training dataset. Defaults to None.
        std (Optional[Tuple[float, ...]]): Standard Deviation of training dataset. Defaults to None.
        input_size (int, optional): Size of input data. Defaults to 3.
        hidden_size (int, optional): Number of hidden size. Defaults to 500.
        embed_size (int, optional): Number of embedding size. Defaults to 32.
        drop (float, optional):  Probability of dropout the units. Defaults to 0.0.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.LorenzEmbedding(("x", "y"), ("u", "v"))
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
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Assign(data),
        )

        data = 0.1 * paddle.rand([2 * embed_size - 3])
        k_ut = paddle.create_parameter(
            shape=data.shape,
            dtype=paddle.get_default_dtype(),
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


class RosslerEmbedding(LorenzEmbedding):
    """Embedding Koopman model for the Rossler ODE system.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("states",).
        output_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        mean (Optional[Tuple[float, ...]]): Mean of training dataset. Defaults to None.
        std (Optional[Tuple[float, ...]]): Standard Deviation of training dataset. Defaults to None.
        input_size (int, optional): Size of input data. Defaults to 3.
        hidden_size (int, optional): Number of hidden size. Defaults to 500.
        embed_size (int, optional): Number of embedding size. Defaults to 32.
        drop (float, optional):  Probability of dropout the units. Defaults to 0.0.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.RosslerEmbedding(("x", "y"), ("u", "v"))
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
        super().__init__(
            input_keys,
            output_keys,
            mean,
            std,
            input_size,
            hidden_size,
            embed_size,
            drop,
        )


class CylinderEmbedding(base.Arch):
    """Embedding Koopman model for the Cylinder system.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("states", "visc").
        output_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        mean (Optional[Tuple[float, ...]]): Mean of training dataset. Defaults to None.
        std (Optional[Tuple[float, ...]]): Standard Deviation of training dataset. Defaults to None.
        embed_size (int, optional): Number of embedding size. Defaults to 128.
        encoder_channels (Optional[Tuple[int, ...]]): Number of channels in encoder network. Defaults to None.
        decoder_channels (Optional[Tuple[int, ...]]): Number of channels in decoder network. Defaults to None.
        drop (float, optional):  Probability of dropout the units. Defaults to 0.0.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.CylinderEmbedding(("x", "y"), ("u", "v"))
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        embed_size: int = 128,
        encoder_channels: Optional[Tuple[int, ...]] = None,
        decoder_channels: Optional[Tuple[int, ...]] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.embed_size = embed_size

        X, Y = np.meshgrid(np.linspace(-2, 14, 128), np.linspace(-4, 4, 64))
        self.mask = paddle.to_tensor(np.sqrt(X**2 + Y**2)).unsqueeze(0).unsqueeze(0)

        encoder_channels = (
            [4, 16, 32, 64, 128] if encoder_channels is None else encoder_channels
        )
        decoder_channels = (
            [embed_size // 32, 128, 64, 32, 16]
            if decoder_channels is None
            else decoder_channels
        )
        self.encoder_net = self.build_encoder(embed_size, encoder_channels, drop)
        self.k_diag_net, self.k_ut_net, self.k_lt_net = self.build_koopman_operator(
            embed_size
        )
        self.decoder_net = self.build_decoder(decoder_channels)

        xidx = []
        yidx = []
        for i in range(1, 5):
            yidx.append(np.arange(i, embed_size))
            xidx.append(np.arange(0, embed_size - i))
        self.xidx = paddle.to_tensor(np.concatenate(xidx), dtype="int64")
        self.yidx = paddle.to_tensor(np.concatenate(yidx), dtype="int64")

        mean = [0.0, 0.0, 0.0, 0.0] if mean is None else mean
        std = [1.0, 1.0, 1.0, 1.0] if std is None else std
        self.register_buffer("mean", paddle.to_tensor(mean).reshape([1, 4, 1, 1]))
        self.register_buffer("std", paddle.to_tensor(std).reshape([1, 4, 1, 1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            k = 1 / m.weight.shape[0]
            uniform = Uniform(-(k**0.5), k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            k = 1 / (m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
            uniform = Uniform(-(k**0.5), k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)

    def _build_conv_relu_list(
        self, in_channels: Tuple[int, ...], out_channels: Tuple[int, ...]
    ):
        net_list = [
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
        ]
        return net_list

    def build_encoder(
        self, embed_size: int, channels: Tuple[int, ...], drop: float = 0.0
    ):
        net = []
        for i in range(1, len(channels)):
            net.extend(self._build_conv_relu_list(channels[i - 1], channels[i]))
        net.append(
            nn.Conv2D(
                channels[-1],
                embed_size // 32,
                kernel_size=(3, 3),
                padding=1,
                padding_mode="replicate",
            )
        )
        net.append(
            nn.LayerNorm(
                (4, 4, 8),
            )
        )
        net.append(nn.Dropout(drop))
        net = nn.Sequential(*net)
        return net

    def _build_upsample_conv_relu(
        self, in_channels: Tuple[int, ...], out_channels: Tuple[int, ...]
    ):
        net_list = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
        ]
        return net_list

    def build_decoder(self, channels: Tuple[int, ...]):
        net = []
        for i in range(1, len(channels)):
            net.extend(self._build_upsample_conv_relu(channels[i - 1], channels[i]))
        net.append(
            nn.Conv2D(
                channels[-1],
                3,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
        )
        net = nn.Sequential(*net)
        return net

    def build_koopman_operator(self, embed_size: int):
        # Learned Koopman operator parameters
        k_diag_net = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, embed_size)
        )

        k_ut_net = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, 4 * embed_size - 10)
        )
        k_lt_net = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, 4 * embed_size - 10)
        )
        return k_diag_net, k_ut_net, k_lt_net

    def encoder(self, x: paddle.Tensor, viscosity: paddle.Tensor):
        B, T, C, H, W = x.shape
        x = x.reshape((B * T, C, H, W))
        viscosity = viscosity.repeat_interleave(T, axis=1).reshape((B * T, 1))
        x = paddle.concat(
            [x, viscosity.unsqueeze(-1).unsqueeze(-1) * paddle.ones_like(x[:, :1])],
            axis=1,
        )
        x = self._normalize(x)
        g = self.encoder_net(x)
        g = g.reshape([B, T, -1])
        return g

    def decoder(self, g: paddle.Tensor):
        B, T, _ = g.shape
        x = self.decoder_net(g.reshape([-1, self.embed_size // 32, 4, 8]))
        x = self._unnormalize(x)
        mask0 = (
            self.mask.repeat_interleave(x.shape[1], axis=1).repeat_interleave(
                x.shape[0], axis=0
            )
            < 1
        )
        x[mask0] = 0
        _, C, H, W = x.shape
        x = x.reshape([B, T, C, H, W])
        return x

    def get_koopman_matrix(self, g: paddle.Tensor, visc: paddle.Tensor):
        # # Koopman operator
        kMatrix = paddle.zeros([g.shape[0], self.embed_size, self.embed_size])
        kMatrix.stop_gradient = False
        # Populate the off diagonal terms
        kMatrixUT_data = self.k_ut_net(100 * visc)
        kMatrixLT_data = self.k_lt_net(100 * visc)

        kMatrix = kMatrix.transpose([1, 2, 0])
        kMatrixUT_data_t = kMatrixUT_data.transpose([1, 0])
        kMatrixLT_data_t = kMatrixLT_data.transpose([1, 0])
        kMatrix[self.xidx, self.yidx] = kMatrixUT_data_t
        kMatrix[self.yidx, self.xidx] = kMatrixLT_data_t

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[1])
        ind = paddle.to_tensor(ind, dtype="int64")

        kMatrixDiag = self.k_diag_net(100 * visc)
        kMatrixDiag_t = kMatrixDiag.transpose([1, 0])
        kMatrix[ind[0], ind[1]] = kMatrixDiag_t
        return kMatrix.transpose([2, 0, 1])

    def koopman_operation(self, embed_data: paddle.Tensor, k_matrix: paddle.Tensor):
        embed_pred_data = paddle.bmm(
            k_matrix, embed_data.transpose([0, 2, 1])
        ).transpose([0, 2, 1])
        return embed_pred_data

    def _normalize(self, x: paddle.Tensor):
        x = (x - self.mean) / self.std
        return x

    def _unnormalize(self, x: paddle.Tensor):
        return self.std[:, :3] * x + self.mean[:, :3]

    def forward_tensor(self, states, visc):
        # states.shape=(B, T, C, H, W)
        embed_data = self.encoder(states, visc)
        recover_data = self.decoder(embed_data)

        k_matrix = self.get_koopman_matrix(embed_data, visc)
        embed_pred_data = self.koopman_operation(embed_data, k_matrix)
        pred_data = self.decoder(embed_pred_data)

        return (pred_data[:, :-1], recover_data, k_matrix)

    def split_to_dict(
        self, data_tensors: Tuple[paddle.Tensor, ...], keys: Tuple[str, ...]
    ):
        return {key: data_tensors[i] for i, key in enumerate(keys)}

    def forward(self, x):

        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.forward_tensor(**x)
        y = self.split_to_dict(y, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y
