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

from __future__ import annotations

import operator
from functools import reduce
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.utils import initializer


class Laplace(nn.Layer):
    """Generic N-Dimensional Laplace Operator with Pole-Residue Method.

    Args:
        in_channels (int):  Number of input channels of the first layer.
        out_channels (int): Number of output channels of the last layer.
        modes (Tuple[int, ...]): Number of modes to use for contraction in Laplace domain during training.
        T (paddle.Tensor): Linspace of time dimension.
        data (Tuple[paddle.Tensor, ...]): Linspaces of other dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Tuple[int, ...],
        T: paddle.Tensor,
        data: Tuple[paddle.Tensor, ...],
    ):
        super().__init__()
        self.char1 = "pqr"
        self.char2 = "mnk"
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.dims = len(modes)

        self.weights_pole_real = nn.ParameterList()
        self.weights_pole_imag = nn.ParameterList()
        for i in range(self.dims):
            weight_real = self._init_weights(
                self.create_parameter((in_channels, out_channels, modes[i], 1))
            )
            weight_imag = self._init_weights(
                self.create_parameter((in_channels, out_channels, modes[i], 1))
            )
            self.weights_pole_real.append(weight_real)
            self.weights_pole_imag.append(weight_imag)

        residues_shape = (in_channels, out_channels) + modes + (1,)
        self.weights_residue_real = self._init_weights(
            self.create_parameter(residues_shape)
        )
        self.weights_residue_imag = self._init_weights(
            self.create_parameter(residues_shape)
        )

        self.initialize_lambdas(T, data)
        self.get_einsum_eqs()

    def _init_weights(self, weight) -> paddle.Tensor:
        return initializer.uniform_(weight, a=0, b=self.scale)

    def initialize_lambdas(self, T, data) -> None:
        self.t_lst = (T,) + data
        self.lambdas = []
        for i in range(self.dims):
            t_i = self.t_lst[i]
            self.register_buffer(f"t_{i}", t_i)
            dt = (t_i[0, 1] - t_i[0, 0]).item()
            omega = paddle.fft.fftfreq(n=tuple(t_i.shape)[1], d=dt) * 2 * np.pi * 1.0j
            lambda_ = omega.reshape([*omega.shape, 1, 1, 1])
            self.register_buffer(f"lambda_{i}", lambda_)
            self.lambdas.append(lambda_)

    def get_einsum_eqs(self) -> None:
        terms_eq = []
        terms_x2_eq = []
        for i in range(self.dims):
            term_eq = self.char1[i] + "io" + self.char2[i]
            terms_eq.append(term_eq)
            term_x2_eq = "io" + self.char2[i] + self.char1[i]
            terms_x2_eq.append(term_x2_eq)
        self.eq1 = (
            "bi"
            + "".join(self.char1)
            + ","
            + "io"
            + "".join(self.char2)
            + ","
            + ",".join(terms_eq)
            + "->"
            + "bo"
            + "".join(self.char1)
        )
        self.eq2 = (
            "bi"
            + "".join(self.char1)
            + ","
            + "io"
            + "".join(self.char2)
            + ","
            + ",".join(terms_eq)
            + "->"
            + "bo"
            + "".join(self.char2)
        )
        self.eq_x2 = (
            "bi"
            + "".join(self.char2)
            + ","
            + ",".join(terms_x2_eq)
            + "->bo"
            + "".join(self.char1)
        )

    def output_PR(self, alpha) -> Tuple[paddle.Tensor, paddle.Tensor]:
        weights_residue = paddle.as_complex(
            paddle.concat(
                [self.weights_residue_real, self.weights_residue_imag], axis=-1
            )
        )
        self.weights_pole = []
        terms = []
        for i in range(self.dims):
            weights_pole = paddle.as_complex(
                paddle.concat(
                    [self.weights_pole_real[i], self.weights_pole_imag[i]], axis=-1
                )
            )
            self.weights_pole.append(weights_pole)
            sub = paddle.subtract(self.lambdas[i], weights_pole)
            terms.append(paddle.divide(paddle.to_tensor(1, dtype=sub.dtype), sub))

        output_residue1 = paddle.einsum(self.eq1, alpha, weights_residue, *terms)
        output_residue2 = (-1) ** self.dims * paddle.einsum(
            self.eq2, alpha, weights_residue, *terms
        )
        return output_residue1, output_residue2

    def forward(self, x):
        alpha = paddle.fft.fftn(x=x, axes=[-3, -2, -1])
        output_residue1, output_residue2 = self.output_PR(alpha)

        x1 = paddle.fft.ifftn(
            x=output_residue1, s=(x.shape[-3], x.shape[-2], x.shape[-1])
        )
        x1 = paddle.real(x=x1)

        exp_terms = []
        for i in range(self.dims):
            term = paddle.einsum(
                "io"
                + self.char2[i]
                + ",d"
                + self.char1[i]
                + "->io"
                + self.char2[i]
                + self.char1[i],
                self.weights_pole[i],
                self.t_lst[i].astype(paddle.complex64).reshape([1, -1]),
            )
            exp_terms.append(paddle.exp(term))

        x2 = paddle.einsum(self.eq_x2, output_residue2, *exp_terms)
        x2 = paddle.real(x2)
        x2 = x2 / reduce(operator.mul, x.shape[-3:], 1)
        return x1 + x2


class LNO(base.Arch):
    """Laplace Neural Operator net.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input1", "input2").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output1", "output2").
        width (int): Tensor width of Laplace Layer.
        modes (Tuple[int, ...]): Number of modes to use for contraction in Laplace domain during training.
        T (paddle.Tensor): Linspace of time dimension.
        data (Tuple[paddle.Tensor, ...]): Linspaces of other dimensions.
        in_features (int, optional): Number of input channels of the first layer.. Defaults to 1.
        hidden_features (int, optional): Number of channels of the fully-connected layer. Defaults to 64.
        activation (str, optional): The activation function. Defaults to "sin".
        use_norm (bool, optional): Whether to use normalization layers. Defaults to True.
        use_grid (bool, optional): Whether to create grid. Defaults to False.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        width: int,
        modes: Tuple[int, ...],
        T: paddle.Tensor,
        data: Optional[Tuple[paddle.Tensor, ...]] = None,
        in_features: int = 1,
        hidden_features: int = 64,
        activation: str = "sin",
        use_norm: bool = True,
        use_grid: bool = False,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.width = width
        self.modes = modes
        self.dims = len(modes)
        assert self.dims <= 3, "Only 3 dims and lower of modes are supported now."

        if data is None:
            data = ()
        assert (
            self.dims == len(data) + 1
        ), f"Dims of modes is {self.dims} but only {len(data)} dims(except T) of data received."

        self.fc0 = nn.Linear(in_features=in_features, out_features=self.width)
        self.laplace = Laplace(self.width, self.width, self.modes, T, data)
        self.conv = getattr(nn, f"Conv{self.dims}D")(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            data_format="NCDHW",
        )
        if use_norm:
            self.norm = getattr(nn, f"InstanceNorm{self.dims}D")(
                num_features=self.width,
                weight_attr=False,
                bias_attr=False,
            )
        self.fc1 = nn.Linear(in_features=self.width, out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=1)
        self.act = act_mod.get_activation(activation)

        self.use_norm = use_norm
        self.use_grid = use_grid

    def get_grid(self, shape):
        batchsize, size_t, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridt = paddle.linspace(0, 1, size_t)
        gridt = gridt.reshape([1, size_t, 1, 1, 1]).tile(
            [batchsize, 1, size_x, size_y, 1]
        )
        gridx = paddle.linspace(0, 1, size_x)
        gridx = gridx.reshape([1, 1, size_x, 1, 1]).tile(
            [batchsize, size_t, 1, size_y, 1]
        )
        gridy = paddle.linspace(0, 1, size_y)
        gridy = gridy.reshape([1, 1, 1, size_y, 1]).tile(
            [batchsize, size_t, size_x, 1, 1]
        )
        return paddle.concat([gridt, gridx, gridy], axis=-1)

    def transpoe_to_NCDHW(self, x):
        perm = [0, self.dims + 1] + list(range(1, self.dims + 1))
        return paddle.transpose(x, perm=perm)

    def transpoe_to_NDHWC(self, x):
        perm = [0] + list(range(2, self.dims + 2)) + [1]
        return paddle.transpose(x, perm=perm)

    def forward_tensor(self, x):
        if self.use_grid:
            grid = self.get_grid(x.shape)
            x = paddle.concat([x, grid], axis=-1)
        x = self.fc0(x)
        x = self.transpoe_to_NCDHW(x)

        if self.use_norm:
            x1 = self.norm(self.laplace(self.norm(x)))
        else:
            x1 = self.laplace(x)

        x2 = self.conv(x)
        x = x1 + x2

        x = self.transpoe_to_NDHWC(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
