# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.utils import initializer


class WeightNormLinear(nn.Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = self.create_parameter((in_features, out_features))
        self.weight_g = self.create_parameter((out_features,))
        if bias:
            self.bias = self.create_parameter((out_features,))
        else:
            self.bias = None
        self._init_weights()

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.weight_v)
        initializer.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            initializer.constant_(self.bias, 0.0)

    def forward(self, input):
        norm = self.weight_v.norm(p=2, axis=0, keepdim=True)
        weight = self.weight_g * self.weight_v / norm
        return nn.functional.linear(input, weight, self.bias)


class RandomWeightFactorization(nn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 0.5,
        std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = self.create_parameter((in_features, out_features))
        self.weight_g = self.create_parameter((out_features,))
        if bias:
            self.bias = self.create_parameter((out_features,))
        else:
            self.bias = None

        self._init_weights(mean, std)

    def _init_weights(self, mean, std):
        with paddle.no_grad():
            initializer.glorot_normal_(self.weight_v)

            nn.initializer.Normal(mean, std)(self.weight_g)
            paddle.assign(paddle.exp(self.weight_g), self.weight_g)
            paddle.assign(self.weight_v / self.weight_g, self.weight_v)
            if self.bias is not None:
                initializer.constant_(self.bias, 0.0)

        self.weight_g.stop_gradient = False
        self.weight_v.stop_gradient = False
        self.bias.stop_gradient = False

    def forward(self, input):
        return nn.functional.linear(input, self.weight_g * self.weight_v, self.bias)


class PeriodEmbedding(nn.Layer):
    def __init__(self, periods: Dict[str, Tuple[float, bool]]):
        super().__init__()
        self.freqs_dict = {
            k: self.create_parameter(
                [],
                attr=paddle.ParamAttr(trainable=trainable),
                default_initializer=nn.initializer.Constant(2 * np.pi / float(p)),
            )  # mu = 2*pi / period for sin/cos function
            for k, (p, trainable) in periods.items()
        }
        self.freqs = paddle.nn.ParameterList(list(self.freqs_dict.values()))

    def forward(self, x: Dict[str, paddle.Tensor]):
        y = {k: v for k, v in x.items()}  # shallow copy to avoid modifying input dict

        for k, w in self.freqs_dict.items():
            y[k] = paddle.concat([paddle.cos(w * x[k]), paddle.sin(w * x[k])], axis=-1)

        return y


class FourierEmbedding(nn.Layer):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, but got {out_features}.")

        self.kernel = self.create_parameter(
            [in_features, out_features // 2],
            default_initializer=nn.initializer.Normal(std=scale),
        )

    def forward(self, x: paddle.Tensor):
        y = paddle.concat(
            [
                paddle.cos(x @ self.kernel),
                paddle.sin(x @ self.kernel),
            ],
            axis=-1,
        )
        return y


class MLP(base.Arch):
    """Multi layer perceptron network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("x", "y", "z").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("u", "v", "w").
        num_layers (int): Number of hidden layers.
        hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size.
            An integer for all layers, or list of integer specify each layer's size.
        activation (str, optional): Name of activation function. Defaults to "tanh".
        skip_connection (bool, optional): Whether to use skip connection. Defaults to False.
        weight_norm (bool, optional): Whether to apply weight norm on parameter(s). Defaults to False.
        input_dim (Optional[int]): Number of input's dimension. Defaults to None.
        output_dim (Optional[int]): Number of output's dimension. Defaults to None.
        periods (Optional[Dict[int, Tuple[float, bool]]]): Period of each input key,
            input in given channel will be period embeded if specified, each tuple of
            periods list is [period, trainable]. Defaults to None.
        fourier (Optional[Dict[str, Union[float, int]]]): Random fourier feature embedding,
            e.g. {'dim': 256, 'sclae': 1.0}. Defaults to None.
        random_weight (Optional[Dict[str, float]]): Mean and std of random weight
            factorization layer, e.g. {"mean": 0.5, "std: 0.1"}. Defaults to None.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.MLP(
        ...     input_keys=("x", "y"),
        ...     output_keys=("u", "v"),
        ...     num_layers=5,
        ...     hidden_size=128
        ... )
        >>> input_dict = {"x": paddle.rand([64, 64, 1]),
        ...               "y": paddle.rand([64, 64, 1])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["u"].shape)
        [64, 64, 1]
        >>> print(output_dict["v"].shape)
        [64, 64, 1]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: Union[int, Tuple[int, ...]],
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
        self.acts = []
        self.periods = periods
        self.fourier = fourier
        if periods:
            self.period_emb = PeriodEmbedding(periods)

        if isinstance(hidden_size, (tuple, list)):
            if num_layers is not None:
                raise ValueError(
                    "num_layers should be None when hidden_size is specified"
                )
        elif isinstance(hidden_size, int):
            if not isinstance(num_layers, int):
                raise ValueError(
                    "num_layers should be an int when hidden_size is an int"
                )
            hidden_size = [hidden_size] * num_layers
        else:
            raise ValueError(
                f"hidden_size should be list of int or int, but got {type(hidden_size)}"
            )

        # initialize FC layer(s)
        cur_size = len(self.input_keys) if input_dim is None else input_dim
        if input_dim is None and periods:
            # period embeded channel(s) will be doubled automatically
            # if input_dim is not specified
            cur_size += len(periods)

        if fourier:
            self.fourier_emb = FourierEmbedding(
                cur_size, fourier["dim"], fourier["scale"]
            )
            cur_size = fourier["dim"]

        for i, _size in enumerate(hidden_size):
            if weight_norm:
                self.linears.append(WeightNormLinear(cur_size, _size))
            elif random_weight:
                self.linears.append(
                    RandomWeightFactorization(
                        cur_size,
                        _size,
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            else:
                self.linears.append(nn.Linear(cur_size, _size))

            # initialize activation function
            self.acts.append(
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(_size)
            )
            # special initialization for certain activation
            # TODO: Adapt code below to a more elegant style
            if activation == "siren":
                if i == 0:
                    act_mod.Siren.init_for_first_layer(self.linears[-1])
                else:
                    act_mod.Siren.init_for_hidden_layer(self.linears[-1])

            cur_size = _size

        self.linears = nn.LayerList(self.linears)
        self.acts = nn.LayerList(self.acts)
        if random_weight:
            self.last_fc = RandomWeightFactorization(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        else:
            self.last_fc = nn.Linear(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
            )

        self.skip_connection = skip_connection

    def forward_tensor(self, x):
        y = x
        skip = None
        for i, linear in enumerate(self.linears):
            y = linear(y)
            if self.skip_connection and i % 2 == 0:
                if skip is not None:
                    skip = y
                    y = y + skip
                else:
                    skip = y
            y = self.acts[i](y)

        y = self.last_fc(y)

        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        if self.periods:
            x = self.period_emb(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)

        if self.fourier:
            y = self.fourier_emb(y)

        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class ModifiedMLP(base.Arch):
    """Modified Multi layer perceptron network.

    Understanding and mitigating gradient pathologies in physics-informed
    neural networks. https://arxiv.org/pdf/2001.04536.pdf.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("x", "y", "z").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("u", "v", "w").
        num_layers (int): Number of hidden layers.
        hidden_size (int): Number of hidden size, an integer for all layers.
        activation (str, optional): Name of activation function. Defaults to "tanh".
        skip_connection (bool, optional): Whether to use skip connection. Defaults to False.
        weight_norm (bool, optional): Whether to apply weight norm on parameter(s). Defaults to False.
        input_dim (Optional[int]): Number of input's dimension. Defaults to None.
        output_dim (Optional[int]): Number of output's dimension. Defaults to None.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.ModifiedMLP(
        ...     input_keys=("x", "y"),
        ...     output_keys=("u", "v"),
        ...     num_layers=5,
        ...     hidden_size=128
        ... )
        >>> input_dict = {"x": paddle.rand([64, 64, 1]),
        ...               "y": paddle.rand([64, 64, 1])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["u"].shape)
        [64, 64, 1]
        >>> print(output_dict["v"].shape)
        [64, 64, 1]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: int,
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
        self.acts = []
        if isinstance(hidden_size, int):
            if not isinstance(num_layers, int):
                raise ValueError("num_layers should be an int")
            hidden_size = [hidden_size] * num_layers
        else:
            raise ValueError(f"hidden_size should be int, but got {type(hidden_size)}")

        # initialize FC layer(s)
        cur_size = len(self.input_keys) if input_dim is None else input_dim
        self.embed_u = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else nn.Linear(cur_size, hidden_size[0])
            ),
            (
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(hidden_size[0])
            ),
        )
        self.embed_v = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else nn.Linear(cur_size, hidden_size[0])
            ),
            (
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(hidden_size[0])
            ),
        )

        for i, _size in enumerate(hidden_size):
            self.linears.append(
                WeightNormLinear(cur_size, _size)
                if weight_norm
                else nn.Linear(cur_size, _size)
            )
            # initialize activation function
            self.acts.append(
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(_size)
            )
            # special initialization for certain activation
            # TODO: Adapt code below to a more elegant style
            if activation == "siren":
                if i == 0:
                    act_mod.Siren.init_for_first_layer(self.linears[-1])
                else:
                    act_mod.Siren.init_for_hidden_layer(self.linears[-1])

            cur_size = _size

        self.linears = nn.LayerList(self.linears)
        self.acts = nn.LayerList(self.acts)
        self.last_fc = nn.Linear(
            cur_size,
            len(self.output_keys) if output_dim is None else output_dim,
        )

        self.skip_connection = skip_connection

    def forward_tensor(self, x):
        u = self.embed_u(x)
        v = self.embed_v(x)

        y = x
        skip = None
        for i, linear in enumerate(self.linears):
            y = linear(y)
            y = self.acts[i](y)
            y = (1 - y) * u + y * v
            if self.skip_connection and i % 2 == 0:
                if skip is not None:
                    skip = y
                    y = y + skip
                else:
                    skip = y

        y = self.last_fc(y)

        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
