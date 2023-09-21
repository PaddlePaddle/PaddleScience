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

from typing import Optional
from typing import Tuple
from typing import Union

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


class FullyConnectedLayers(base.Arch):
    """Fully Connected Layers, core implementation of MLP.

    Args:
        input_dim (int): Number of input's dimension.
        output_dim (int): Number of output's dimension.
        num_layers (int): Number of hidden layers.
        hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size.
            An integer for all layers, or list of integer specify each layer's size.
        activation (str, optional): Name of activation function. Defaults to "tanh".
        skip_connection (bool, optional): Whether to use skip connection. Defaults to False.
        weight_norm (bool, optional): Whether to apply weight norm on parameter(s). Defaults to False.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.FullyConnectedLayers(3, 4, num_layers=5, hidden_size=128)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_size: Union[int, Tuple[int, ...]],
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
    ):
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
                f"hidden_size should be list of int or int"
                f"but got {type(hidden_size)}"
            )

        self.linears = nn.LayerList()
        self.acts = nn.LayerList()

        # initialize FC layer(s)
        cur_size = input_dim
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
            # spetial initialization for certain activation
            # TODO: Adapt code below to a more elegent style
            if activation == "siren":
                if i == 0:
                    act_mod.Siren.init_for_first_layer(self.linears[-1])
                else:
                    act_mod.Siren.init_for_hidden_layer(self.linears[-1])

            cur_size = _size

        self.last_fc = nn.Linear(cur_size, output_dim)

        self.skip_connection = skip_connection

    def forward(self, x):
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


class MLP(FullyConnectedLayers):
    """Multi layer perceptron network derivated by FullyConnectedLayers.
    Which accepts input/output string key(s) for symbolic computation.

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

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.MLP(("x", "y"), ("u", "v"), num_layers=5, hidden_size=128)
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
    ):
        self.input_keys = input_keys
        self.output_keys = output_keys
        super().__init__(
            len(input_keys) if not input_dim else input_dim,
            len(output_keys) if not output_dim else output_dim,
            num_layers,
            hidden_size,
            activation,
            skip_connection,
            weight_norm,
        )

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = super().forward(x)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
