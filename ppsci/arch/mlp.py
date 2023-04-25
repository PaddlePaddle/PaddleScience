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

from typing import Tuple
from typing import Union

import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base


class MLP(base.NetBase):
    """Multi layer perceptron network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ["x", "y", "z"].
        output_keys (Tuple[str, ...]): Name of output keys, such as ["u", "v", "w"].
        num_layers (int): Number of hidden layers.
        hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size.
            An integer for all layers, or list of integer specify each layer's size.
        activation (str, optional): Name of activation function. Defaults to "tanh".
        skip_connection (bool, optional): Whether to use skip connection. Defaults to False.
        weight_norm (bool, optional): Whether to apply weight norm on parameter(s). Defaults to False.

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
        weight_init=None,
        bias_init=None,
        net_special_name: str = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
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

        # initialize FC layer(s)
        cur_size = len(self.input_keys)
        for i, _size in enumerate(hidden_size):
            w_para = paddle.nn.initializer.Assign(weight_init[f"w_{i}"])
            b_para = paddle.nn.initializer.Assign(bias_init[f"b_{i}"])
            self.linears.append(
                nn.Linear(
                    cur_size,
                    _size,
                    weight_attr=paddle.ParamAttr(initializer=w_para),
                    bias_attr=paddle.ParamAttr(initializer=b_para),
                )
            )
            if weight_norm:
                self.linears[-1] = nn.utils.weight_norm(self.linears[-1], dim=1)
            cur_size = _size
        self.linears = nn.LayerList(self.linears)
        w_para = paddle.nn.initializer.Assign(weight_init[f"w_{num_layers}"])
        b_para = paddle.nn.initializer.Assign(bias_init[f"b_{num_layers}"])
        self.last_fc = nn.Linear(
            cur_size,
            len(self.output_keys),
            weight_attr=paddle.ParamAttr(initializer=w_para),
            bias_attr=paddle.ParamAttr(initializer=b_para),
        )

        # initialize activation function
        self.act = act_mod.get_activation(activation)

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
            y = self.act(y)

        y = self.last_fc(y)

        return y

    def forward(self, x):
        if self._input_transform is not None:
            _x = self._input_transform(x)
        else:
            _x = x
        y = self.concat_to_tensor(_x, self.input_keys, axis=1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=1)

        if self._output_transform is not None:
            y = self._output_transform(y, x)
        return y
