# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
MLP模块 - ChannelMLP和LinearChannelMLP实现
输入: paddle.Tensor | 输出: paddle.Tensor | 地位: 基础组件，被AGNO/GeomEmb等使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from typing import Callable
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ChannelMLP(nn.Layer):
    """
    Channel-wise MLP with configurable hidden layers.

    Parameters
    ----------
    in_channels : int
        Input channel dimension
    hidden_channels : int
        Hidden layer dimension
    out_channels : int
        Output channel dimension
    n_layers : int, default 2
        Number of hidden layers (not counting input/output)
    activation : str, default 'gelu'
        Activation function: 'gelu', 'relu', 'silu'
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()

        self.layers = nn.LayerList()

        # Input layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))

        # Output layer
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        # Activation function
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.Silu()
        else:
            self.act = nn.GELU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class LinearChannelMLP(nn.Layer):
    """
    Linear Channel MLP with flexible layer configuration.

    Used as kernel function in AGNO layer.

    Parameters
    ----------
    layers : List[int]
        Layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
    non_linearity : Callable, default F.gelu
        Activation function
    """

    def __init__(self, layers: List[int], non_linearity: Callable = F.gelu):
        super().__init__()

        self.n_layers = len(layers) - 1
        self.non_linearity = non_linearity

        self.linears = nn.LayerList()
        for i in range(self.n_layers):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < self.n_layers - 1:  # No activation on last layer
                x = self.non_linearity(x)
        return x
