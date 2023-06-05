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

from typing import Callable

import paddle
import paddle.nn.functional as F
from paddle import nn


class Swish(nn.Layer):
    r"""
    Swish Activation.

    .. math::

        Swish(x)= \aplpha x \cdot \frac{1}{1 + e^{-x}}

    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = self.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(beta),
        )
        self.add_parameter("beta", self.beta)

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)


def silu(x):
    """numeric stable silu"""
    return x * F.sigmoid(x)


class SILU(nn.Layer):
    r"""
    SILU Activation.

    .. math::

        SILU(x)= x \cdot \frac{1}{1 + e^{-x}}

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return silu(x)


class Cos(nn.Layer):
    r"""
    Cos Activation.

    .. math::

        Cos(x)= cos(x)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.cos(x)


class Sin(nn.Layer):
    r"""
    Sin Activation.

    .. math::

        Sin(x)= sin(x)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.sin(x)


act_func_dict = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "silu": SILU(),
    "sin": Sin(),
    "cos": Cos(),
    "swish": Swish(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity(),
}


def get_activation(act_name: str) -> Callable:
    """Get activation function according to act_name.

    Args:
        act_name (str): Name of activation, such as "tanh".

    Returns:
        Callable: Paddle activation function.
    """
    if act_name.lower() not in act_func_dict:
        raise ValueError(f"act_name({act_name}) not found in act_func_dict")

    return act_func_dict[act_name.lower()]
