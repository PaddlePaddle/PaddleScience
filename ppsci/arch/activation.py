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
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = paddle.create_parameter(
            shape=[0],
            default_initializer=paddle.nn.initializer.Constant(beta),
        )

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)


def silu(x):
    """numeric stable silu"""
    return x * F.sigmoid(x)


act_func_dict = {
    "elu": F.elu,
    "relu": F.relu,
    "selu": F.selu,
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
    "sin": paddle.sin,
    "cos": paddle.cos,
    "swish": Swish(),
    "silu": silu,
    "tanh": F.tanh,
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
