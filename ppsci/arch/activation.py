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


class Stan(nn.Layer):
    """Self-scalable Tanh.
    paper: https://arxiv.org/abs/2204.12589v1

    Args:
        out_features (int, optional): Output features. Defaults to 1.
    """

    def __init__(self, out_features: int = 1):
        super().__init__()
        self.beta = self.create_parameter(
            shape=(out_features,),
            default_initializer=nn.initializer.Constant(1),
        )

    def forward(self, x):
        # TODO: manually broadcast beta to x.shape for preventing backward error yet.
        return F.tanh(x) * (1 + paddle.broadcast_to(self.beta, x.shape) * x)
        # return F.tanh(x) * (1 + self.beta * x)


class Swish(nn.Layer):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = self.create_parameter(
            shape=[],
            default_initializer=paddle.nn.initializer.Constant(beta),
        )

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)


class Cos(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.cos(x)


class Sin(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.sin(x)


class Silu(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


act_func_dict = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "silu": Silu(),
    "sin": Sin(),
    "cos": Cos(),
    "swish": Swish(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity(),
    "stan": Stan,
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
