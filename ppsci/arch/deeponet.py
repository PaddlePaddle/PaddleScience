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

from typing import Tuple
from typing import Union

import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.arch import mlp


class DeepOperatorLayers(base.Arch):
    """Deep operator network, core implementation of `DeepONet`.

    [Lu et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nat Mach Intell, 2021.](https://doi.org/10.1038/s42256-021-00302-5)

    Args:
        trunck_dim (int): Dimension of sampled u(x)(1 for scalar function, >1 for vector function).
        num_loc (int): Number of sampled u(x), i.e. `m` in paper.
        num_features (int): Number of features extracted from u(x), same for y.
        branch_num_layers (int): Number of hidden layers of branch net.
        trunk_num_layers (int): Number of hidden layers of trunk net.
        branch_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of branch net.
            An integer for all layers, or list of integer specify each layer's size.
        trunk_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of trunk net.
            An integer for all layers, or list of integer specify each layer's size.
        branch_skip_connection (bool, optional): Whether to use skip connection for branch net. Defaults to False.
        trunk_skip_connection (bool, optional): Whether to use skip connection for trunk net. Defaults to False.
        branch_activation (str, optional): Name of activation function. Defaults to "tanh".
        trunk_activation (str, optional): Name of activation function. Defaults to "tanh".
        branch_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for branch net. Defaults to False.
        trunk_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for trunk net. Defaults to False.
        use_bias (bool, optional): Whether to add bias on predicted G(u)(y). Defaults to True.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.DeepOperatorLayers(
        ...     1,
        ...     100, 40,
        ...     1, 1,
        ...     40, 40,
        ...     branch_activation="relu", trunk_activation="relu",
        ...     use_bias=True,
        ... )
    """

    def __init__(
        self,
        trunck_dim: int,
        num_loc: int,
        num_features: int,
        branch_num_layers: int,
        trunk_num_layers: int,
        branch_hidden_size: Union[int, Tuple[int, ...]],
        trunk_hidden_size: Union[int, Tuple[int, ...]],
        branch_skip_connection: bool = False,
        trunk_skip_connection: bool = False,
        branch_activation: str = "tanh",
        trunk_activation: str = "tanh",
        branch_weight_norm: bool = False,
        trunk_weight_norm: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.trunck_dim = trunck_dim
        self.branch_net = mlp.FullyConnectedLayers(
            num_loc,
            num_features,
            branch_num_layers,
            branch_hidden_size,
            branch_activation,
            branch_skip_connection,
            branch_weight_norm,
        )

        self.trunk_net = mlp.FullyConnectedLayers(
            trunck_dim,
            num_features,
            trunk_num_layers,
            trunk_hidden_size,
            trunk_activation,
            trunk_skip_connection,
            trunk_weight_norm,
        )
        self.trunk_act = act_mod.get_activation(trunk_activation)

        self.use_bias = use_bias
        if use_bias:
            # register bias to parameter for updating in optimizer and storage
            self.b = self.create_parameter(
                shape=(1,),
                attr=nn.initializer.Constant(0.0),
            )

    def forward(self, u, y):
        # Branch net to encode the input function
        u_features = self.branch_net(u)

        # Trunk net to encode the domain of the output function
        y_features = self.trunk_net(y)
        y_features = self.trunk_act(y_features)

        # Dot product
        G_u = paddle.einsum("bi,bi->b", u_features, y_features)  # [batch_size, ]
        G_u = paddle.reshape(
            G_u, [-1, self.trunck_dim]
        )  # reshape [batch_size, ] to [batch_size, 1]

        # Add bias
        if self.use_bias:
            G_u += self.b

        return G_u


class DeepONet(DeepOperatorLayers):
    """Deep operator network.
    Different from `DeepOperatorLayers`, this class accepts input/output string key(s) for symbolic computation.

    [Lu et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nat Mach Intell, 2021.](https://doi.org/10.1038/s42256-021-00302-5)

    Args:
        u_key (str): Name of function data for input function u(x).
        y_key (str): Name of location data for input function G(u).
        G_key (str): Output name of predicted G(u)(y).
        num_loc (int): Number of sampled u(x), i.e. `m` in paper.
        num_features (int): Number of features extracted from u(x), same for y.
        branch_num_layers (int): Number of hidden layers of branch net.
        trunk_num_layers (int): Number of hidden layers of trunk net.
        branch_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of branch net.
            An integer for all layers, or list of integer specify each layer's size.
        trunk_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of trunk net.
            An integer for all layers, or list of integer specify each layer's size.
        branch_skip_connection (bool, optional): Whether to use skip connection for branch net. Defaults to False.
        trunk_skip_connection (bool, optional): Whether to use skip connection for trunk net. Defaults to False.
        branch_activation (str, optional): Name of activation function. Defaults to "tanh".
        trunk_activation (str, optional): Name of activation function. Defaults to "tanh".
        branch_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for branch net. Defaults to False.
        trunk_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for trunk net. Defaults to False.
        use_bias (bool, optional): Whether to add bias on predicted G(u)(y). Defaults to True.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.DeepONet(
        ...     "u", "y", "G",
        ...     100, 40,
        ...     1, 1,
        ...     40, 40,
        ...     branch_activation="relu", trunk_activation="relu",
        ...     use_bias=True,
        ... )
    """

    def __init__(
        self,
        u_key: str,
        y_key: str,
        G_key: str,
        num_loc: int,
        num_features: int,
        branch_num_layers: int,
        trunk_num_layers: int,
        branch_hidden_size: Union[int, Tuple[int, ...]],
        trunk_hidden_size: Union[int, Tuple[int, ...]],
        branch_skip_connection: bool = False,
        trunk_skip_connection: bool = False,
        branch_activation: str = "tanh",
        trunk_activation: str = "tanh",
        branch_weight_norm: bool = False,
        trunk_weight_norm: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_keys = (u_key, y_key)
        self.output_keys = (G_key,)

        super().__init__(
            1,
            num_loc,
            num_features,
            branch_num_layers,
            trunk_num_layers,
            branch_hidden_size,
            trunk_hidden_size,
            branch_skip_connection,
            trunk_skip_connection,
            branch_activation,
            trunk_activation,
            branch_weight_norm,
            trunk_weight_norm,
            use_bias,
        )

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        G_u = super().forward(x[self.input_keys[0]], x[self.input_keys[1]])
        result_dict = {self.output_keys[0]: G_u}

        if self._output_transform is not None:
            result_dict = self._output_transform(x, result_dict)

        return result_dict
