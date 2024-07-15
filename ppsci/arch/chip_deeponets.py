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


class ChipDeepONets(base.Arch):
    """Multi-branch physics-informed deep operator neural network. The network consists of three branch networks: random heat source, boundary function, and boundary type, as well as a trunk network.

    Args:
        branch_input_keys (Tuple[str, ...]): Name of input data for internal heat source on branch nets.
        BCtype_input_keys (Tuple[str, ...]): Name of input data for boundary types on branch nets.
        BC_input_keys (Tuple[str, ...]): Name of input data for boundary on branch nets.
        trunk_input_keys (Tuple[str, ...]): Name of input data for trunk net.
        output_keys (Tuple[str, ...]): Output name of predicted temperature.
        num_loc (int): Number of sampled input data for internal heat source.
        bctype_loc (int): Number of sampled input data for boundary types.
        BC_num_loc (int): Number of sampled input data for boundary.
        num_features (int): Number of features extracted from trunk net, same for all branch nets.
        branch_num_layers (int): Number of hidden layers of internal heat source on branch nets.
        BC_num_layers (int): Number of hidden layers of boundary on branch nets.
        trunk_num_layers (int): Number of hidden layers of trunk net.
        branch_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of internal heat source on branch nets.
            An integer for all layers, or list of integer specify each layer's size.
        BC_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of boundary on branch nets.
            An integer for all layers, or list of integer specify each layer's size.
        trunk_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of trunk net.
            An integer for all layers, or list of integer specify each layer's size.
        branch_skip_connection (bool, optional): Whether to use skip connection for internal heat source on branch net. Defaults to False.
        BC_skip_connection (bool, optional): Whether to use skip connection for boundary on branch net. Defaults to False.
        trunk_skip_connection (bool, optional): Whether to use skip connection for trunk net. Defaults to False.
        branch_activation (str, optional): Name of activation function for internal heat source on branch net. Defaults to "tanh".
        BC_activation (str, optional): Name of activation function for boundary on branch net. Defaults to "tanh".
        trunk_activation (str, optional): Name of activation function for trunk net. Defaults to "tanh".
        branch_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for internal heat source on branch net. Defaults to False.
        BC_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for boundary on branch net. Defaults to False.
        trunk_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for trunk net. Defaults to False.
        use_bias (bool, optional): Whether to add bias on predicted G(u)(y). Defaults to True.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.ChipDeepONets(
        ...     ('u',),
        ...     ('bc',),
        ...     ('bc_data',),
        ...     ("x",'y'),
        ...     ("T",),
        ...     324,
        ...     1,
        ...     76,
        ...     400,
        ...     9,
        ...     9,
        ...     6,
        ...     256,
        ...     256,
        ...     128,
        ...     branch_activation="swish",
        ...     BC_activation="swish",
        ...     trunk_activation="swish",
        ...     use_bias=True,
        ... )
    """

    def __init__(
        self,
        branch_input_keys: Tuple[str, ...],
        BCtype_input_keys: Tuple[str, ...],
        BC_input_keys: Tuple[str, ...],
        trunk_input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_loc: int,
        bctype_loc: int,
        BC_num_loc: int,
        num_features: int,
        branch_num_layers: int,
        BC_num_layers: int,
        trunk_num_layers: int,
        branch_hidden_size: Union[int, Tuple[int, ...]],
        BC_hidden_size: Union[int, Tuple[int, ...]],
        trunk_hidden_size: Union[int, Tuple[int, ...]],
        branch_skip_connection: bool = False,
        BC_skip_connection: bool = False,
        trunk_skip_connection: bool = False,
        branch_activation: str = "tanh",
        BC_activation: str = "tanh",
        trunk_activation: str = "tanh",
        branch_weight_norm: bool = False,
        BC_weight_norm: bool = False,
        trunk_weight_norm: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.trunk_input_keys = trunk_input_keys
        self.branch_input_keys = branch_input_keys
        self.BCtype_input_keys = BCtype_input_keys
        self.BC_input_keys = BC_input_keys
        self.input_keys = (
            self.trunk_input_keys
            + self.branch_input_keys
            + self.BC_input_keys
            + self.BCtype_input_keys
        )
        self.output_keys = output_keys

        self.branch_net = mlp.MLP(
            self.branch_input_keys,
            ("b",),
            branch_num_layers,
            branch_hidden_size,
            branch_activation,
            branch_skip_connection,
            branch_weight_norm,
            input_dim=num_loc,
            output_dim=num_features,
        )

        self.BCtype_net = mlp.MLP(
            self.BCtype_input_keys,
            ("bctype",),
            BC_num_layers,
            BC_hidden_size,
            BC_activation,
            BC_skip_connection,
            BC_weight_norm,
            input_dim=bctype_loc,
            output_dim=num_features,
        )

        self.BC_net = mlp.MLP(
            self.BC_input_keys,
            ("bc",),
            BC_num_layers,
            BC_hidden_size,
            BC_activation,
            BC_skip_connection,
            BC_weight_norm,
            input_dim=BC_num_loc,
            output_dim=num_features,
        )

        self.trunk_net = mlp.MLP(
            self.trunk_input_keys,
            ("t",),
            trunk_num_layers,
            trunk_hidden_size,
            trunk_activation,
            trunk_skip_connection,
            trunk_weight_norm,
            input_dim=len(self.trunk_input_keys),
            output_dim=num_features,
        )
        self.trunk_act = act_mod.get_activation(trunk_activation)
        self.bc_act = act_mod.get_activation(BC_activation)
        self.branch_act = act_mod.get_activation(branch_activation)

        self.use_bias = use_bias
        if use_bias:
            # register bias to parameter for updating in optimizer and storage
            self.b = self.create_parameter(
                shape=(1,),
                attr=nn.initializer.Constant(0.0),
            )

    def forward(self, x):

        if self._input_transform is not None:
            x = self._input_transform(x)

        # Branch net to encode the input function
        u_features = self.branch_net(x)[self.branch_net.output_keys[0]]
        bc_features = self.BC_net(x)[self.BC_net.output_keys[0]]
        bctype_features = self.BCtype_net(x)[self.BCtype_net.output_keys[0]]
        # Trunk net to encode the domain of the output function
        y_features = self.trunk_net(x)[self.trunk_net.output_keys[0]]
        y_features = self.trunk_act(y_features)
        # Dot product
        G_u = paddle.sum(
            u_features * y_features * bc_features * bctype_features,
            axis=1,
            keepdim=True,
        )
        # Add bias
        if self.use_bias:
            G_u += self.b

        result_dict = {
            self.output_keys[0]: G_u,
        }
        if self._output_transform is not None:
            result_dict = self._output_transform(x, result_dict)

        return result_dict
