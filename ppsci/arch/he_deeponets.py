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


class HEDeepONets(base.Arch):
    """Physical information deep operator networks.

    Args:
        heat_input_keys (Tuple[str, ...]): Name of input data for heat boundary.
        cold_input_keys (Tuple[str, ...]): Name of input data for cold boundary.
        trunk_input_keys (Tuple[str, ...]): Name of input data for trunk net.
        output_keys (Tuple[str, ...]): Output name of predicted temperature.
        heat_num_loc (int): Number of sampled input data for heat boundary.
        cold_num_loc (int): Number of sampled input data for cold boundary.
        num_features (int): Number of features extracted from heat boundary, same for cold boundary and trunk net.
        branch_num_layers (int): Number of hidden layers of branch net.
        trunk_num_layers (int): Number of hidden layers of trunk net.
        branch_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of branch net.
            An integer for all layers, or list of integer specify each layer's size.
        trunk_hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size of trunk net.
            An integer for all layers, or list of integer specify each layer's size.
        branch_skip_connection (bool, optional): Whether to use skip connection for branch net. Defaults to False.
        trunk_skip_connection (bool, optional): Whether to use skip connection for trunk net. Defaults to False.
        branch_activation (str, optional): Name of activation function for branch net. Defaults to "tanh".
        trunk_activation (str, optional): Name of activation function for trunk net. Defaults to "tanh".
        branch_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for branch net. Defaults to False.
        trunk_weight_norm (bool, optional): Whether to apply weight norm on parameter(s) for trunk net. Defaults to False.
        use_bias (bool, optional): Whether to add bias on predicted G(u)(y). Defaults to True.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.HEDeepONets(
        ...     ('qm_h',),
        ...     ('qm_c',),
        ...     ("x",'t'),
        ...     ("T_h",'T_c','T_w'),
        ...     1,
        ...     1,
        ...     100,
        ...     9,
        ...     6,
        ...     256,
        ...     128,
        ...     branch_activation="swish",
        ...     trunk_activation="swish",
        ...     use_bias=True,
        ... )
    """

    def __init__(
        self,
        heat_input_keys: Tuple[str, ...],
        cold_input_keys: Tuple[str, ...],
        trunk_input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        heat_num_loc: int,
        cold_num_loc: int,
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
        self.trunk_input_keys = trunk_input_keys
        self.heat_input_keys = heat_input_keys
        self.cold_input_keys = cold_input_keys
        self.input_keys = (
            self.trunk_input_keys + self.heat_input_keys + self.cold_input_keys
        )
        self.output_keys = output_keys
        self.num_features = num_features

        self.heat_net = mlp.MLP(
            self.heat_input_keys,
            ("h",),
            branch_num_layers,
            branch_hidden_size,
            branch_activation,
            branch_skip_connection,
            branch_weight_norm,
            input_dim=heat_num_loc,
            output_dim=num_features * len(self.output_keys),
        )

        self.cold_net = mlp.MLP(
            self.cold_input_keys,
            ("c",),
            branch_num_layers,
            branch_hidden_size,
            branch_activation,
            branch_skip_connection,
            branch_weight_norm,
            input_dim=cold_num_loc,
            output_dim=num_features * len(self.output_keys),
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
            output_dim=num_features * len(self.output_keys),
        )
        self.trunk_act = act_mod.get_activation(trunk_activation)
        self.heat_act = act_mod.get_activation(branch_activation)
        self.cold_act = act_mod.get_activation(branch_activation)

        self.use_bias = use_bias
        if use_bias:
            # register bias to parameter for updating in optimizer and storage
            self.b = self.create_parameter(
                shape=(len(self.output_keys),),
                attr=nn.initializer.Constant(0.0),
            )

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        # Branch net to encode the input function
        heat_features = self.heat_net(x)[self.heat_net.output_keys[0]]
        cold_features = self.cold_net(x)[self.cold_net.output_keys[0]]
        # Trunk net to encode the domain of the output function
        y_features = self.trunk_net(x)[self.trunk_net.output_keys[0]]
        y_features = self.trunk_act(y_features)
        # Dot product
        G_u_h = paddle.sum(
            heat_features[:, : self.num_features]
            * y_features[:, : self.num_features]
            * cold_features[:, : self.num_features],
            axis=1,
            keepdim=True,
        )
        G_u_c = paddle.sum(
            heat_features[:, self.num_features : 2 * self.num_features]
            * y_features[:, self.num_features : 2 * self.num_features]
            * cold_features[:, self.num_features : 2 * self.num_features],
            axis=1,
            keepdim=True,
        )
        G_u_w = paddle.sum(
            heat_features[:, 2 * self.num_features :]
            * y_features[:, 2 * self.num_features :]
            * cold_features[:, 2 * self.num_features :],
            axis=1,
            keepdim=True,
        )
        # Add bias
        if self.use_bias:
            G_u_h += self.b[0]
            G_u_c += self.b[1]
            G_u_w += self.b[2]

        result_dict = {
            self.output_keys[0]: G_u_h,
            self.output_keys[1]: G_u_c,
            self.output_keys[2]: G_u_w,
        }
        if self._output_transform is not None:
            result_dict = self._output_transform(x, result_dict)

        return result_dict
