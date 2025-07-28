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

# Elasto-Plastic Neural Network (EPNN)

# DEVELOPED AT:
#                     COMPUTATIONAL GEOMECHANICS LABORATORY
#                     DEPARTMENT OF CIVIL ENGINEERING
#                     UNIVERSITY OF CALGARY, AB, CANADA
#                     DIRECTOR: Prof. Richard Wan

# DEVELOPED BY:
#                     MAHDAD EGHBALIAN

# MIT License

# Copyright (c) 2022 Mahdad Eghbalian

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Tuple

import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base


class Epnn(base.Arch):
    """Builds a feedforward network with arbitrary layers.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("x", "y", "z").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("u", "v", "w").
        node_sizes (Tuple[int, ...]): The tuple of node size.
        activations (Tuple[str, ...]): Name of activation functions.
        drop_p (float): The parameter p of nn.Dropout.

    Examples:
        >>> import ppsci
        >>> ann_node_sizes_state = [1, 20]
        >>> model = ppsci.arch.Epnn(
        ...     ("x",),
        ...     ("y",),
        ...     node_sizes=ann_node_sizes_state,
        ...     activations=("leaky_relu",),
        ...     drop_p=0.0,
        ... )
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        node_sizes: Tuple[int, ...],
        activations: Tuple[str, ...],
        drop_p: float,
    ):
        super().__init__()
        self.active_func = [
            act_mod.get_activation(act_name) for act_name in activations
        ]
        self.node_sizes = node_sizes
        self.drop_p = drop_p
        self.layers = []
        self.layers.append(
            nn.Linear(in_features=node_sizes[0], out_features=node_sizes[1])
        )
        layer_sizes = zip(node_sizes[1:-2], node_sizes[2:-1])
        self.layers.extend(
            [nn.Linear(in_features=h1, out_features=h2) for h1, h2 in layer_sizes]
        )
        self.layers.append(
            nn.Linear(
                in_features=node_sizes[-2], out_features=node_sizes[-1], bias_attr=False
            )
        )

        self.layers = nn.LayerList(self.layers)
        self.dropout = nn.Dropout(p=drop_p)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = x[self.input_keys[0]]
        for ilayer in range(len(self.layers)):
            y = self.layers[ilayer](y)
            if ilayer != len(self.layers) - 1:
                y = self.active_func[ilayer + 1](y)
            if ilayer != len(self.layers) - 1:
                y = self.dropout(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
