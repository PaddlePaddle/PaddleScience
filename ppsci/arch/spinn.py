# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import paddle.nn as nn

from ppsci.arch import base
from ppsci.arch.mlp import ModifiedMLP
from ppsci.utils import initializer


class SPINN(base.Arch):
    """Separable Physics-Informed Neural Networks.

    Args:
        input_keys (Tuple[str, ...]): Keys of input variables.
        output_keys (Tuple[str, ...]): Keys of output variables.
        r (int): Number of features for each output dimension.
        num_layers (int): Number of layers.
        hidden_size (Union[int, Tuple[int, ...]]): Size of hidden layer.
        activation (str, optional): Name of activation function.
        skip_connection (bool, optional): Whether to use skip connection.
        weight_norm (bool, optional): Whether to use weight normalization.
        periods (Optional[Dict[int, Tuple[float, bool]]], optional): Periodicity of input variables.
        fourier (Optional[Dict[str, Union[float, int]]], optional): Frequency of input variables.
        random_weight (Optional[Dict[str, float]], optional): Random weight of linear layer.

    Examples:
        >>> from ppsci.arch import SPINN
        >>> model = SPINN(
        ...     input_keys=('x', 'y', 'z'),
        ...     output_keys=('u', 'v'),
        ...     r=32,
        ...     num_layers=4,
        ...     hidden_size=32,
        ... )
        >>> input_dict = {"x": paddle.rand([3, 1]),
        ...               "y": paddle.rand([4, 1]),
        ...               "z": paddle.rand([5, 1])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["u"].shape)
        [3, 4, 5, 1]
        >>> print(output_dict["v"].shape)
        [3, 4, 5, 1]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        r: int,
        num_layers: int,
        hidden_size: Union[int, Tuple[int, ...]],
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
    ):

        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.r = r
        input_dim = len(self.input_keys)

        self.branch_nets = nn.LayerList()
        for i in range(input_dim):
            self.branch_nets.append(
                ModifiedMLP(
                    input_keys=(input_keys[i],),
                    output_keys=("f",),
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    activation=activation,
                    skip_connection=skip_connection,
                    weight_norm=weight_norm,
                    output_dim=r * len(output_keys),
                    periods=periods,
                    fourier=fourier,
                    random_weight=random_weight,
                )
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers(True):
            if isinstance(m, nn.Linear):
                initializer.glorot_normal_(m.weight)
                initializer.zeros_(m.bias)

    def _tensor_contraction(self, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        """Tensor contraction between two tensors along the last channel.

        Args:
            x (Tensor): Input tensor with shape [*N, C].
            y (Tensor): Input tensor with shape [*M, C]

        Returns:
            Tensor: Output tensor with shape [*N, *M, C].
        """
        x_ndim = x.ndim
        y_ndim = y.ndim
        out_dim = x_ndim + y_ndim - 1

        # Align the dimensions of x and y to out_dim
        if x_ndim < out_dim:
            # Add singleton dimensions to x at the end of dimensions
            x = x.unsqueeze([-2] * (out_dim - x_ndim))
        if y_ndim < out_dim:
            # Add singleton dimensions to y at the begin of dimensions
            y = y.unsqueeze([0] * (out_dim - y_ndim))

        # Multiply x and y with implicit broadcasting
        out = x * y

        return out

    def forward_tensor(self, x, y, z) -> List[paddle.Tensor]:
        # forward each dim branch
        feature_f = []
        for i, input_var in enumerate((x, y, z)):
            input_i = {self.input_keys[i]: input_var}
            output_f_i = self.branch_nets[i](input_i)
            feature_f.append(output_f_i["f"])  # [B, r*output_dim]

        output = []
        for i, key in enumerate(self.output_keys):
            st, ed = i * self.r, (i + 1) * self.r
            # do tensor contraction and sum over all branch outputs
            if ed - st == self.r:
                output_i = feature_f[0]
            else:
                output_i = feature_f[0][:, st:ed]

            for j in range(1, len(self.input_keys)):
                if ed - st == self.r:
                    output_ii = feature_f[j]
                else:
                    output_ii = feature_f[j][:, st:ed]
                output_i = self._tensor_contraction(output_i, output_ii)

            output_i = output_i.sum(-1, keepdim=True)
            output.append(output_i)

        return output

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        output = self.forward_tensor(*[x[key] for key in self.input_keys])

        output = {key: output[i] for i, key in enumerate(self.output_keys)}

        if self._output_transform is not None:
            output = self._output_transform(x, output)

        return output
