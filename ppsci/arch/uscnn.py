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

import numpy as np
from paddle import nn

import ppsci
from ppsci.arch import base


class USCNN(base.Arch):
    """Physics-informed convolutional neural networks.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("coords").
        output_keys (Tuple[str, ...]):Name of output keys, such as ("outputV").
        hidden_size (Union[int, Tuple[int, ...]]): the hidden channel for convolutional layers
        h (float): the spatial step
        nx (int):  the number of grids along x-axis
        ny (int): the number of grids along y-axis
        nvar_in (int, optional):  input channel. Defaults to 1.
        nvar_out (int, optional): output channel. Defaults to 1.
        pad_singleside (int, optional): pad for hard boundary constraint. Defaults to 1.
        k (int, optional): kernel_size. Defaults to 5.
        s (int, optional): stride. Defaults to 1.
        p (int, optional): padding. Defaults to 2.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.USCNN(
        ...     ["coords"],
        ...     ["outputV"],
        ...     [16, 32, 16],
        ...     h=0.01,
        ...     ny=19,
        ...     nx=84,
        ...     nvar_in=2,
        ...     nvar_out=1,
        ...     pad_singleside=1,
        ... )
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        hidden_size: Union[int, Tuple[int, ...]],
        h: float,
        nx: int,
        ny: int,
        nvar_in: int = 1,
        nvar_out: int = 1,
        pad_singleside: int = 1,
        k: int = 5,
        s: int = 1,
        p: int = 2,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.nvar_in = nvar_in
        self.nvar_out = nvar_out
        self.k = k
        self.s = s
        self.p = p
        self.deltaX = h
        self.nx = nx
        self.ny = ny
        self.pad_singleside = pad_singleside
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode="bicubic")
        self.conv1 = nn.Conv2D(
            self.nvar_in, hidden_size[0], kernel_size=k, stride=s, padding=p
        )
        self.conv2 = nn.Conv2D(
            hidden_size[0], hidden_size[1], kernel_size=k, stride=s, padding=p
        )
        self.conv3 = nn.Conv2D(
            hidden_size[1], hidden_size[2], kernel_size=k, stride=s, padding=p
        )
        self.conv4 = nn.Conv2D(
            hidden_size[2], self.nvar_out, kernel_size=k, stride=s, padding=p
        )
        self.pixel_shuffle = nn.PixelShuffle(1)
        self.apply(self.init_weights)
        self.udfpad = nn.Pad2D(
            [pad_singleside, pad_singleside, pad_singleside, pad_singleside], value=0
        )

    def init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            bound = 1 / np.sqrt(np.prod(m.weight.shape[1:]))
            ppsci.utils.initializer.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                ppsci.utils.initializer.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.US(y)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.pixel_shuffle(self.conv4(y))

        y = self.udfpad(y)
        y = y[:, 0, :, :].reshape([y.shape[0], 1, y.shape[2], y.shape[3]])
        y = self.split_to_dict(y, self.output_keys)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
