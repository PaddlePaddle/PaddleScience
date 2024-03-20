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

from typing import Optional
from typing import Tuple
from typing import Type

import paddle
from paddle import nn

from ppsci.arch import base


def create_layer(
    in_channel,
    out_channel,
    kernel_size,
    weight_norm=True,
    batch_norm=True,
    activation=nn.ReLU,
    convolution=nn.Conv2D,
):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should even number")
    conv = convolution(in_channel, out_channel, kernel_size, padding=kernel_size // 2)
    if weight_norm:
        conv = nn.util.weight_norm(conv)
    layer = []
    layer.append(conv)
    if activation is not None:
        layer.append(activation())
    if batch_norm:
        layer.append(nn.BatchNorm2D(out_channel))
    return nn.Sequential(*layer)


def create_encoder_block(
    in_channel,
    out_channel,
    kernel_size,
    weight_norm=True,
    batch_norm=True,
    activation=nn.ReLU,
    layers=2,
):
    encoder = []
    encoder.append(
        create_layer(
            in_channel,
            out_channel,
            kernel_size,
            weight_norm,
            batch_norm,
            activation,
            nn.Conv2D,
        )
    )
    for i in range(layers - 1):
        encoder.append(
            create_layer(
                out_channel,
                out_channel,
                kernel_size,
                weight_norm,
                batch_norm,
                activation,
                nn.Conv2D,
            )
        )
    return nn.Sequential(*encoder)


def create_decoder_block(
    in_channel,
    out_channel,
    kernel_size,
    weight_norm=True,
    batch_norm=True,
    activation=nn.ReLU,
    layers=2,
    final_layer=False,
):
    decoder = []
    for i in range(layers):
        _in = in_channel
        _out = in_channel
        _batch_norm = batch_norm
        _activation = activation
        if i == 0:
            _in = in_channel * 2
        if i == layers - 1:
            _out = out_channel
            if final_layer:
                _batch_norm = False
                _activation = None
        decoder.append(
            create_layer(
                _in,
                _out,
                kernel_size,
                weight_norm,
                _batch_norm,
                _activation,
                nn.Conv2DTranspose,
            )
        )
    return nn.Sequential(*decoder)


def create_encoder(
    in_channel, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2
):
    encoder = []
    for i in range(len(filters)):
        encoder_layer = create_encoder_block(
            in_channel if i == 0 else filters[i - 1],
            filters[i],
            kernel_size,
            wn,
            bn,
            activation,
            layers,
        )
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)


def create_decoder(
    out_channel,
    filters,
    kernel_size,
    weight_norm=True,
    batch_norm=True,
    activation=nn.ReLU,
    layers=2,
):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(
                filters[i],
                out_channel,
                kernel_size,
                weight_norm,
                batch_norm,
                activation,
                layers,
                final_layer=True,
            )
        else:
            decoder_layer = create_decoder_block(
                filters[i],
                filters[i - 1],
                kernel_size,
                weight_norm,
                batch_norm,
                activation,
                layers,
                final_layer=False,
            )
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)


class UNetEx(base.Arch):
    """U-Net Extension for CFD.

    Reference: [Ribeiro M D, Rehman A, Ahmed S, et al. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks[J]. arXiv preprint arXiv:2004.08826, 2020.](https://arxiv.org/abs/2004.08826)

    Args:
        input_key (str): Name of function data for input.
        output_key (str): Name of function data for output.
        in_channel (int): Number of channels of input.
        out_channel (int): Number of channels of output.
        kernel_size (int, optional): Size of kernel of convolution layer. Defaults to 3.
        filters (Tuple[int, ...], optional): Number of filters. Defaults to (16, 32, 64).
        layers (int, optional): Number of encoders or decoders. Defaults to 3.
        weight_norm (bool, optional): Whether use weight normalization layer. Defaults to True.
        batch_norm (bool, optional): Whether add batch normalization layer. Defaults to True.
        activation (Type[nn.Layer], optional): Name of activation function. Defaults to nn.ReLU.
        final_activation (Optional[Type[nn.Layer]]): Name of final activation function. Defaults to None.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.UNetEx(
        ...     input_key="input",
        ...     output_key="output",
        ...     in_channel=3,
        ...     out_channel=3,
        ...     kernel_size=5,
        ...     filters=(4, 4, 4, 4),
        ...     layers=3,
        ...     weight_norm=False,
        ...     batch_norm=False,
        ...     activation=None,
        ...     final_activation=None,
        ... )
        >>> input_dict = {'input': paddle.rand([4, 3, 4, 4])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict['output']) # doctest: +SKIP
        >>> print(output_dict['output'].shape)
        [4, 3, 4, 4]
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        filters: Tuple[int, ...] = (16, 32, 64),
        layers: int = 3,
        weight_norm: bool = True,
        batch_norm: bool = True,
        activation: Type[nn.Layer] = nn.ReLU,
        final_activation: Optional[Type[nn.Layer]] = None,
    ):
        if len(filters) == 0:
            raise ValueError("The filters shouldn't be empty ")

        super().__init__()
        self.input_keys = (input_key,)
        self.output_keys = (output_key,)
        self.final_activation = final_activation
        self.encoder = create_encoder(
            in_channel,
            filters,
            kernel_size,
            weight_norm,
            batch_norm,
            activation,
            layers,
        )
        decoders = [
            create_decoder(
                1, filters, kernel_size, weight_norm, batch_norm, activation, layers
            )
            for i in range(out_channel)
        ]
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoder:
            x = encoder(x)
            sizes.append(x.shape)
            tensors.append(x)
            x, ind = nn.functional.max_pool2d(x, 2, 2, return_mask=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, x, tensors, indices, sizes):
        y = []
        for _decoder in self.decoders:
            _x = x
            _tensors = tensors[:]
            _indices = indices[:]
            _sizes = sizes[:]
            for decoder in _decoder:
                tensor = _tensors.pop()
                size = _sizes.pop()
                indice = _indices.pop()
                # upsample operations
                _x = nn.functional.max_unpool2d(_x, indice, 2, 2, output_size=size)
                _x = paddle.concat([tensor, _x], axis=1)
                _x = decoder(_x)
            y.append(_x)
        return paddle.concat(y, axis=1)

    def forward(self, x):
        x = x[self.input_keys[0]]
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return {self.output_keys[0]: x}
