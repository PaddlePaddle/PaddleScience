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

from math import ceil
from math import sqrt
from typing import Tuple

import paddle

import ppsci.utils.initializer as init
from ppsci.arch import base


class VelocityGenerator(base.Arch):
    """The Generator Of VelocityGAN.
        VelocityGAN is applied to full waveform inversion tasks, and the structure of the model
        comes from https://arxiv.org/abs/2111.02926#

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        dim1 (int, optional): Number of channels in the outermost layers of both encoder and decoder segments. Default is 32.
        dim2 (int, optional): Number of channels in the second set of layers from the outermost in both encoder and decoder segments. Default is 64.
        dim3 (int, optional): Number of channels in the intermediate layers. Default is 128.
        dim4 (int, optional): Number of channels near the bottleneck, just before and after the deepest layer. Default is 256.
        dim5 (int, optional): Number of channels at the bottleneck, the deepest layer in the network. Default is 512.
        sample_spatial (float, optional): Spatial sampling rate of the input, used to dynamically calculate the kernel size in the last encoder layer. Default is 1.0.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> model = ppsci.arch.VelocityGenerator(("input", ), ("output", ))
        >>> input_dict = {"input": paddle.randn((1, 5, 1000, 70))}
        >>> output_dict = model(input_dict) # doctest: +SKIP
        >>> print(output_dict["output"].shape) # doctest: +SKIP
        [1, 1, 70, 70]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        sample_spatial: float = 1.0,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.generator = Generator(
            dim1=dim1,
            dim2=dim2,
            dim3=dim3,
            dim4=dim4,
            dim5=dim5,
            sample_spatial=sample_spatial,
        )

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.generator(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)

        return y


class VelocityDiscriminator(base.Arch):
    """The Discriminator Of VelocityGAN.
        VelocityGAN is applied to full waveform inversion tasks, and the structure of the model
        comes from https://arxiv.org/abs/2111.02926#

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        dim1 (int, optional): The number of output channels for convblock1_1 and convblock1_2. Default is 32.
        dim2 (int, optional): The number of output channels for convblock2_1 and convblock2_2. Default is 64.
        dim3 (int, optional): The number of output channels for convblock3_1 and convblock3_2. Default is 128.
        dim4 (int, optional): The number of output channels for convblock4_1 and convblock4_2. Default is 256.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> model = ppsci.arch.VelocityDiscriminator(("input", ), ("output", ))
        >>> input_dict = {"input": paddle.randn((1, 1, 70, 70))}
        >>> output_dict = model(input_dict) # doctest: +SKIP
        >>> print(output_dict["output"].shape) # doctest: +SKIP
        [1, 1]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.discriminator = Discriminator(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4)

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.discriminator(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)

        return y


class Generator(paddle.nn.Layer):
    """The specific implementation of the generator, which is encapsulated in the VelocityGenerator class.

    Args:
        dim1 (int, optional): Number of channels in the outermost layers of both encoder and decoder segments. Default is 32.
        dim2 (int, optional): Number of channels in the second set of layers from the outermost in both encoder and decoder segments. Default is 64.
        dim3 (int, optional): Number of channels in the intermediate layers. Default is 128.
        dim4 (int, optional): Number of channels near the bottleneck, just before and after the deepest layer. Default is 256.
        dim5 (int, optional): Number of channels at the bottleneck, the deepest layer in the network. Default is 512.
        sample_spatial (float, optional): Spatial sampling rate of the input, used to dynamically calculate the kernel size in the last encoder layer. Default is 1.0.
    """

    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        sample_spatial: float = 1.0,
    ):
        super(Generator, self).__init__()
        self.convblock1 = ConvBlock(
            5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)
        )
        self.convblock2_1 = ConvBlock(
            dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(
            dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(
            dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(
            dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0
        )
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        self.initial_weight()

    def initial_weight(self):
        for _, m in self.named_sublayers():
            if isinstance(m, paddle.nn.Conv2D) or isinstance(
                m, paddle.nn.Conv2DTranspose
            ):
                init.kaiming_uniform_(m.weight, a=sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = paddle.nn.functional.pad(x, pad=[-5, -5, -5, -5], mode="constant", value=0)
        x = self.deconv6(x)
        return x


class Discriminator(paddle.nn.Layer):
    """The specific implementation of the discriminator, which is encapsulated in the VelocityDiscriminator class.

    Args:
        dim1 (int, optional): The number of output channels for convblock1_1 and convblock1_2. Default is 32.
        dim2 (int, optional): The number of output channels for convblock2_1 and convblock2_2. Default is 64.
        dim3 (int, optional): The number of output channels for convblock3_1 and convblock3_2. Default is 128.
        dim4 (int, optional): The number of output channels for convblock4_1 and convblock4_2. Default is 256.
    """

    def __init__(
        self, dim1: int = 32, dim2: int = 64, dim3: int = 128, dim4: int = 256
    ):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)
        self.initial_weight()

    def initial_weight(self):
        for _, m in self.named_sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                init.kaiming_uniform_(m.weight, a=sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.reshape([x.shape[0], -1])
        return x


class ConvBlock(paddle.nn.Layer):
    """A convolution block, including Conv2D, BatchNorm2D, and LeakyReLU"""

    def __init__(
        self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, relu_slop=0.2
    ):
        super(ConvBlock, self).__init__()
        layers = [
            paddle.nn.Conv2D(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            paddle.nn.BatchNorm2D(out_fea),
            paddle.nn.LeakyReLU(negative_slope=relu_slop),
        ]
        self.layers = paddle.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(paddle.nn.Layer):
    """A deconvolution  block, including Conv2DTranspose, BatchNorm2D, and LeakyReLU"""

    def __init__(
        self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0
    ):
        super(DeconvBlock, self).__init__()
        layers = [
            paddle.nn.Conv2DTranspose(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            paddle.nn.BatchNorm2D(out_fea),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        ]
        self.layers = paddle.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(paddle.nn.Layer):
    """A convolution block, including Conv2D, BatchNorm2D, and Tanh"""

    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_Tanh, self).__init__()
        layers = [
            paddle.nn.Conv2D(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            paddle.nn.BatchNorm2D(out_fea),
            paddle.nn.Tanh(),
        ]
        self.layers = paddle.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
