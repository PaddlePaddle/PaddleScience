# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as Initializer


class SpectralConv1d(nn.Layer):
    """
    1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)

        real = paddle.rand(shape=[in_channels, out_channels, modes1])
        real.stop_gradient = False
        img = paddle.rand(shape=[in_channels, out_channels, modes1])
        img.stop_gradient = False
        self.weights1_real = self.create_parameter(
            [in_channels, out_channels, self.modes1],
            attr=Initializer.Assign(self.scale * real),
        )
        self.weights1_imag = self.create_parameter(
            [in_channels, out_channels, self.modes1],
            attr=Initializer.Assign(self.scale * img),
        )

    def compl_mul1d(self, op1, op2_real, op2_imag):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        eq = "bix,iox->box"
        op1_real = op1.real()
        op1_imag = op1.imag()
        result_real = paddle.unsqueeze(
            paddle.einsum(eq, op1_real, op2_real)
            - paddle.einsum(eq, op1_imag, op2_imag),
            axis=-1,
        )
        result_imag = paddle.unsqueeze(
            paddle.einsum(eq, op1_real, op2_imag)
            + paddle.einsum(eq, op1_imag, op2_real),
            axis=-1,
        )
        result_complex = paddle.as_complex(
            paddle.concat([result_real, result_imag], axis=-1)
        )
        return result_complex

    def forward(self, x, output_size=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft_real = paddle.zeros(
            [batchsize, self.out_channels, x.shape[-1] // 2 + 1], dtype="float32"
        )
        out_ft_img = paddle.zeros(
            [batchsize, self.out_channels, x.shape[-1] // 2 + 1], dtype="float32"
        )
        out_ft = paddle.complex(out_ft_real, out_ft_img)

        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1_real, self.weights1_imag
        )

        # Return to physical space
        if output_size is None:
            x = paddle.fft.irfft(out_ft, n=x.shape[-1])
        else:
            x = paddle.fft.irfft(out_ft, n=output_size)

        return x


class FNO1d(nn.Layer):
    """The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
         W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .

    Args:
        input_key (Tuple[str, ...], optional): Key to get the input tensor from the dict. Defaults to ("intput",).
        output_key (Tuple[str, ...], optional): Key to save the output tensor into the dict. Defaults to ("output",).
        modes (int, optional, optional): Number of Fourier modes to compute, it should be the same as
            that in fft part of the code below. Defaults to 64.
        width (int, optional, optional): Number of channels in each Fourier layer. Defaults to 64.
        padding (int, optional, optional): How many zeros to pad to the input Tensor. Defaults to 100.
        input_channel (int, optional, optional): Number of channels of the input tensor. Defaults to 2.
        output_np (int, optional, optional): Number of points to sample the solution. Defaults to 2001.

    Examples:
        >>> model = ppsci.arch.FNO1d()
        >>> input_data = paddle.randn([100, 2001, 2])
        >>> input_dict = {"input": input_data}
        >>> out_dict = model(input_dict)
        >>> for k, v in out_dict.items():
        ...     print(k, v.shape)
        output [100, 1]
    """

    def __init__(
        self,
        input_key=("input",),
        output_key=("output",),
        modes=64,
        width=64,
        padding=100,
        input_channel=2,
        output_np=2001,
    ):
        super().__init__()
        self.input_keys = input_key
        self.output_keys = output_key

        self.output_np = output_np
        self.modes1 = modes
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(input_channel, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1D(self.width, self.width, 1)
        self.w1 = nn.Conv1D(self.width, self.width, 1)
        self.w2 = nn.Conv1D(self.width, self.width, 1)
        self.w3 = nn.Conv1D(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def _functional_pad(self, x, pad, mode="constant", value=0.0, data_format="NCL"):
        if len(x.shape) * 2 == len(pad) and mode == "constant":
            pad = (
                paddle.to_tensor(pad, dtype="float32")
                .reshape((-1, 2))
                .flip([0])
                .flatten()
                .tolist()
            )
        return F.pad(x, pad, mode, value, data_format)

    def forward(self, x):
        x = x[self.input_keys[0]]
        # Dict
        x = self.fc0(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        # pad the domain if input is non-periodic
        x = self._functional_pad(x, [0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x=x, approximate=False)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x, approximate=False)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x, approximate=False)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x, approximate=False)

        x = x[..., : -self.padding]
        x1 = self.conv4(x, self.output_np)
        x2 = F.interpolate(x, size=[self.output_np], mode="linear", align_corners=True)
        x = x1 + x2
        # Change the x-dimension to (batch, channel, 2001)
        x = x.transpose(perm=[0, 2, 1])
        x = self.fc1(x)
        x = F.gelu(x, approximate=False)
        x = self.fc2(x)

        return {self.output_keys[0]: x}
