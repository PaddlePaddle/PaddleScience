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

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import utils

from ppsci.arch import base

# define the high-order finite difference kernels
LALP_OP = [
    [
        [
            [0, 0, -1 / 12, 0, 0],
            [0, 0, 4 / 3, 0, 0],
            [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
            [0, 0, 4 / 3, 0, 0],
            [0, 0, -1 / 12, 0, 0],
        ]
    ]
]

PARTIAL_Y = [
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    ]
]

PARTIAL_X = [
    [
        [
            [0, 0, 1 / 12, 0, 0],
            [0, 0, -8 / 12, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 8 / 12, 0, 0],
            [0, 0, -1 / 12, 0, 0],
        ]
    ]
]


# specific parameters for burgers equation
def _initialize_weights(module):
    if isinstance(module, nn.Conv2D):
        c = 1.0  # 0.5
        initializer = nn.initializer.Uniform(
            -c * np.sqrt(1 / (3 * 3 * 320)), c * np.sqrt(1 / (3 * 3 * 320))
        )
        initializer(module.weight)
    elif isinstance(module, nn.Linear):
        initializer = nn.initializer.Constant(0.0)
        initializer(module.bias)


class PhyCRNet(base.Arch):
    """Physics-informed convolutional-recurrent neural networks.

    Args:
        input_channels (int): The input channels.
        hidden_channels (Tuple[int, ...]): The hidden channels.
        input_kernel_size (Tuple[int, ...]):  The input kernel size(s).
        input_stride (Tuple[int, ...]): The input stride(s).
        input_padding (Tuple[int, ...]): The input padding(s).
        dt (float): The dt parameter.
        num_layers (Tuple[int, ...]): The number of layers.
        upscale_factor (int): The upscale factor.
        step (int, optional): The step(s). Defaults to 1.
        effective_step (Tuple[int, ...], optional): The effective step. Defaults to (1, ).

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.PhyCRNet(
        ...     input_channels=2,
        ...     hidden_channels=[8, 32, 128, 128],
        ...     input_kernel_size=[4, 4, 4, 3],
        ...     input_stride=[2, 2, 2, 1],
        ...     input_padding=[1, 1, 1, 1],
        ...     dt=0.002,
        ...     num_layers=[3, 1],
        ...     upscale_factor=8
        ... )
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Tuple[int, ...],
        input_kernel_size: Tuple[int, ...],
        input_stride: Tuple[int, ...],
        input_padding: Tuple[int, ...],
        dt: float,
        num_layers: Tuple[int, ...],
        upscale_factor: int,
        step: int = 1,
        effective_step: Tuple[int, ...] = (1,),
    ):
        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        # encoder - downsampling
        self.encoder = paddle.nn.LayerList(
            [
                encoder_block(
                    input_channels=self.input_channels[i],
                    hidden_channels=self.hidden_channels[i],
                    input_kernel_size=self.input_kernel_size[i],
                    input_stride=self.input_stride[i],
                    input_padding=self.input_padding[i],
                )
                for i in range(self.num_encoder)
            ]
        )

        # ConvLSTM
        self.ConvLSTM = paddle.nn.LayerList(
            [
                ConvLSTMCell(
                    input_channels=self.input_channels[i],
                    hidden_channels=self.hidden_channels[i],
                    input_kernel_size=self.input_kernel_size[i],
                    input_stride=self.input_stride[i],
                    input_padding=self.input_padding[i],
                )
                for i in range(self.num_encoder, self.num_encoder + self.num_convlstm)
            ]
        )

        # output layer
        self.output_layer = nn.Conv2D(
            2, 2, kernel_size=5, stride=1, padding=2, padding_mode="circular"
        )

        # pixelshuffle - upscale
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        # initialize weights
        self.apply(_initialize_weights)
        initializer_0 = paddle.nn.initializer.Constant(0.0)
        initializer_0(self.output_layer.bias)
        self.enable_transform = True

    def forward(self, x):
        if self.enable_transform:
            if self._input_transform is not None:
                x = self._input_transform(x)
        output_x = x

        self.initial_state = x["initial_state"]
        x = x["input"]
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x

            # encoder
            for encoder in self.encoder:
                x = encoder(x)

            # convlstm
            for i, LSTM in enumerate(self.ConvLSTM):
                if step == 0:
                    (h, c) = LSTM.init_hidden_tensor(
                        prev_state=self.initial_state[i - self.num_encoder]
                    )
                    internal_state.append((h, c))

                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                x, new_c = LSTM(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)

            # output
            x = self.pixelshuffle(x)
            x = self.output_layer(x)

            # residual connection
            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()

            if step in self.effective_step:
                outputs.append(x)

        result_dict = {"outputs": outputs, "second_last_state": second_last_state}
        if self.enable_transform:
            if self._output_transform is not None:
                result_dict = self._output_transform(output_x, result_dict)
        return result_dict


class ConvLSTMCell(nn.Layer):
    """Convolutional LSTM"""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kernel_size,
        input_stride,
        input_padding,
        hidden_kernel_size=3,
        num_features=4,
    ):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = hidden_kernel_size  # Page 9, The convolutional operations in ConvLSTM have 3x3 kernels.
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = (
            num_features  # Page 10, block of different dense layers {4, 3, 4}
        )

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Whi = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        self.Wxf = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Whf = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        self.Wxc = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Whc = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        self.Wxo = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Who = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        initializer_0 = paddle.nn.initializer.Constant(0.0)
        initializer_1 = paddle.nn.initializer.Constant(1.0)

        initializer_0(self.Wxi.bias)
        initializer_0(self.Wxf.bias)
        initializer_0(self.Wxc.bias)
        initializer_1(self.Wxo.bias)

    def forward(self, x, h, c):
        ci = paddle.nn.functional.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = paddle.nn.functional.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * paddle.tanh(self.Wxc(x) + self.Whc(h))
        co = paddle.nn.functional.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * paddle.tanh(cc)
        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return ((prev_state[0]).cuda(), (prev_state[1]).cuda())


class encoder_block(nn.Layer):
    """encoder with CNN"""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kernel_size,
        input_stride,
        input_padding,
    ):
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = utils.weight_norm(
            nn.Conv2D(
                self.input_channels,
                self.hidden_channels,
                self.input_kernel_size,
                self.input_stride,
                self.input_padding,
                bias_attr=None,
                padding_mode="circular",
            )
        )

        self.act = nn.ReLU()

        initializer_0 = paddle.nn.initializer.Constant(0.0)
        initializer_0(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class Conv2DDerivative(nn.Layer):
    def __init__(self, der_filter, resol, kernel_size=3, name=""):
        super(Conv2DDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2D(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            padding=0,
            bias_attr=False,
        )

        # Fixed gradient operator
        self.filter.weight = self.create_parameter(
            shape=self.filter.weight.shape,
            dtype=self.filter.weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    der_filter, dtype=paddle.get_default_dtype(), stop_gradient=True
                )
            ),
        )
        self.filter.weight.stop_gradient = True

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1DDerivative(nn.Layer):
    def __init__(self, der_filter, resol, kernel_size=3, name=""):
        super(Conv1DDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1D(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            padding=0,
            bias_attr=False,
        )

        # Fixed gradient operator
        self.filter.weight = self.create_parameter(
            shape=self.filter.weight.shape,
            dtype=self.filter.weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    der_filter, dtype=paddle.get_default_dtype(), stop_gradient=True
                )
            ),
        )
        self.filter.weight.stop_gradient = True

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Layer):
    """Loss generator for physics loss"""

    def __init__(self, dt, dx):
        """Construct the derivatives, X = Width, Y = Height"""
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2DDerivative(
            der_filter=LALP_OP, resol=(dx**2), kernel_size=5, name="laplace_operator"
        )

        self.dx = Conv2DDerivative(
            der_filter=PARTIAL_X, resol=(dx * 1), kernel_size=5, name="dx_operator"
        )

        self.dy = Conv2DDerivative(
            der_filter=PARTIAL_Y, resol=(dx * 1), kernel_size=5, name="dy_operator"
        )

        # temporal derivative operator
        self.dt = Conv1DDerivative(
            der_filter=[[[-1, 0, 1]]], resol=(dt * 2), kernel_size=3, name="partial_t"
        )

    def get_phy_Loss(self, output):
        # spatial derivatives
        laplace_u = self.laplace(output[1:-1, 0:1, :, :])  # [t,c,h,w]
        laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        u_y = self.dy(output[1:-1, 0:1, :, :])
        v_x = self.dx(output[1:-1, 1:2, :, :])
        v_y = self.dy(output[1:-1, 1:2, :, :])

        # temporal derivative - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.transpose((2, 3, 1, 0))  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape((lenx * leny, 1, lent))
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape((leny, lenx, 1, lent - 2))
        u_t = u_t.transpose((3, 2, 0, 1))  # [step-2, c, height(Y), width(X)]

        # temporal derivative - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.transpose((2, 3, 1, 0))  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape((lenx * leny, 1, lent))
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape((leny, lenx, 1, lent - 2))
        v_t = v_t.transpose((3, 2, 0, 1))  # [step-2, c, height(Y), width(X)]

        u = output[1:-1, 0:1, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]
        v = output[1:-1, 1:2, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]

        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Reynolds number
        R = 200.0

        # 2D burgers eqn
        f_u = u_t + u * u_x + v * u_y - (1 / R) * laplace_u
        f_v = v_t + u * v_x + v * v_y - (1 / R) * laplace_v

        return f_u, f_v
