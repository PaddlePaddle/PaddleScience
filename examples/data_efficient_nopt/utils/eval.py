import math

import paddle
from paddle import nn


# https://github.com/erichson/SuperBench/blob/3719ef9010dc081c3f8e9644813764ef56420fc9/eval.py#L123
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=""):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) // 2)
        self.filter = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            padding=1,
            bias=False,
        )  # TODO:

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(
            paddle.FloatTensor(DerFilter), requires_grad=False
        )

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class LossGenerator(nn.Module):
    def __init__(self, dx=2.0 * math.pi / 2048.0, kernel_size=3, device=None):
        super(LossGenerator, self).__init__()

        self.delta_x = paddle.to_tensor(dx)

        # https://en.wikipedia.org/wiki/Finite_difference_coefficient
        self.filter_y4 = [
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

        self.filter_x4 = [
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

        self.filter_x2 = [[[[0, -1 / 2, 0], [0, 0, 0], [0, 1 / 2, 0]]]]

        self.filter_y2 = [[[[0, 0, 0], [-1 / 2, 0, 1 / 2], [0, 0, 0]]]]

        if kernel_size == 5:
            self.dx = Conv2dDerivative(
                DerFilter=self.filter_x4,
                resol=self.delta_x,
                kernel_size=5,
                name="dx_operator",
            )
            self.dy = Conv2dDerivative(
                DerFilter=self.filter_y4,
                resol=self.delta_x,
                kernel_size=5,
                name="dy_operator",
            )
        elif kernel_size == 3:
            self.dx = Conv2dDerivative(
                DerFilter=self.filter_x2,
                resol=self.delta_x,
                kernel_size=3,
                name="dx_operator",
            )
            self.dy = Conv2dDerivative(
                DerFilter=self.filter_y2,
                resol=self.delta_x,
                kernel_size=3,
                name="dy_operator",
            )
        if device is not None:
            self.dx = self.dx.to(device)
            self.dy = self.dy.to(device)

    def get_div_loss(self, output):
        """compute divergence loss"""
        u = output[:, 0:1, :, :]
        # bu,xu,yu = u.shape
        # u = u.reshape(bu,1,xu,yu)

        v = output[:, 1:2, :, :]
        # bv,xv,yv = v.shape
        # v = v.reshape(bv,1,xv,yv)

        # w = output[:,0,:,:]
        u_x = self.dx(u)
        v_y = self.dy(v)
        # div
        div = u_x + v_y

        return div
