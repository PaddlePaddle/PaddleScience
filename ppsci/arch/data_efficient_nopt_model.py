# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# refs: https://github.com/delta-lab-ai/data_efficient_nopt

""" From PINO: https://github.com/devzhk/PINO/blob/master/models/basics.py """
from collections import OrderedDict
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange

try:
    from timm.models.layers import drop_path
    from timm.models.layers import to_2tuple
    from timm.models.layers import trunc_normal_
    from timm.models.layers import trunc_normal_ as __call_trunc_normal_
except ImportError:
    pass

import logging
import math
import os
from typing import List

import paddle.tensor as Tensor
from ruamel.yaml import YAML
from tqdm import tqdm


def _get_act(activation):
    if activation == "tanh":
        func = F.tanh
    elif activation == "gelu":
        func = F.gelu
    elif activation == "relu":
        func = F.relu_
    elif activation == "elu":
        func = F.elu_
    elif activation == "leaky_relu":
        func = F.leaky_relu_
    else:
        raise ValueError(f"{activation} is not supported")
    return func


def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return paddle.einsum("bix,iox->box", a, b)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return paddle.einsum("bixy,ioxy->boxy", a, b)


def compl_mul3d(a, b):
    return paddle.einsum("bixyz,ioxyz->boxyz", a, b)


def compl_mul2d_v2(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    tmp = paddle.einsum("bixys,ioxyt->stboxy", a, b)
    return paddle.stack(
        [
            tmp[0, 0, :, :, :, :] - tmp[1, 1, :, :, :, :],
            tmp[1, 0, :, :, :, :] + tmp[0, 1, :, :, :, :],
        ],
        axis=-1,
    )


################################################################
# 1d fourier layer
################################################################


class SpectralConv1d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.to_tensor(
                self.scale * paddle.rand([in_channels, out_channels, self.modes1, 2])
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=[2])

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            [batchsize, self.in_channels, x.size(-1) // 2 + 1], dtype=paddle.complex64
        )
        out_ft[:, :, : self.modes1] = compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = paddle.fft.irfft(out_ft, s=[x.size(-1)], axis=[2])
        return x


################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand(
                [in_channels, out_channels, self.modes1, self.modes2],
                dtype=paddle.complex64,
            )
        )
        self.weights2 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand(
                [in_channels, out_channels, self.modes1, self.modes2],
                dtype=paddle.complex64,
            )
        )

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=[2, 3])

        if gridy is None:
            # Multiply relevant Fourier modes
            out_ft = paddle.zeros(
                [batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1],
                dtype=paddle.complex64,
            )
            out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
                x_ft[:, :, : self.modes1, : self.modes2], self.weights1
            )
            out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
                x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
            )

            # Return to physical space
            x = paddle.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), axes=[2, 3])
        else:
            factor1 = compl_mul2d(
                x_ft[:, :, : self.modes1, : self.modes2], self.weights1
            )
            factor2 = compl_mul2d(
                x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
            )
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (
                size1 * size2
            )
        return x

    def ifft2d(self, gridy, coeff1, coeff2, k1, k2):

        # y (batch, N, 2) locations in [0,1]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        # device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2 - 1

        # wavenumber (m1, m2)
        k_x1 = (
            paddle.concat(
                (
                    paddle.arange(start=0, end=k1, step=1),
                    paddle.arange(start=-(k1), end=0, step=1),
                ),
                0,
            )
            .reshape([m1, 1])
            .repeat([1, m2])
        )
        k_x2 = (
            paddle.concat(
                (
                    paddle.arange(start=0, end=k2, step=1),
                    paddle.arange(start=-(k2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape([1, m2])
            .repeat([m1, 1])
        )

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = paddle.outer(gridy[:, :, 0].view(-1), k_x1.view(-1)).reshape(
            [batchsize, N, m1, m2]
        )
        K2 = paddle.outer(gridy[:, :, 1].view(-1), k_x2.view(-1)).reshape(
            [batchsize, N, m1, m2]
        )
        K = K1 + K2

        # basis (N, m1, m2)
        basis = paddle.exp(1j * 2 * np.pi * K)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:, :, 1:, 1:].flip(-1, -2).conj()
        coeff4 = paddle.concat(
            [
                coeff1[:, :, 0:1, 1:].flip(-1).conj(),
                coeff2[:, :, :, 1:].flip(-1, -2).conj(),
            ],
            axis=-2,
        )
        coeff12 = paddle.concat([coeff1, coeff2], axis=-2)
        coeff43 = paddle.concat([coeff4, coeff3], axis=-2)
        coeff = paddle.concat([coeff12, coeff43], axis=-1)

        # Y (batch, channels, N)
        Y = paddle.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y


class SpectralConv2dV2(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand([in_channels, out_channels, self.modes1, self.modes2, 2])
        )
        # self.weights1 = paddle.base.framework.EagerParamBase.from_tensor(self.scale * paddle.rand([in_channels, out_channels, self.modes1+1, self.modes2, 2]))
        self.weights2 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand([in_channels, out_channels, self.modes1, self.modes2, 2])
        )

    def forward(self, x: paddle.Tensor):
        size_0 = x.shape[-2]
        size_1 = x.shape[-1]
        batchsize = x.shape[0]
        # dtype = x.dtype

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft2(x.astype(paddle.float32), axes=(-2, -1), norm="ortho")
        x_ft = paddle.as_real(x_ft)

        out_ft = paddle.zeros(
            [batchsize, self.out_channels, size_0, size_1 // 2 + 1, 2]
        )
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d_v2(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d_v2(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        out_ft = paddle.as_complex(out_ft)

        # Return to physical space
        x = paddle.fft.irfft2(out_ft, axes=(-2, -1), norm="ortho", s=(size_0, size_1))

        return x


class SpectralConv3d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand(
                [in_channels, out_channels, self.modes1, self.modes2, self.modes3],
                dtype=paddle.complex64,
            )
        )
        self.weights2 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand(
                [in_channels, out_channels, self.modes1, self.modes2, self.modes3],
                dtype=paddle.complex64,
            )
        )
        self.weights3 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand(
                [in_channels, out_channels, self.modes1, self.modes2, self.modes3],
                dtype=paddle.complex64,
            )
        )
        self.weights4 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand(
                [in_channels, out_channels, self.modes1, self.modes2, self.modes3],
                dtype=paddle.complex64,
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=[2, 3, 4])
        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            [batchsize, self.out_channels, x.size(2), x.size(3), x.size(4) // 2 + 1],
            dtype=paddle.complex64,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = paddle.fft.irfftn(
            out_ft, s=(x.size(2), x.size(3), x.size(4)), axes=[2, 3, 4]
        )
        return x


class FourierBlock(nn.Layer):
    def __init__(
        self, in_channels, out_channels, modes1, modes2, modes3, activation="tanh"
    ):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv1D(in_channels, out_channels, 1)
        if activation == "tanh":
            self.activation = paddle.tanh_
        elif activation == "gelu":
            self.activation = nn.GELU
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError(f"{activation} is not supported")

    def forward(self, x):
        """
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        """
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(
            x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4]
        )
        if self.activation is not None:
            out = self.activation(out)
        return out


def set_activation(activation):
    if activation == "identity":
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        print("WARNING: invalid activation function!")
        return -1


class FeedForward(nn.Layer):
    """An n-layer-feed-forward-layer module"""

    def __init__(self, in_dim=2, out_dim=1, depth=5, hidden_dim=50, activation="tanh"):
        super().__init__()
        self.depth = depth
        self.activation = set_activation(activation)
        self.ff_in = nn.Linear(in_dim, hidden_dim)
        self.linears = nn.LayerList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(self.depth - 2)]
        )
        self.ff_out = nn.Linear(hidden_dim, out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier Normal Initialization"""
        if isinstance(m, nn.Linear):
            init_xaiverNormal = paddle.nn.initializer.XavierNormal(gain=1.0)
            init_xaiverNormal(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant()
                init_constant(m.bias)

    def forward(self, x):
        x = self.ff_in(x)
        x = self.activation(x)
        for i in range(self.depth - 2):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.ff_out(x)
        return x


def ffn_pinns(params):
    return FeedForward(
        in_dim=params.in_dim,
        out_dim=params.out_dim,
        depth=params.depth,
        hidden_dim=params.hidden_dim,
        activation="tanh",
    )


def add_padding(x, num_pad):
    if max(num_pad) > 0:
        res = F.pad(x, (num_pad[0], num_pad[1]), "constant", 0)
    else:
        res = x
    return res


def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(
            x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), "constant", 0.0
        )
    else:
        res = x
    return res


def remove_padding(x, num_pad):
    if max(num_pad) > 0:
        res = x[..., num_pad[0] : -num_pad[1]]
    else:
        res = x
    return res


def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0] : -num_pad1[1], num_pad2[0] : -num_pad2[1]]
    else:
        res = x
    return res


def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    else:
        raise ValueError(f"{act} is not supported")
    return func


class FNO3d_Backbone(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width=16,
        layers=None,
        in_dim=4,
        act="gelu",
        pad_ratio=[0.0, 0.0],
    ):
        """
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        """
        super(FNO3d_Backbone, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions."

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.LayerList(
            [
                SpectralConv3d(in_size, out_size, mode1_num, mode2_num, mode3_num)
                for in_size, out_size, mode1_num, mode2_num, mode3_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3
                )
            ]
        )

        self.ws = nn.LayerList(
            [
                nn.Conv1D(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.act = _get_act(act)

    def forward(self, x):
        """
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            feature: (batchsize, layers[-1], x_grid, y_grid, t_grid)

        """
        size_z = x.shape[-2]
        if max(self.pad_ratio) > 0:
            num_pad = [round(size_z * i) for i in self.pad_ratio]
        else:
            num_pad = [0.0, 0.0]
        length = len(self.ws)
        batchsize = x.shape[0]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = add_padding(x, num_pad=num_pad)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y, size_z
            )
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding(x, num_pad=num_pad)
        return x


class FNO3d(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width=16,
        fc_dim=128,
        layers=None,
        in_dim=4,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.0, 0.0],
        num_demos=0,
    ):
        """
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        """
        super(FNO3d, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions."

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers

        self.backbone = FNO3d_Backbone(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            layers=layers,
            in_dim=in_dim,
            act=act,
            pad_ratio=pad_ratio,
        )

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

        self.num_demos = num_demos
        # if self.num_demos and self.num_demos > 0:
        #     self.lossgen = LossGenerator(dx=2.0*math.pi/512., kernel_size=3) # TODO:

    def forward(self, x):
        """
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        """
        x = self.backbone(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward_icl(self, x, demo_xs, demo_ys, use_tqdm=False):
        raise NotImplementedError("not implemented yet")


class FNO3d_MAE(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width=16,
        fc_dim=128,
        layers=None,
        in_dim=4,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.0, 0.0],
    ):
        """
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        """
        super(FNO3d_MAE, self).__init__()
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions."

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers

        self.encoder = FNO3d_Backbone(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            layers=layers,
            in_dim=in_dim,
            act=act,
            pad_ratio=pad_ratio,
        )
        self.encoder_to_decoder = nn.Linear(layers[-1], layers[-1])
        self.decoder = FNO3d_Backbone(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            layers=layers[:-1] + [in_dim],
            in_dim=layers[0],
            act=act,
            pad_ratio=pad_ratio,
        )

    def forward(self, x, mask):
        """
        x: (b, h, w, t, 4)
        """
        # B, C, H, W = x.shape
        x_enc = self.encoder(x * mask)
        x_enc = self.encoder_to_decoder(x_enc.permute(0, 2, 3, 4, 1))
        x_dec = self.decoder(x_enc).permute(0, 2, 3, 4, 1)
        return x_dec


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (B, num_windows, C, window_size, window_size)
    """
    if len(x.shape) == 4:
        B, C, H, W = x.shape
        x = x.view([B, C, H // window_size, window_size, W // window_size, window_size])
        windows = x.transpose([0, 2, 4, 1, 3, 5]).view(
            [B, -1, C, window_size, window_size]
        )  # B, n_win*n_win, C, win_s, win_s
    elif len(x.shape) == 5:
        B, J, C, H, W = x.shape
        # x = x.view(B*J, C, H, W)
        x = x.view(
            [B, J, C, H // window_size, window_size, W // window_size, window_size]
        )
        windows = x.transpose([0, 1, 3, 5, 2, 4, 6]).view(
            [B, J, -1, C, window_size, window_size]
        )  # B, J, n_win*n_win, C, win_s, win_s
    return windows


# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L60
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B, num_windows, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = windows.shape[0]
    if len(windows.shape) == 4:
        # B, n_win_sq, self.win_s, self.win_s
        x = windows.view(
            [B, H // window_size, W // window_size, window_size, window_size]
        )
        x = x.transpose([0, 1, 3, 2, 4]).view([B, H, W])
    if len(windows.shape) == 5:
        # B, n_win_sq, C, self.win_s, self.win_s
        x = windows.view(
            [B, H // window_size, W // window_size, -1, window_size, window_size]
        )
        x = x.transpose([0, 3, 1, 4, 2, 5]).contiguous().view([B, -1, H, W])
    elif len(windows.shape) == 6:
        # B, J, n_win_sq, C, self.win_s, self.win_s
        J = windows.shape[1]
        x = windows.view(
            [B, J, H // window_size, W // window_size, -1, window_size, window_size]
        )
        x = x.transpose([0, 1, 4, 2, 5, 3, 6]).contiguous().view([B, J, -1, H, W])
    return x


# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L26
class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FNN2d_Backbone(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        layers=None,
        in_dim=3,
        dropout=0,
        activation="tanh",
    ):
        super(FNN2d_Backbone, self).__init__()

        """
        The backbone network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the feature
        output shape: (batchsize, c=width, x=s, y=s)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.LayerList(
            [
                SpectralConv2dV2(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2
                )
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        self.ws = nn.LayerList(
            [
                nn.Conv1D(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.activation = _get_act(activation)

    def forward(self, x):
        """
        (b,c,h,w) -> (b,1,h,w)
        """
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.transpose([0, 2, 3, 1])
        x = self.fc0(x)  # project

        x = x.transpose([0, 3, 1, 2])

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view([batchsize, self.layers[i], -1])).view(
                [batchsize, self.layers[i + 1], size_x, size_y]
            )
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
            x = self.dropout(x)

        return x


class FNN2d(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
    ):
        super(FNN2d, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """

        self.backbone = FNN2d_Backbone(
            modes1, modes2, width, layers, in_dim, dropout, activation
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x):
        """
        (b,c,h,w) -> (b,1,h,w)
        """
        x = self.backbone(x)
        x = x.transpose([0, 2, 3, 1])
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.transpose([0, 3, 1, 2])

        if self.mean_constraint:
            x = x - paddle.mean(x, axis=(-2, -1), keepdim=True)

        return x

    def forward_icl(self, x, demo_xs, demo_ys, use_tqdm=False):
        """
        x: B, C, H, W
        demo_xs: J, C, H, W
        demo_ys: J, H, W
        """
        C_out = 1
        B, C, H, W = x.shape

        # repeat = 20; p = 0.05; sigma_range = [0, 0]
        repeat = 1
        p = 0.0
        sigma_range = [0, 0]
        x_aug = []
        demo_xs_aug = []
        for _ in range(repeat):
            if sum(sigma_range) > 0:
                import random

                sigma = random.uniform(*sigma_range)
                # https://github.com/scipy/scipy/blob/v1.11.4/scipy/ndimage/_filters.py#L232
                _kernel = min(
                    int((sigma * 4 + 1) / 2) * 2 + 1, (x.shape[1] // 2) * 2 - 1
                )
            mask = paddle.nn.functional.dropout(paddle.ones([1, C, H, W]), p=p)
            ######
            from .gaussian_blur import gaussian_blur

            if sum(sigma_range) > 0:
                _x_aug = gaussian_blur(
                    x.clone(), kernel_size=[_kernel, _kernel], sigma=sigma
                )
            else:
                _x_aug = x.clone()
            _x_aug = _x_aug * mask
            x_aug.append(_x_aug)
            ######
            _demo_xs_aug = []
            if sum(sigma_range) > 0:
                _demo_xs_aug = gaussian_blur(
                    demo_xs.clone(), kernel_size=[_kernel, _kernel], sigma=sigma
                )
            else:
                _demo_xs_aug = demo_xs.clone()
            _demo_xs_aug = _demo_xs_aug * mask
            demo_xs_aug.append(_demo_xs_aug)
        x_aug = paddle.stack(x_aug, axis=0)
        demo_xs_aug = paddle.stack(demo_xs_aug, axis=0)

        J = demo_xs.shape[0]
        pred0 = self.forward(x)  # B, H, W, T, 1
        pred = paddle.stack([self.forward(_x) for _x in x_aug], axis=-1)  # B, 1, H, W
        C = pred.shape[-1]
        demo_pred = []
        idx = 0
        for _demo_xs_aug in demo_xs_aug:
            idx = 0
            _demo_pred = []
            while idx < _demo_xs_aug.shape[0]:
                _x = _demo_xs_aug[idx : idx + B]
                _pred = self.forward(_x)
                _demo_pred.append(_pred)
                idx += _x.shape[0]
            demo_pred.append(paddle.concat(_demo_pred, axis=0))
        demo_pred = paddle.stack(demo_pred, axis=-1)

        demo_pred_flat = demo_pred.view([1, -1, C])
        y_nn = paddle.zeros([B, C_out, H, W])
        stds_nn = paddle.zeros([B, 1, H, W])
        batch_b = 1
        _b = 0
        batch_h = 64
        _h = 0
        batch_w = 64
        _w = 0

        topk = int(20 * (J**0.5))  # TODO:
        pbar = None
        while _b < B:
            _h = 0
            while _h < H:
                _w = 0
                while _w < W:
                    if pbar is not None:
                        pbar.set_description("_b %d, _h %d, _w %d" % (_b, _h, _w))
                        pbar.update(1)
                    pred_flat = pred[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ]
                    __b, _, __h, __w, _ = pred_flat.shape
                    pred_flat = pred_flat.view([-1, 1, C])

                    gap = paddle.linalg.norm(
                        (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2), axis=-1
                    )
                    gap_re = gap.view([__b, __h, __w, -1])
                    index = paddle.argsort(paddle.abs(gap_re), -1)[:, :, :, :topk]
                    _y_nn = paddle.stack(
                        [
                            paddle.take_along_axis(
                                demo_ys.view([-1, C_out]),
                                index[:, :, :, _k].view([-1, 1]),
                                axis=0,
                            ).view([__b, C_out, __h, __w])
                            for _k in range(topk)
                        ],
                        -1,
                    )

                    y_nn[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ] = _y_nn.mean(-1)
                    stds_nn[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ] = paddle.abs(_y_nn.std(-1) / _y_nn.mean(-1))

                    _w += batch_w
                _h += batch_h
            _b += batch_b

        mask = (stds_nn < stds_nn.mean()).astype(paddle.float32)  # TODO:
        return mask * y_nn + (1 - mask) * pred0


# channel-wise concatenating X_demo and Y_demo
class FNN2d_FewShot_Baseline(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
        n_demos=7,
    ):
        super(FNN2d_FewShot_Baseline, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.backbone = FNN2d_Backbone(
            modes1, modes2, width, layers, in_dim, dropout, activation
        )
        self.dropout = nn.Dropout(p=dropout)
        self.num_heads = 8
        self.fc1 = nn.Linear(
            layers[-1] * (n_demos + 1) + out_dim * n_demos * self.num_heads, fc_dim
        )
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.n_demos = n_demos

    def forward(self, demo_XY_query_x):
        """
        demo_XY_query_x: (b, J*c + J*1 + c, h, w)
        """
        demo_X, demo_Y, query_x = [], [], None
        B = len(demo_XY_query_x)
        C = self.in_dim
        J = (demo_XY_query_x.shape[1] - C) // (C + 1)
        H, W = demo_XY_query_x.shape[-2:]
        query_x = demo_XY_query_x[:, -C:]
        demo_X = demo_XY_query_x[:, : J * C]
        demo_Y = demo_XY_query_x[:, J * C : -C]
        """
        demo_X: (b, J*c, h, w)
        demo_Y: (b, J*1, h, w)
        query_x: (b, c, h, w)
        -> (b,1,h,w)
        """
        query_features = self.backbone(query_x).transpose(
            [0, 2, 3, 1]
        )  # B, C_fno, H, W
        demo_features = (
            self.backbone(demo_X.view([B, J, C, H, W]).view([B * J, C, H, W]))
            .view([B, J * self.C_fno, H, W])
            .transpose([0, 2, 3, 1])
        )

        x = paddle.stack(
            [
                paddle.concat([query_features[_b], demo_features[_b]], axis=-1)
                for _b in range(B)
            ],
            axis=0,
        )
        x = paddle.stack(
            [
                paddle.concat(
                    [
                        x[_b],
                        demo_Y[_b].repeat([self.num_heads, 1, 1]).transpose([1, 2, 0]),
                    ],
                    axis=-1,
                )
                for _b in range(B)
            ],
            axis=0,
        )  # b, h, w, (1+J)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)  # b, h, w, 1

        x = x.transpose([0, 3, 1, 2])

        if self.mean_constraint:
            x = x - paddle.mean(x, axis=(-2, -1), keepdim=True)

        return x


# https://github.com/facebookresearch/deit/blob/main/models_v2.py#L42
class TransformerBlock(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=nn.MultiHeadAttention,
        Mlp_block=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        self.attn = Attention_block(
            dim,
            num_heads,
            dropout=attn_drop,
            bias=True,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, query, key=None, value=None):
        """
        x: B, J+1, C
        """
        if key is None:
            key = query
        if value is None:
            value = query
        query = self.norm1(query)
        # x = x + self.drop_path(self.attn(x, x, x)[0])
        _query, attn_mat = self.attn(
            query, key, value, need_weights=True, average_attn_weights=True
        )  # attn_mat: B,L,L
        query = query + self.drop_path(_query)
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        return query, attn_mat


def simple_attention(query, key=None):
    if key is None:
        key = query
    attn = paddle.einsum("blc,bsc->bls", query, key)
    attn = attn.softmax(dim=-1)
    return attn


class DownSample(nn.Layer):
    def __init__(self, C_in, C_2d, C_out, shape, k=7, s=4, p=2):
        super(DownSample, self).__init__()
        # 7x7 2D conv and reduce the dimension down to 16x16x16, then flatten this and run it through a 1D conv
        self.conv2d = nn.Conv2D(C_in, C_2d, k, stride=s, padding=p)
        self.fc = nn.Linear(C_2d * (shape[0] // s) * (shape[1] // s), C_out)

    def forward(self, x):
        # B, C_fno, H, W
        # B, J, C_fno, H, W
        # query_features_down = self.query_downsample(query_features.view(B, -1)).view(B, 1, -1)
        J = 1
        if len(x.shape) == 5:
            B, J, C, H, W = x.shape
            x = x.view(-1, C, H, W)
        else:
            B, C, H, W = x.shape
        x = self.conv2d(x)
        x = x.view(B * J, -1)
        x = self.fc(x)
        return x.view(B, J, -1)


class UpSample(nn.Layer):
    def __init__(self, C_in, shape):
        super(UpSample, self).__init__()
        self.fc = nn.Linear(C_in, np.prod(shape))
        self.H, self.W = shape

    def forward(self, x):
        # x: B, C' (just for the query)
        B, C = x.shape
        x = self.fc(x).view(B, 1, self.H, self.W)
        return x


class FNN2d_FewShot_Spatial(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
        n_demos=7,
        l_attn=1,
        # input_shape=(64, 64),
        down=1,
        win_s=8,
        c_attn_hidden=1024,
        skip_backbone=False,
    ):
        super(FNN2d_FewShot_Spatial, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.down = down
        self.win_s = win_s
        self.skip_backbone = skip_backbone
        if not skip_backbone:
            self.backbone = FNN2d_Backbone(
                modes1, modes2, width, layers, in_dim, dropout, activation
            )
        else:
            self.in_dim = self.C_fno
        self.dropout = nn.Dropout(p=dropout)
        self.num_heads = 8
        self.l_attn = l_attn
        self.fc1 = nn.Linear(self.C_fno, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.n_demos = n_demos

    def forward(self, demo_XY_query_x):
        """
        demo_XY_query_x: (b, J*c + J*1 + c, h, w)
        """
        demo_X, demo_Y, query_x = [], [], None
        B = len(demo_XY_query_x)
        C = self.in_dim
        J = (demo_XY_query_x.shape[1] - C) // (C + 1)
        H, W = demo_XY_query_x.shape[-2:]
        query_x = demo_XY_query_x[:, :C]
        demo_X = demo_XY_query_x[:, C : (J + 1) * C]
        demo_Y = demo_XY_query_x[:, (J + 1) * C :]
        """
        demo_X: (b, J*c, h, w)
        demo_Y: (b, J*1, h, w)
        query_x: (b, c, h, w)
        -> (b,1,h,w)
        """
        # TODO:Ablation
        # demo_X[:, :C] = query_x
        # demo_Y[:, :1] = self.targets
        # TODO:Ablation

        if not self.skip_backbone:
            query_features = self.backbone(query_x)  # B, C_fno, H, W
            demo_features = self.backbone(
                demo_X.view([B, J, C, H, W]).view([B * J, C, H, W])
            ).view(B, J, self.C_fno, H, W)
        else:
            query_features = query_x
            demo_features = demo_X.view([B, J, self.C_fno, H, W])

        # TODO: downsample query_x, demo_X
        query_features_down = F.interpolate(
            query_features,
            size=(int(H) // self.down, int(W) // self.down),
            mode="bilinear",
            align_corners=True,
        )
        demo_features_down = paddle.stack(
            [
                F.interpolate(
                    demo_features[_b],
                    size=(int(H) // self.down, int(W) // self.down),
                    mode="bilinear",
                    align_corners=True,
                )
                for _b in range(B)
            ],
            axis=0,
        )
        # TODO: chunk Xs into windows
        # windows = window_partition(query_demo_features_down, self.win_s) # B, J+1, n_win*n_win, C, win_s, win_s
        # n_win = int(windows.shape[2] ** 0.5)
        # sequence = windows.permute(0, 1, 2, 4, 5, 3).view(B*(J+1)*n_win*n_win, self.win_s**2, self.C_fno) # N, L, C; N = B*(J+1)*n_win*n_win; L = win_s**2
        query_windows = window_partition(
            query_features_down, self.win_s
        )  # B, n_win*n_win, C, win_s, win_s
        demo_windows = window_partition(
            demo_features_down, self.win_s
        )  # B, J, n_win*n_win, C, win_s, win_s
        n_win = int(query_windows.shape[1] ** 0.5)
        query_windows = query_windows.transpose([0, 1, 3, 4, 2]).view(
            [B * n_win**2, self.win_s**2, self.C_fno]
        )  # B*n_win*n_win, win_s*win_s, C
        demo_windows = demo_windows.transpose([0, 2, 1, 4, 5, 3]).view(
            [B * n_win**2, J * self.win_s**2, self.C_fno]
        )  # B*n_win*n_win, J*win_s*win_s, C

        self._attn_mats = []
        # # TODO: add position embedding
        # for _l in range(self.l_attn):
        #     # sequence, attn_mat = self.attns[_l](sequence)
        #     # cross-attention
        #     query_windows, attn_mat = self.attns[_l](query_windows, demo_windows)
        #     self._attn_mats.append(attn_mat.detach().cpu().numpy()) # N, L, S; N = B*n_win*n_win; L = win_s**2; S = J*win_s**2

        # TODO: simple attention
        attn_mat = simple_attention(query_windows, demo_windows)
        self._attn_mats.append(
            attn_mat.detach().cpu().numpy()
        )  # N, L, S; N = B*n_win*n_win; L = win_s**2; S = J*win_s**2

        demo_Y_down = F.interpolate(
            demo_Y,
            size=(int(H) // self.down, int(W) // self.down),
            mode="bilinear",
            align_corners=True,
        )
        # B, J, n_win*n_win, 1, win_s, win_s => B, J, n_win*n_win, win_s, win_s
        windows_Y = (
            window_partition(demo_Y_down.unsqueeze(2), self.win_s)
            .squeeze(3)
            .transpose([0, 2, 1, 3, 4])
            .view([B * n_win**2, J * self.win_s**2])
        )
        demo_Y_reweighted = paddle.einsum("bls,bs->bl", attn_mat, windows_Y).view(
            [B, n_win**2, 1, self.win_s, self.win_s]
        )
        demo_Y_reweighted = window_reverse(
            demo_Y_reweighted, self.win_s, int(H) // self.down, int(W) // self.down
        )  # B 1, H_down, W_down
        demo_Y_reweighted = F.interpolate(
            demo_Y_reweighted, size=(H, W), mode="bilinear", align_corners=True
        )  # B, 1, H, W
        self.query_score = demo_Y_reweighted[:, 0].detach().cpu().numpy()

        # query_demo_features = self.upsample(sequence[:, 0]).view(B, 1, H, W).transpose([0, 2, 3, 1]) # B, 1, H, W => B, H, W, 1
        # self.query_score = query_demo_features[:, :, :, 0].detach().cpu().numpy()

        # query_features = query_features.transpose(0, 2, 3, 1) # B, C_fno, H, W => B, H, W, C_fno
        y = self.fc1(
            query_features.transpose([0, 2, 3, 1])
        )  # B, C_fno, H, W => B, H, W, C_fno
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)  # b, h, w, 1

        y = y.transpose([0, 3, 1, 2])
        if self.mean_constraint:
            y = y - paddle.mean(y, axis=(-2, -1), keepdim=True)

        y = (y + demo_Y_reweighted) / 2  # TODO:

        return y


class FNN2d_FewShot_Spatial_v2(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
        n_demos=7,
        l_attn=1,
        # input_shape=(64, 64),
        down=1,
        win_s=8,
        c_attn_hidden=1024,
        skip_backbone=False,
    ):
        super(FNN2d_FewShot_Spatial_v2, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.down = down
        self.win_s = win_s
        self.skip_backbone = skip_backbone
        if not skip_backbone:
            self.backbone = FNN2d_Backbone(
                modes1, modes2, width, layers, in_dim, dropout, activation
            )
        else:
            self.in_dim = self.C_fno
        self.dropout = nn.Dropout(p=dropout)
        self.num_heads = 8
        self.l_attn = l_attn
        self.fc1 = nn.Linear(self.C_fno, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.n_demos = n_demos

    def forward(self, demo_XY_query_x):
        """
        demo_XY_query_x: (b, J*c + J*1 + c, h, w)
        """
        demo_X, demo_Y, query_x = [], [], None
        B = len(demo_XY_query_x)
        C = self.in_dim
        J = (demo_XY_query_x.shape[1] - C) // (C + 1)
        H, W = demo_XY_query_x.shape[-2:]
        query_x = demo_XY_query_x[:, :C]
        demo_X = demo_XY_query_x[:, C : (J + 1) * C]
        demo_Y = demo_XY_query_x[:, (J + 1) * C :]
        """
        demo_X: (b, J*c, h, w)
        demo_Y: (b, J*1, h, w)
        query_x: (b, c, h, w)
        -> (b,1,h,w)
        """
        # TODO:Ablation
        # demo_X[:, :C] = query_x
        # demo_Y[:, :1] = self.targets
        # TODO:Ablation

        if not self.skip_backbone:
            query_features = self.backbone(query_x)  # B, C_fno, H, W
            demo_features = self.backbone(
                demo_X.view([B, J, C, H, W]).view([B * J, C, H, W])
            ).view(B, J, self.C_fno, H, W)
        else:
            query_features = query_x
            demo_features = demo_X.view([B, J, self.C_fno, H, W])

        self.query_score = None
        self._attn_mats = [None]

        y = self.fc1(
            query_features.transpose([0, 2, 3, 1])
        )  # B, C_fno, H, W => B, H, W, C_fno
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)  # b, h, w, 1
        y = y.transpose([0, 3, 1, 2])  # b, 1, h, w
        if self.mean_constraint:
            y = y - paddle.mean(y, axis=(-2, -1), keepdim=True)

        y_demo = self.fc1(
            demo_features.transpose([0, 1, 3, 4, 2])
        )  # B, J, C_fno, H, W => B, J, H, W, C_fno
        y_demo = self.activation(y_demo)
        y_demo = self.dropout(y_demo)
        y_demo = self.fc2(y_demo)
        y_demo = self.dropout(y_demo)  # b, j, h, w, 1
        y_demo = y_demo.transpose([0, 1, 4, 2, 3])  # b, j, 1, h, w
        if self.mean_constraint:
            y_demo = y_demo - paddle.mean(y_demo, axis=(-2, -1), keepdim=True)

        B, C, H, W = y.shape

        # # #########################################################
        y_flat = y.view([-1, 1])
        y_demo_flat = y_demo.view([1, -1])
        gap = y_flat - y_demo_flat
        gap_re = gap.view([B, C, H, W, -1])

        index = paddle.argsort(paddle.abs(gap_re), -1)

        topk = 100
        y_nn = 0
        for _k in range(topk):
            y_nn += paddle.take(demo_Y.view([-1, 1]), index[:, :, :, :, _k])
        y_nn /= topk
        y = (y + y_nn) / 2  # TODO:
        return y_nn


def build_fno(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut] * len(params.modes1)
        params.modes2 = [params.mode_cut] * len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut] * len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    if params.n_demos == 0:
        return FNN2d(
            params.modes1,
            params.modes2,
            layers=params.layers,
            fc_dim=params.fc_dim,
            in_dim=input_dim,
            out_dim=params.out_dim,
            dropout=params.dropout,
            activation="gelu",
            mean_constraint=(params.loss_func == "pde"),
        )
    else:
        if hasattr(params, "baseline") and params.baseline:
            return FNN2d_FewShot_Baseline(
                params.modes1,
                params.modes2,
                layers=params.layers,
                fc_dim=params.fc_dim,
                in_dim=input_dim,
                out_dim=params.out_dim,
                dropout=params.dropout,
                activation="gelu",
                mean_constraint=(params.loss_func == "pde"),
                n_demos=params.n_demos,
            )
        elif hasattr(params, "spatial") and params.spatial:
            return FNN2d_FewShot_Spatial_v2(
                params.modes1,
                params.modes2,
                layers=params.layers,
                fc_dim=params.fc_dim,
                in_dim=input_dim,
                out_dim=params.out_dim,
                dropout=params.dropout,
                activation="gelu",
                mean_constraint=(params.loss_func == "pde"),
                n_demos=params.n_demos,
                l_attn=params.l_attn,
                c_attn_hidden=params.c_attn_hidden,
                down=params.down,
                win_s=params.win_s,
                skip_backbone=(
                    params.train_path.endswith("npy")
                    and ("feature_data" in params.train_path)
                ),
            )
        # else:
        #     return FNN2d_FewShot(params.modes1, params.modes2, layers=params.layers, fc_dim=params.fc_dim,
        #                 in_dim=input_dim, out_dim=params.out_dim, dropout=params.dropout,
        #                 activation='gelu', mean_constraint=(params.loss_func == 'pde'), n_demos=params.n_demos, l_attn=params.l_attn,
        #                 input_shape=(params.nx, params.ny), k_conv2d=params.k_conv2d, s_conv2d=params.s_conv2d, c_conv2d=params.c_conv2d, c_attn_hidden=params.c_attn_hidden,
        #                 skip_backbone=(params.train_path.endswith("npy") and ("feature_data" in params.train_path))
        #                 )


class FNN2d_MAE(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
    ):
        super(FNN2d_MAE, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        Backbone:
          1) Lift the input to the desire channel dimension by self.fc0 .
          2) 4 layers of the integral operators u' = (W + K)(u).
              W defined by self.w; K defined by self.conv .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.encoder = FNN2d_Backbone(
            modes1, modes2, width, layers, in_dim, dropout, activation
        )
        self.decoder = FNN2d_Backbone(
            modes1,
            modes2,
            width,
            layers[:-1] + [in_dim],
            self.C_fno,
            dropout,
            activation,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_to_decoder = nn.Linear(self.C_fno, self.C_fno)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x, mask=None):
        """
        x: (b, c, h, w)
        """
        # import
        # B, C, H, W = x.shape
        if mask is None:
            x_enc = self.encoder(x)
        else:
            x_enc = self.encoder(x * mask)
        x_enc = self.encoder_to_decoder(x_enc.transpose([0, 2, 3, 1])).transpose(
            [0, 3, 1, 2]
        )
        x_dec = self.decoder(x_enc)
        return x_dec


def fno_pretrain(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut] * len(params.modes1)
        params.modes2 = [params.mode_cut] * len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut] * len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    return FNN2d_MAE(
        params.modes1,
        params.modes2,
        layers=params.layers,
        fc_dim=params.fc_dim,
        in_dim=input_dim,
        out_dim=params.out_dim,
        dropout=params.dropout,
        activation="gelu",
        mean_constraint=(params.loss_func == "pde"),
    )


def _cast_squeeze_in(img: Tensor, req_dtypes: List[paddle.dtype]):
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(axis=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _get_gaussian_kernel1d(
    kernel_size: int, sigma: float, dtype: paddle.dtype
) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = paddle.linspace(-ksize_half, ksize_half, num=kernel_size, dtype=dtype)
    pdf = paddle.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: paddle.dtype
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype)
    kernel2d = paddle.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: paddle.dtype
) -> Tensor:
    if need_squeeze:
        img = img.squeeze(axis=0)

    if need_cast:
        if out_dtype in (
            paddle.uint8,
            paddle.int8,
            paddle.int16,
            paddle.int32,
            paddle.int64,
        ):
            # it is better to round before cast
            img = paddle.round(img)
        img = img.to(out_dtype)

    return img


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(
            f"sigma should be either float or sequence of floats. Got {type(sigma)}"
        )
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(
            f"If sigma is a sequence, its length should be 2. Got {len(sigma)}"
        )
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")
    # print(f"img: {img}")
    # if not (isinstance(img, Tensor)):
    #     raise TypeError(f"img should be Tensor. Got {type(img)}")

    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype)
    kernel = kernel.expand([img.shape[-3], 1, kernel.shape[0], kernel.shape[1]])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    img = paddle.nn.functional.pad(img, padding, mode="reflect")
    img = paddle.nn.functional.conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


# https://github.com/erichson/SuperBench/blob/3719ef9010dc081c3f8e9644813764ef56420fc9/eval.py#L123
class Conv2dDerivative(nn.Layer):
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


class LossGenerator(nn.Layer):
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


def trunc_normal_(tensor, mean=0.0, std=1.0):  # noqa
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    # 'pretrain_videomae_small_patch16_224',
    # "pretrain_videomae_base_patch16_224",
    # 'pretrain_videomae_large_patch16_224',
    # 'pretrain_videomae_huge_patch16_224',
]


def _cfg(url="", **kwargs):
    return {
        # 'url': url,
        # 'num_classes': 400,
        "input_size": (3, 512, 512),  # TODO:
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


def build_vmae(params):
    """Builds model from parameter file.

    General recipe is to build the spatial and temporal modules separately and then
    combine them in a model. Eventually the "stem" and "destem" should
    also be parameterized.
    """
    # space_time_block = build_spacetime_block(params)
    # processor_blocks=params.processor_blocks,
    # n_states=params.n_states,
    # override_block=space_time_block,)
    model = PretrainVisionTransformer(
        img_size=params.input_size,
        patch_size=params.patch_size,
        encoder_embed_dim=params.encoder_embed_dim,
        encoder_depth=12,
        decoder_depth=params.decoder_depth,
        encoder_num_heads=params.encoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        encoder_num_classes=0,
        decoder_num_classes=params.decoder_num_classes,
        tubelet_size=params.tubelet_size,
        decoder_embed_dim=params.decoder_embed_dim,
        decoder_num_heads=params.decoder_num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_frames=params.n_steps,
        num_demos=params.num_demos if hasattr(params, "num_demos") else 0,
        # drop_path_rate=params.drop_path_rate, # TODO:
        # n_states=params.n_states, # TODO:
    )
    model.default_cfg = _cfg()
    if params.vmae_pretrained:
        checkpoint = paddle.load(params.vmae_pretrained)
        if "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
        elif "model_state" in checkpoint.keys():
            # model.load_state_dict(checkpoint["model_state"])
            # state = {key[7:] if 'module' in key else key: value for key, value in checkpoint["model_state"].items()}
            # model.load_state_dict(state)

            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:] if "module" in key else key
                new_state_dict[name] = val
            state = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in state and state[k].size() == new_state_dict[k].size()
            }
            # 2. overwrite entries in the existing state dict
            state.update(pretrained_dict)
            # 3. load the new state dict
            # message = model.load_state_dict(state)
            # self.model.load_state_dict(new_state_dict)
            unload_keys = [k for k in new_state_dict.keys() if k not in pretrained_dict]
            if len(unload_keys) > 0:
                import warnings

                warnings.warn(
                    "Warning: missing keys during restoring checkpoint: %s"
                    % (str(unload_keys))
                )

    return model


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.zeros([all_head_dim])
            )
            self.v_bias = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.zeros([all_head_dim])
            )
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat(
                (self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias)
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).permute([2, 0, 3, 1, 4])
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            gamma_1 = init_values * paddle.ones((dim))
            gamma_1.stop_gradient = False
            self.gamma_1 = paddle.base.framework.EagerParamBase.from_tensor(gamma_1)
            gamma_2 = init_values * paddle.ones((dim))
            gamma_2.stop_gradient = False
            self.gamma_2 = paddle.base.framework.EagerParamBase.from_tensor(gamma_2)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCTHW -> BC'T'H'W' -> BC'(T'H'W')
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with paddle instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return paddle.to_tensor(
        sinusoid_table, dtype=paddle.float32, stop_gradient=True
    ).unsqueeze(0)


class PretrainVisionTransformerEncoder(nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
        num_frames=16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.zeros([1, num_patches + 1, embed_dim])
            )
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_xaiverUniform = paddle.nn.initializer.XavierUniform()
            init_xaiverUniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant(value=0)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, mask=None):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if mask is not None:
            x_vis = x[~mask].reshape([B, -1, C])  # ~mask means visible
        else:
            x_vis = x.reshape([B, -1, C])

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = paddle.distributed.fleet.utils.recompute(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size**2
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_xaiverUniform = paddle.nn.initializer.XavierUniform()
            init_xaiverUniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant(value=0)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num=0):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = paddle.distributed.fleet.utils.recompute(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,  #  decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        num_frames=16,
        num_demos=0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
            num_frames=num_frames,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        self.mask_token = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.zeros([1, 1, decoder_embed_dim])
        )

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        trunc_normal_(self.mask_token, std=0.02)

        self.num_demos = num_demos
        if self.num_demos > 0:
            self.lossgen = LossGenerator(dx=1 / 256, kernel_size=3)  # TODO:
            # self.lossgen = LossGenerator(dx=2.0*math.pi/2048.0, kernel_size=3) # TODO:

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_xaiverUniform = paddle.nn.initializer.XavierUniform()
            init_xaiverUniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant(value=0)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x, mask=None):
        """
        x: T, B, C, H, W
        """
        T_in, B, C_in, H, W = x.shape
        x = x.permute(1, 2, 0, 3, 4)
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        if mask is not None:
            pos_emd_vis = expand_pos_embed[~mask].reshape([B, -1, C])
            if mask.sum() == 0:
                # mask_ratio = 0: all tokens are visible
                x_full = x_vis + pos_emd_vis  # [B, N, C_d]
                x = self.decoder(x_full)  # [B, :, 3 * 16 * 16]
            else:
                pos_emd_mask = expand_pos_embed[mask].reshape([B, -1, C])
                x_full = paddle.concat(
                    [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], axis=1
                )  # [B, N, C_d]
                x = self.decoder(
                    x_full, pos_emd_mask.shape[1]
                )  # [B, N_mask, 3 * 16 * 16]
            return x
        else:
            x = self.decoder(x_vis)
            x = rearrange(
                x,
                "b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)",
                t=T_in // self.tubelet_size,
                h=H // self.patch_size,
                w=W // self.patch_size,
                p0=self.tubelet_size,
                p1=self.patch_size,
                p2=self.patch_size,
                c=C_in,
            )
            x = x.permute(2, 0, 1, 3, 4)
            return x[-1]

    def forward_icl(self, x, demo_xs, demo_ys):
        """
        x: T, B, C, H, W
        demo_xs: T, J, C, H, W
        demo_ys: J, H, W
        """
        _, B, _, H, W = x.shape
        # J = demo_xs.shape[1]
        pred = self.forward(x)  # B, 3, H, W
        C = pred.shape[1]
        # div = self.lossgen.get_div_loss(pred)
        demo_pred = []
        idx = 0
        demo_div = []
        while idx < demo_xs.shape[1]:
            _x = demo_xs[:, idx : idx + B]
            _pred = self.forward(_x)
            demo_pred.append(_pred)
            idx += _x.shape[1]
            _div = self.lossgen.get_div_loss(_pred)
            demo_div.append(_div)
        demo_pred = paddle.concat(demo_pred, axis=0)
        demo_div = paddle.concat(demo_div, axis=0)

        demo_pred_flat = demo_pred.permute(0, 2, 3, 1).view([1, -1, C])
        # demo_div_flat = demo_div.view([1, -1])
        y_nn = paddle.zeros([B, C, H, W])
        gap_nn = paddle.zeros([B, H, W])
        batch_b = 1
        _b = 0
        batch_h = 16
        _h = 0
        batch_w = 16
        _w = 0
        # topk1 = round(0.2 * H * W * J) # TODO:
        # topk1 = 20  # TODO:
        # topk = round(0.02 * H * W * J) # TODO:
        topk = 10  # TODO:
        pbar = tqdm(
            total=np.ceil(B / batch_b) * np.ceil(H / batch_h) * np.ceil(W / batch_w)
        )
        while _b < B:
            _h = 0
            while _h < H:
                _w = 0
                while _w < W:
                    pbar.set_description("_b %d, _h %d, _w %d" % (_b, _h, _w))
                    pbar.update(1)
                    pred_flat = pred[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ]
                    __b, _, __h, __w = pred_flat.shape
                    pred_flat = pred_flat.permute(0, 2, 3, 1).view([-1, 1, C])
                    gap = paddle.linalg.norm(
                        (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2), axis=-1
                    )
                    gap_re = gap.view([__b, __h, __w, -1])
                    gap_nn[
                        _b : _b + batch_b, _h : _h + batch_h, _w : _w + batch_w
                    ] = paddle.mean(paddle.sort(gap_re, -1)[0][:, :, :, :topk], -1)
                    index = paddle.argsort(paddle.abs(gap_re), -1)[
                        :, :, :, :topk
                    ]  # TODO: spatial index of ascending sort by pred gap
                    _y_nn = 0
                    for _k in range(topk):
                        _y_nn += (
                            paddle.take_along_dim(
                                demo_ys.permute(0, 2, 3, 1).view([-1, C]),
                                index[:, :, :, _k].view([-1, 1]),
                                dim=0,
                            )
                            .view([__b, __h, __w, C])
                            .permute(0, 3, 1, 2)
                        )
                    _y_nn /= topk
                    # spatial_dims = (2, 3)
                    # print("pred:", (((self.target[:, :, _h:_h+batch_h, _w:_w+batch_w] - pred[:, :, _h:_h+batch_h, _w:_w+batch_w]).pow(2).mean(spatial_dims, keepdim=True) / (1e-7 + self.target[:, :, _h:_h+batch_h, _w:_w+batch_w].pow(2).mean(spatial_dims, keepdim=True))).sqrt()).mean().item(), "    ICL:", (((self.target[:, :, _h:_h+batch_h, _w:_w+batch_w] - _y_nn).pow(2).mean(spatial_dims, keepdim=True) / (1e-7 + self.target[:, :, _h:_h+batch_h, _w:_w+batch_w].pow(2).mean(spatial_dims, keepdim=True))).sqrt()).mean().item())
                    y_nn[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ] = _y_nn
                    # np.set_printoptions(precision=5)
                    # print(_h, _w, sum(pred[_b, :2, _h+1, _w+1]-y_nn[_b, :2, _h+1, _w+1]).item(), np.round(pred[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist(), np.round(y_nn[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist())

                    _w += batch_w
                _h += batch_h
            _b += batch_b
        # bp()
        print(y_nn.mean(), demo_ys.mean())
        # return y_nn
        # return (y_nn + pred) / 2
        mask = (paddle.clip(gap_nn, 0, 1) ** 0.5 > 0.1).astype(paddle.float32)  # TODO:
        return (1 - mask) * y_nn + mask * pred


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_format, level=log_level)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"
):

    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    import paddle

    logging.info("--------------- Versions ---------------")
    logging.info("Paddle: " + str(paddle.__version__))
    logging.info("----------------------------------------")


"""
  loss functions
# """


class LossMSE:
    """mse loss"""

    def __init__(self, params, model):
        self.params = params
        self.model = model

    def data(self, inputs, pred, target):
        if self.params.loss_style == "mean":
            loss = paddle.mean((target - pred) ** 2)
        elif self.params.loss_style == "sum":
            loss = paddle.sum((target - pred) ** 2) / pred.shape[0]
        return loss

    def bc(self, inputs, pred, targets):
        # currently no BC
        return paddle.to_tensor(0.0).astype(dtype=paddle.float32)

    def pde(self, inputs, pred, targets):
        # currently no PDE loss
        return paddle.to_tensor(0.0).astype(dtype=paddle.float32)


def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    # ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    # ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    # uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    # uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = -(auxx + auyy)
    return Du


def darcy_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    Du = FDM_Darcy(u, a)
    f = paddle.ones(Du.shape)
    loss_f = lploss.rel(Du, f)

    return loss_f


def FDM_NS_vorticity(w, v=1 / 40, t_interval=1.0):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    w = w.reshape([batchsize, nx, ny, nt])

    w_h = paddle.fft.fft2(w, axes=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = (
        paddle.concat(
            (
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            0,
        )
        .reshape([N, 1])
        .repeat([1, N])
        .reshape([1, N, N, 1])
    )
    k_y = (
        paddle.concat(
            (
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            0,
        )
        .reshape([1, N])
        .repeat([N, 1])
        .reshape([1, N, N, 1])
    )
    # Negative Laplacian in Fourier space
    lap = k_x**2 + k_y**2
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = paddle.fft.irfft2(ux_h[:, :, : k_max + 1], axes=[1, 2])
    uy = paddle.fft.irfft2(uy_h[:, :, : k_max + 1], axes=[1, 2])
    wx = paddle.fft.irfft2(wx_h[:, :, : k_max + 1], axes=[1, 2])
    wy = paddle.fft.irfft2(wy_h[:, :, : k_max + 1], axes=[1, 2])
    wlap = paddle.fft.irfft2(wlap_h[:, :, : k_max + 1], axes=[1, 2])

    dt = t_interval / (nt - 1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux * wx + uy * wy - v * wlap)[..., 1:-1]  # - forcing
    return Du1


def Autograd_Burgers(u, grid, v=1 / 100):
    from paddle import grad

    gridt, gridx = grid

    ut = grad(u.sum(), gridt, create_graph=True)[0]
    ux = grad(u.sum(), gridx, create_graph=True)[0]
    uxx = grad(ux.sum(), gridx, create_graph=True)[0]
    Du = ut + ux * u - v * uxx
    return Du, ux, uxx, ut


def AD_loss(u, u0, grid, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = Autograd_Burgers(u, grid)

    if index_ic is None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = paddle.zeros(
            nx,
        ).astype(paddle.int64)
        index_x = paddle.to_tensor(range(nx)).astype(paddle.int64)
        boundary_u = u[:, index_t, index_x]

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = (
            paddle.to_tensor(range(batchsize)).reshape([batchsize, 1]).repeat([1, p])
        )
        u0 = u0[batch_index, index_ic]

        # loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        # loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = paddle.zeros(Du.shape)
    loss_f = F.mse_loss(Du, f)
    return loss_ic, loss_f


class LpLoss(object):
    """
    loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * paddle.linalg.norm(
            x.view([num_examples, -1]) - y.view([num_examples, -1]), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = paddle.linalg.norm(
            x.reshape([num_examples, -1]) - y.reshape([num_examples, -1]), self.p, 1
        )
        y_norms = paddle.linalg.norm(y.reshape([num_examples, -1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms / y_norms)
            else:
                return paddle.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def FDM_Burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt - 1)
    # dx = D / (nx)

    u_h = paddle.fft.fft(u, axis=2)
    # Wavenumbers in y-direction
    k_max = nx // 2
    k_x = paddle.concat(
        (
            paddle.arange(start=0, end=k_max, step=1),
            paddle.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).reshape([1, 1, nx])
    ux_h = 2j * np.pi * k_x * u_h
    uxx_h = 2j * np.pi * k_x * ux_h
    ux = paddle.fft.irfft(ux_h[:, :, : k_max + 1], axis=2, n=nx)
    uxx = paddle.fft.irfft(uxx_h[:, :, : k_max + 1], axis=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux * u - v * uxx)[:, 1:-1, :]
    return Du


def PINO_loss(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = paddle.zeros(
        nx,
    ).astype(paddle.int64)
    index_x = paddle.tensor(range(nx)).astype(paddle.int64)
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers(u, v)[:, :, :]
    f = paddle.zeros(
        Du.shape,
    )
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_u, loss_f


def PINO_loss3d(u, u0, forcing, v=1 / 40, t_interval=1.0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.repeat(batchsize, 1, 1, nt - 2)
    loss_f = lploss(Du, f)

    return loss_ic, loss_f


def PDELoss(model, x, t, nu):
    """
    Compute the residual of PDE:
        residual = u_t + u * u_x - nu * u_{xx} : (N,1)

    Params:
        - model
        - x, t: (x, t) pairs, (N, 2) tensor
        - nu: constant of PDE
    Return:
        - mean of residual : scalar
    """
    u = model(paddle.concat([x, t], axis=1))
    # First backward to compute u_x (shape: N x 1), u_t (shape: N x 1)
    grad_x, grad_t = paddle.grad(outputs=[u.sum()], inputs=[x, t], create_graph=True)
    # Second backward to compute u_{xx} (shape N x 1)

    (gradgrad_x,) = paddle.grad(outputs=[grad_x.sum()], inputs=[x], create_graph=True)

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual


def get_forcing(S):
    x2 = (
        paddle.to_tensor(
            np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=paddle.float32
        )
        .reshape([1, S])
        .repeat([S, 1])
    )
    return -4 * (paddle.cos(4 * (x2))).reshape([1, S, S, 1])


def vor2vel(w, L=2 * np.pi):
    """
    Convert vorticity into velocity
    Args:
        w: vorticity with shape (batchsize, num_x, num_y, num_t)

    Returns:
        ux, uy with the same shape
    """
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = paddle.fft.fft2(w, axes=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = (
        paddle.concat(
            (
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            0,
        )
        .reshape([N, 1])
        .repeat([1, N])
        .reshape([1, N, N, 1])
    )
    k_y = (
        paddle.concat(
            (
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            0,
        )
        .reshape([1, N])
        .repeat([N, 1])
        .reshape([1, N, N, 1])
    )
    # Negative Laplacian in Fourier space
    lap = k_x**2 + k_y**2
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 2 * np.pi / L * 1j * k_y * f_h
    uy_h = -2 * np.pi / L * 1j * k_x * f_h

    ux = paddle.fft.irfft2(ux_h[:, :, : k_max + 1], axes=[1, 2])
    uy = paddle.fft.irfft2(uy_h[:, :, : k_max + 1], axes=[1, 2])
    return ux, uy


def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = paddle.randint(s, shape=(N, p))
    sample_ic_t = paddle.zeros([N, p])
    sample_ic_x = index_ic / s

    # sample BC
    sample_bc = paddle.rand(shape=(N, p // 2))
    sample_bc_t = paddle.concat([sample_bc, sample_bc], axis=1)
    sample_bc_x = paddle.concat(
        [paddle.zeros([N, p // 2]), paddle.ones([N, p // 2])], axis=1
    )

    # sample I
    sample_i_t = -paddle.cos(paddle.rand(shape=(N, q)) * np.pi / 2) + 1
    sample_i_x = paddle.rand(shape=(N, q))

    sample_t = paddle.concat([sample_ic_t, sample_bc_t, sample_i_t], axis=1)
    sample_t.stop_gradient = False
    sample_x = paddle.concat([sample_ic_x, sample_bc_x, sample_i_x], a=1)
    sample_x.stop_gradient = False
    sample = paddle.stack([sample_t, sample_x], axis=-1).reshape([N, (p + p + q), 2])
    return sample, sample_t, sample_x, index_ic.astype(paddle.int64)


def get_grid(N, T, s):
    gridt = (
        paddle.to_ensor(np.linspace(0, 1, T), dtype=paddle.float32)
        .reshape([1, T, 1])
        .repeat([N, 1, s])
    )
    gridt.stop_gradient = False
    gridx = (
        paddle.to_tensor(np.linspace(0, 1, s + 1)[:-1], dtype=paddle.float32)
        .reshape([1, 1, s])
        .repeat([N, T, 1])
    )
    gridx.stop_gradient = False
    grid = paddle.stack([gridt, gridx], axis=-1).reshape([N, T * s, 2])
    return grid, gridt, gridx


def get_2dgrid(S):
    """
    get array of points on 2d grid in (0,1)^2
    Args:
        S: resolution

    Returns:
        points: flattened grid, ndarray (N, 2)
    """
    xarr = np.linspace(0, 1, S)
    yarr = np.linspace(0, 1, S)
    xx, yy = np.meshgrid(xarr, yarr, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=0).T
    return points


def paddle2dgrid(num_x, num_y, bot=(0, 0), top=(1, 1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = paddle.linspace(x_bot, x_top, num=num_x)
    y_arr = paddle.linspace(y_bot, y_top, num=num_y)
    xx, yy = paddle.meshgrid(x_arr, y_arr)
    mesh = paddle.stack([xx, yy], axis=2)
    return mesh


def get_grid3d(S, T, time_scale=1.0, device="cpu"):
    gridx = paddle.to_tensor(np.linspace(0, 1, S + 1)[:-1], dtype=paddle.float32)
    gridx = gridx.reshape([1, S, 1, 1, 1]).repeat([1, 1, S, T, 1])
    gridy = paddle.to_tensor(np.linspace(0, 1, S + 1)[:-1], dtype=paddle.float32)
    gridy = gridy.reshape([1, 1, S, 1, 1]).repeat([1, S, 1, T, 1])
    gridt = paddle.to_tensor(np.linspace(0, 1 * time_scale, T), dtype=paddle.float32)
    gridt = gridt.reshape([1, 1, 1, T, 1]).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


def convert_ic(u0, N, S, T, time_scale=1.0):
    u0 = u0.reshape(N, S, S, 1, 1).repeat([1, 1, 1, T, 1])
    gridx, gridy, gridt = get_grid3d(S, T, time_scale=time_scale, device=u0.device)
    a_data = paddle.concat(
        (
            gridx.repeat([N, 1, 1, 1, 1]),
            gridy.repeat([N, 1, 1, 1, 1]),
            gridt.repeat([N, 1, 1, 1, 1]),
            u0,
        ),
        axis=-1,
    )
    return a_data


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag


def zero_grad(params):
    """
    set grad field to 0
    """
    if isinstance(params, paddle.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count


def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = "checkpoints/%s/" % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    paddle.save({"model": model_state_dict, "optim": optim_dict}, ckpt_dir + name)
    print("Checkpoint is saved at %s" % ckpt_dir + name)


def save_ckpt(path, model, optimizer=None, scheduler=None):
    model_state = model.state_dict()
    if optimizer:
        optim_state = optimizer.state_dict()
    else:
        optim_state = None

    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    paddle.save(
        {"model": model_state, "optim": optim_state, "scheduler": scheduler_state}, path
    )
    print(f"Checkpoint is saved to {path}")


def dict2str(log_dict):
    res = ""
    for key, value in log_dict.items():
        res += f"{key}: {value}|"
    return res


class YParams:
    """Yaml file parser"""

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:

            for key, val in YAML().load(_file)[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None

                self.params[key] = val
                self.__setattr__(key, val)

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")
