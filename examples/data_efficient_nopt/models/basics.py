""" From PINO: https://github.com/devzhk/PINO/blob/master/models/basics.py """
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


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
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        # dtype = x.dtype

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft2(x.float(), axes=(-2, -1), norm="ortho")
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
