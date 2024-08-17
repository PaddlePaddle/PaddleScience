import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.initializer as Initializer

def compl_mul1d(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    res = paddle.einsum("bix,iox->box", a, b)
    return res

def compl_mul2d(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res =  paddle.einsum("bixy,ioxy->boxy", a, b)
    return res

def compl_mul3d(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    res = paddle.einsum("bixyz,ioxyz->boxyz", a, b)
    return res

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

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.complex64, attr=Initializer.Assign(self.scale * paddle.rand(in_channels, out_channels, self.modes1)))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(batchsize, self.in_channels, x.shape[-1]//2 + 1, dtype=paddle.complex64)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = paddle.fft.irfftn(out_ft, s=[x.shape[-1]], axes=[2])
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

        self.scale = (1 / (in_channels * out_channels))
        weights1_attr = Initializer.Assign(self.scale * paddle.rand([in_channels, out_channels, self.modes1, self.modes2], dtype=paddle.float32))
        self.weights1_real = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.float32, attr=weights1_attr)
        self.weights1_imag = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.float32, attr=weights1_attr)
        self.weights1 = paddle.concat([self.weights1_real, self.weights1_imag], axis=0)
        self.weights2 = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.float32, attr=Initializer.Assign(self.scale * paddle.rand([in_channels, out_channels, self.modes1, self.modes2], dtype=paddle.float32)))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=[2, 3])

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros([batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1],
                                dtype=paddle.float32)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = paddle.fft.irfftn(out_ft, s=(x.shape[-2], x.shape[-1]), axes=[2, 3])
        return x

class SpectralConv3d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.complex64, attr=Initializer.Assign(self.scale * paddle.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3)))
        self.weights2 = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.complex64, attr=Initializer.Assign(self.scale * paddle.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3)))
        self.weights3 = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.complex64, attr=Initializer.Assign(self.scale * paddle.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3)))
        self.weights4 = paddle.create_parameter(shape=(in_channels, out_channels), dtype=paddle.complex64, attr=Initializer.Assign(self.scale * paddle.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3)))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, dim=[2,3,4])
        
        z_dim = min(x_ft.shape[4], self.modes3)
        
        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], self.modes3, dtype=paddle.complex64)
        
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding 
        coeff = paddle.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, dtype=paddle.complex64)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :z_dim]
        out_ft[:, :, :self.modes1, :self.modes2, :] = compl_mul3d(coeff, self.weights1)
        
        coeff = paddle.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, dtype=paddle.complex64)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :z_dim]
        out_ft[:, :, -self.modes1:, :self.modes2, :] = compl_mul3d(coeff, self.weights2)
        
        coeff = paddle.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, dtype=paddle.complex64)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, :z_dim]
        out_ft[:, :, :self.modes1, -self.modes2:, :] = compl_mul3d(coeff, self.weights3)
        
        coeff = paddle.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, dtype=paddle.complex64)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, :z_dim]
        out_ft[:, :, -self.modes1:, -self.modes2:, :] = compl_mul3d(coeff, self.weights4)

        #Return to physical space
        x = paddle.fft.irfftn(out_ft, s=(x.shape[2], x.shape[3], x.shape[4]), axes=[2,3,4])
        return x

class FourierBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, act='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv1D(in_channels, out_channels, 1)
        if act == 'tanh':
            self.act = paddle.tanh_
        elif act == 'gelu':
            self.act = nn.GELU
        elif act == 'none':
            self.act = None
        else:
            raise ValueError(f'{act} is not supported')

    def forward(self, x):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x.reshape(x.shape[0], self.in_channel, -1))
        out = x1 + x2.reshape(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        if self.act is not None:
            out = self.act(out)
        return out

