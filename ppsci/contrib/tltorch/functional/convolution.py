import torch
import torch.nn.functional as F

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg

from ..factorized_tensors import CPTensor, TTTensor, TuckerTensor, DenseTensor

# Author: Jean Kossaifi
# License: BSD 3 clause


_CONVOLUTION = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}

def convolve(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution of any specified order, wrapper on torch's F.convNd

    Parameters
    ----------
    x : torch.Tensor or FactorizedTensor
        input tensor
    weight : torch.Tensor
        convolutional weights
    bias : bool, optional
        by default None
    stride : int, optional
        by default 1
    padding : int, optional
        by default 0
    dilation : int, optional
        by default 1
    groups : int, optional
        by default 1

    Returns
    -------
    torch.Tensor
        `x` convolved with `weight`
    """
    try:
        if torch.is_tensor(weight):
            return _CONVOLUTION[weight.ndim - 2](x, weight, bias=bias, stride=stride, padding=padding, 
                                                 dilation=dilation, groups=groups)
        else:
            if isinstance(weight, TTTensor):
                weight = tl.moveaxis(weight.to_tensor(), -1, 0)
            else:
                weight = weight.to_tensor()
            return _CONVOLUTION[weight.ndim - 2](x, weight, bias=bias, stride=stride, padding=padding, 
                                                 dilation=dilation, groups=groups)
    except KeyError:
        raise ValueError(f'Got tensor of order={weight.ndim} but pytorch only supports up to 3rd order (3D) Convs.')


def general_conv1d_(x, kernel, mode, bias=None, stride=1, padding=0, groups=1, dilation=1, verbose=False):
    """General 1D convolution along the mode-th dimension

    Parameters
    ----------
    x : batch-dize, in_channels, K1, ..., KN
    kernel : out_channels, in_channels/groups, K{mode}
    mode : int
        weight along which to perform the decomposition
    stride : int
    padding : int
    groups : 1
        typically would be equal to thhe number of input-channels
        at least for CP convolutions

    Returns
    -------
    x convolved with the given kernel, along dimension `mode`
    """
    if verbose:
        print(f'Convolving {x.shape} with {kernel.shape} along mode {mode}, '
              f'stride={stride}, padding={padding}, groups={groups}')

    in_channels = tl.shape(x)[1]
    n_dim = tl.ndim(x)
    permutation = list(range(n_dim))
    spatial_dim = permutation.pop(mode)
    channels_dim = permutation.pop(1)
    permutation += [channels_dim, spatial_dim]
    x = tl.transpose(x, permutation)
    x_shape = list(x.shape)
    x = tl.reshape(x, (-1, in_channels, x_shape[-1]))
    x = F.conv1d(x.contiguous(), kernel, bias=bias, stride=stride, dilation=dilation, padding=padding, groups=groups)
    x_shape[-2:] = x.shape[-2:]
    x = tl.reshape(x, x_shape)
    permutation = list(range(n_dim))[:-2]
    permutation.insert(1, n_dim - 2)
    permutation.insert(mode, n_dim - 1)
    x = tl.transpose(x, permutation)
    
    return x


def general_conv1d(x, kernel, mode, bias=None, stride=1, padding=0, groups=1, dilation=1, verbose=False):
    """General 1D convolution along the mode-th dimension

    Uses an ND convolution under the hood

    Parameters
    ----------
    x : batch-dize, in_channels, K1, ..., KN
    kernel : out_channels, in_channels/groups, K{mode}
    mode : int
        weight along which to perform the decomposition
    stride : int
    padding : int
    groups : 1
        typically would be equal to the number of input-channels
        at least for CP convolutions

    Returns
    -------
    x convolved with the given kernel, along dimension `mode`
    """
    if verbose:
        print(f'Convolving {x.shape} with {kernel.shape} along mode {mode}, '
              f'stride={stride}, padding={padding}, groups={groups}')

    def _pad_value(value, mode, order, padding=1):
        return tuple([value if i == (mode - 2) else padding for i in range(order)])

    ndim = tl.ndim(x)
    order = ndim - 2
    for i in range(2, ndim):
        if i != mode:
            kernel = kernel.unsqueeze(i)

    return _CONVOLUTION[order](x, kernel, bias=bias, 
                               stride=_pad_value(stride, mode, order),
                               padding=_pad_value(padding, mode, order, padding=0), 
                               dilation=_pad_value(dilation, mode, order), 
                               groups=groups)


def tucker_conv(x, tucker_tensor, bias=None, stride=1, padding=0, dilation=1):
    # Extract the rank from the actual decomposition in case it was changed by, e.g. dropout
    rank = tucker_tensor.rank

    batch_size = x.shape[0]
    n_dim = tl.ndim(x)

    # Change the number of channels to the rank
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

    # This can be done with a tensor contraction
    # First conv == tensor contraction
    # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
    x = F.conv1d(x, tl.transpose(tucker_tensor.factors[1]).unsqueeze(2))

    x_shape[1] = rank[1]
    x = x.reshape(x_shape)

    modes = list(range(2, n_dim+1))
    weight = tl.tenalg.multi_mode_dot(tucker_tensor.core, tucker_tensor.factors[2:], modes=modes)
    x = convolve(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation)

    # Revert back number of channels from rank to output_channels
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1))
    # Last conv == tensor contraction
    # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
    x = F.conv1d(x, tucker_tensor.factors[0].unsqueeze(2), bias=bias)

    x_shape[1] = x.shape[1]
    x = x.reshape(x_shape)

    return x


def tt_conv(x, tt_tensor, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized tt convolution

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

    Returns
    -------
    NDConv(x) with an tt kernel
    """
    shape = tt_tensor.shape
    rank = tt_tensor.rank

    batch_size = x.shape[0]
    order = len(shape) - 2

    if isinstance(padding, int):
        padding = (padding, )*order
    if isinstance(stride, int):
        stride = (stride, )*order
    if isinstance(dilation, int):
        dilation = (dilation, )*order

    # Change the number of channels to the rank
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

    # First conv == tensor contraction
    # from (1, in_channels, rank) to (rank == out_channels, in_channels, 1)
    x = F.conv1d(x, tl.transpose(tt_tensor.factors[0], [2, 1, 0]))

    x_shape[1] = x.shape[1]#rank[1]
    x = x.reshape(x_shape)

    # convolve over non-channels
    for i in range(order):
        # From (in_rank, kernel_size, out_rank) to (out_rank, in_rank, kernel_size)
        kernel = tl.transpose(tt_tensor.factors[i+1], [2, 0, 1])
        x = general_conv1d(x.contiguous(), kernel, i+2, stride=stride[i], padding=padding[i], dilation=dilation[i])

    # Revert back number of channels from rank to output_channels
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1))
    # Last conv == tensor contraction
    # From (rank, out_channels, 1) to (out_channels, in_channels == rank, 1)
    x = F.conv1d(x, tl.transpose(tt_tensor.factors[-1], [1, 0, 2]), bias=bias)

    x_shape[1] = x.shape[1]
    x = x.reshape(x_shape)

    return x


def cp_conv(x, cp_tensor, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized CP convolution

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

    Returns
    -------
    NDConv(x) with an CP kernel
    """
    shape = cp_tensor.shape
    rank = cp_tensor.rank

    batch_size = x.shape[0]
    order = len(shape) - 2

    if isinstance(padding, int):
        padding = (padding, )*order
    if isinstance(stride, int):
        stride = (stride, )*order
    if isinstance(dilation, int):
        dilation = (dilation, )*order

    # Change the number of channels to the rank
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

    # First conv == tensor contraction
    # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
    x = F.conv1d(x, tl.transpose(cp_tensor.factors[1]).unsqueeze(2))

    x_shape[1] = rank
    x = x.reshape(x_shape)

    # convolve over non-channels
    for i in range(order):
        # From (kernel_size, rank) to (rank, 1, kernel_size)
        kernel = tl.transpose(cp_tensor.factors[i+2]).unsqueeze(1)             
        x = general_conv1d(x.contiguous(), kernel, i+2, stride=stride[i], padding=padding[i], dilation=dilation[i], groups=rank)

    # Revert back number of channels from rank to output_channels
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1))                
    # Last conv == tensor contraction
    # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
    x = F.conv1d(x*cp_tensor.weights.unsqueeze(1).unsqueeze(0), cp_tensor.factors[0].unsqueeze(2), bias=bias)

    x_shape[1] = x.shape[1] # = out_channels
    x = x.reshape(x_shape)

    return x


def cp_conv_mobilenet(x, cp_tensor, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized CP convolution

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

    Returns
    -------
    NDConv(x) with an CP kernel
    """
    factors = cp_tensor.factors
    shape = cp_tensor.shape
    rank = cp_tensor.rank

    batch_size = x.shape[0]
    order = len(shape) - 2

    # Change the number of channels to the rank
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

    # First conv == tensor contraction
    # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
    x = F.conv1d(x, tl.transpose(factors[1]).unsqueeze(2))

    x_shape[1] = rank
    x = x.reshape(x_shape)

    # convolve over merged actual dimensions
    # Spatial convs
    # From (kernel_size, rank) to (out_rank, 1, kernel_size)
    if order == 1:
        weight = tl.transpose(factors[2]).unsqueeze(1)
        x = F.conv1d(x.contiguous(), weight, stride=stride, padding=padding, dilation=dilation, groups=rank)
    elif order == 2:
        weight = tenalg.tensordot(tl.transpose(factors[2]), 
                                  tl.transpose(factors[3]), modes=(), batched_modes=0
                                  ).unsqueeze(1)
        x = F.conv2d(x.contiguous(), weight, stride=stride, padding=padding, dilation=dilation, groups=rank)
    elif order == 3:
        weight = tenalg.tensordot(tl.transpose(factors[2]), 
                                  tenalg.tensordot(tl.transpose(factors[3]), tl.transpose(factors[4]), modes=(), batched_modes=0),
                                  modes=(), batched_modes=0
                                  ).unsqueeze(1)
        x = F.conv3d(x.contiguous(), weight, stride=stride, padding=padding, dilation=dilation, groups=rank)

    # Revert back number of channels from rank to output_channels
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1))

    # Last conv == tensor contraction
    # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
    x = F.conv1d(x*cp_tensor.weights.unsqueeze(1).unsqueeze(0), factors[0].unsqueeze(2), bias=bias)

    x_shape[1] = x.shape[1] # = out_channels
    x = x.reshape(x_shape)

    return x


def _get_factorized_conv(factorization, implementation='factorized'):
    if implementation == 'reconstructed' or factorization == 'Dense':
        return convolve
    if isinstance(factorization, CPTensor):
        if implementation == 'factorized':
            return cp_conv
        elif implementation == 'mobilenet':
            return cp_conv_mobilenet
    elif isinstance(factorization, TuckerTensor):
        return tucker_conv
    elif isinstance(factorization, TTTensor):
        return tt_conv
    raise ValueError(f'Got unknown type {factorization}')


def convNd(x, weight, bias=None, stride=1, padding=0, dilation=1, implementation='factorized'):
    if implementation=='reconstructed':
        weight = weight.to_tensor()

    if isinstance(weight, DenseTensor):
        return convolve(x, weight.tensor, bias=bias, stride=stride, padding=padding, dilation=dilation)

    if torch.is_tensor(weight):
        return convolve(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    
    if isinstance(weight, CPTensor):
        if implementation == 'factorized':
            return cp_conv(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
        elif implementation == 'mobilenet':
            return cp_conv_mobilenet(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    elif isinstance(weight, TuckerTensor):
        return tucker_conv(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    elif isinstance(weight, TTTensor):
        return tt_conv(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
