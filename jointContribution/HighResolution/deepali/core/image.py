import math
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import paddle

from ..utils import paddle_aux
from .enum import PaddingMode
from .enum import Sampling
from .enum import SpatialDerivativeKeys
from .enum import SpatialDim
from .enum import SpatialDimArg
from .grid import ALIGN_CORNERS
from .grid import Axes
from .grid import Grid
from .grid import grid_transform_points
from .kernels import gaussian1d
from .kernels import gaussian1d_I
from .nnutils import same_padding
from .nnutils import stride_minus_kernel_padding
from .random import multinomial
from .tensor import as_tensor
from .tensor import cat_scalars
from .tensor import move_dim
from .types import Array
from .types import Device
from .types import Scalar
from .types import ScalarOrTuple
from .types import Shape
from .types import Size
from .types import is_float_dtype


def avg_pool(
    data: paddle.Tensor,
    kernel_size: ScalarOrTuple[int],
    stride: Optional[ScalarOrTuple[int]] = None,
    padding: Optional[ScalarOrTuple[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[Scalar] = None,
) -> paddle.Tensor:
    """Average pooling of image data."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("avg_pool() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("avg_pool() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if D == 1:
        avg_pool_fn = paddle.nn.functional.avg_pool1d
    elif D == 2:
        avg_pool_fn = paddle.nn.functional.avg_pool2d
    elif D == 3:
        avg_pool_fn = paddle.nn.functional.avg_pool3d
    else:
        raise ValueError(
            "avg_pool() number of spatial 'data' dimensions must be 1, 2, or 3"
        )
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * D
    elif len(kernel_size) != D:
        raise ValueError(f"avg_pool() 'kernel_size' must be scalar or {D}-tuple")
    if stride is None:
        stride = kernel_size
    if padding is None:
        padding = tuple(n // 2 for n in kernel_size)
    return avg_pool_fn(
        data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


def max_pool(
    data: paddle.Tensor,
    kernel_size: ScalarOrTuple[int],
    stride: Optional[ScalarOrTuple[int]] = None,
    padding: Optional[ScalarOrTuple[int]] = 0,
    dilation: ScalarOrTuple[int] = 1,
    ceil_mode: bool = False,
) -> paddle.Tensor:
    """Max pooling of image data."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("max_pool() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("max_pool() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if D == 1:
        max_pool_fn = paddle.nn.functional.max_pool1d
    elif D == 2:
        max_pool_fn = paddle.nn.functional.max_pool2d
    elif D == 3:
        max_pool_fn = paddle.nn.functional.max_pool3d
    else:
        raise ValueError(
            "max_pool() number of spatial 'data' dimensions must be 1, 2, or 3"
        )
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * D
    elif len(kernel_size) != D:
        raise ValueError(f"max_pool() 'kernel_size' must be scalar or {D}-tuple")
    if stride is None:
        stride = kernel_size
    if padding is None:
        padding = tuple(n // 2 for n in kernel_size)
    return max_pool_fn(
        data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def min_pool(
    data: paddle.Tensor,
    kernel_size: ScalarOrTuple[int],
    stride: Optional[ScalarOrTuple[int]] = None,
    padding: Optional[ScalarOrTuple[int]] = 0,
    dilation: ScalarOrTuple[int] = 1,
    ceil_mode: bool = False,
) -> paddle.Tensor:
    """Min pooling of image data, i.e., negate max_pool() result of negated input data."""
    return -max_pool(
        -data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def conv(
    data: paddle.Tensor,
    kernel: Union[paddle.Tensor, Sequence[Optional[paddle.Tensor]]],
    stride: ScalarOrTuple[int] = 1,
    dilation: ScalarOrTuple[int] = 1,
    padding: Union[PaddingMode, str, ScalarOrTuple[int]] = None,
    output_padding: Optional[ScalarOrTuple[int]] = None,
    transpose: bool = False,
) -> paddle.Tensor:
    """Convolve images in batch with a given (separable) kernel.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        kernel: paddle.Tensor of shape ``(..., X)`` with weights of kernel used to filter the images
            in this batch by. If the input ``data`` tensor is of non-floating point type, the
            dtype of the kernel defines the intermediate data type used for convolutions.
            If a 1-dimensional kernel is given, it is used as separable convolution kernel in
            all spatial image dimensions. Otherwise, the kernel is applied to the last spatial
            image dimensions. For example, a 2D kernel applied to a batch of 3D image volumes
            is applied slice-by-slice by convolving along the y and x image axes.
            In order to anisotropically convolve the input data with 1-dimensional kernels of
            different sizes, a sequence of at most ``D`` 1-dimensional kernel tensors can be given,
            where ``D`` is the number of spatial dimensions. If the sequence contains ``None``,
            no convolution is performed along the corresponding spatial dimension. The first kernel
            in the sequence is applied to the last spatial grid dimension, which corresponds to
            the ``data`` tensor dimension ``X``, e.g., ``(kz, ky, kx)``.
        stride: Stride by which convolution kernel is advanced.
        dilation: Spacing between kernel elements.
        padding: Image padding mode. If ``int``, pad with zeros the specified margin at each
            side. Otherwise, use ``same_padding()`` calculated from kernel size and dilation such
            that output size is equal to input size, unless ``PaddingMode.NONE`` is given. If ``None``,
            use default mode ``PaddingMode.ZEROS`` with "same" padding.
        output_padding: Output padding for transposed convolution.
        transpose: Whether to compute transposed convolution.

    Returns:
        Result of filtering operation with data type set to the image data type before convolution.
        If dtype is not a floating point data type, the filtered data is being rounded and clamped.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("conv() 'data' must be paddle.Tensor")
    if data.ndim < 3:
        raise ValueError("conv() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if isinstance(kernel, paddle.Tensor) and kernel.ndim == 1:
        kernel = [kernel] * D
    dtype = data.dtype
    if isinstance(kernel, paddle.Tensor):
        if isinstance(padding, int):
            margin = (padding,) * kernel.ndim
            padding = PaddingMode.ZEROS
        elif isinstance(padding, Sequence):
            margin = padding
            padding = PaddingMode.ZEROS
        else:
            margin = same_padding(tuple(kernel.shape), dilation)
            padding = PaddingMode.from_arg(padding)
        kernel_dtype = kernel.dtype
    else:
        kernel = list(kernel)
        kernel_dtype = None
        kernel_size = []
        for k in kernel:
            if k is None:
                kernel_size.append(1)
            else:
                if kernel_dtype is None:
                    kernel_dtype = k.dtype
                if k.ndim != 1:
                    raise ValueError(
                        "conv() 'kernel' must be n-dimensional tensor or sequence of 1-dimensional tensors"
                    )
                kernel_size.append(len(k))
        if kernel_dtype is None:
            return data
        if isinstance(padding, int):
            margin = (padding,) * len(kernel)
            padding = PaddingMode.ZEROS
        elif isinstance(padding, Sequence):
            margin = padding
            padding = PaddingMode.ZEROS
        else:
            margin = same_padding(kernel_size, dilation)
            padding = PaddingMode.from_arg(padding)
    if sum(margin) != 0 and padding not in (PaddingMode.NONE, PaddingMode.ZEROS):
        if transpose:
            raise NotImplementedError(
                f"conv() 'transpose=True' with padding {padding.value}"
            )
        margin = tuple(reversed(margin))
        tensor = pad(data, margin=margin, mode=padding)
        return conv(
            tensor, kernel, stride=stride, dilation=dilation, padding=PaddingMode.NONE
        )
    if not is_float_dtype(dtype):
        dtype = kernel_dtype if is_float_dtype(kernel_dtype) else "float32"
    tensor = data.astype(dtype)
    device = tensor.place
    if isinstance(kernel, paddle.Tensor):
        K = kernel.ndim
        if K > D:
            raise ValueError("conv() 'kernel' has too many dimensions")
        if K == 2:
            conv_fn = (
                paddle.nn.functional.conv2d_transpose
                if transpose
                else paddle.nn.functional.conv2d
            )
        elif K == 3:
            conv_fn = (
                paddle.nn.functional.conv3d_transpose
                if transpose
                else paddle.nn.functional.conv3d
            )
        else:
            raise ValueError("conv() 'kernel' must have 1, 2, or 3 spatial dimensions")
        shape_ = tuple(tensor.shape)
        kernel = kernel.to(dtype=dtype, device=device)
        kernel = kernel.reshape(1, 1, *tuple(kernel.shape))
        if tensor.ndim > kernel.ndim:
            groups = tuple(tensor.shape)[1:-K].size
            tensor = tensor.reshape(
                tuple(tensor.shape)[0], groups, *tuple(tensor.shape)[-K:]
            )
        else:
            groups = tuple(tensor.shape)[1]
        weight = kernel.expand(shape=[groups, 1, *tuple(kernel.shape)[-K:]])
        kwargs = dict(
            tensor,
            weight,
            stride=stride,
            dilation=dilation,
            padding=0 if padding == PaddingMode.NONE else margin,
            groups=groups,
        )
        if transpose:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, stride)
            kwargs["output_padding"] = output_padding
        result = conv_fn(tensor, weight, **kwargs)
        result = result.reshape(shape_[:-K] + tuple(result.shape)[-K:])
    else:
        result = tensor
        kernels = list(kernel)
        for i, k in enumerate(kernels):
            if k is not None:
                kernels[i] = k.to(dtype=dtype, device=device)
        K = len(kernels)
        if isinstance(stride, int):
            stride = (stride,) * K
        elif len(stride) != K:
            raise ValueError(f"conv() 'stride' must be int or sequence of {K} ints")
        if isinstance(dilation, int):
            dilation = (dilation,) * K
        elif len(dilation) != K:
            raise ValueError(f"conv() 'dilation' must be int or sequence of {K} ints")
        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride)
        elif isinstance(output_padding, int):
            output_padding = (output_padding,) * K
        elif len(output_padding) != K:
            raise ValueError(
                f"conv() 'output_padding' must be None, int, or sequence of {K} ints"
            )
        args = zip(kernels, stride, dilation, margin, output_padding)
        for i, (k, s, d, p, op) in enumerate(args):
            if k is None:
                continue
            result = conv1d(
                result,
                k,
                dim=result.ndim - K + i,
                stride=s,
                dilation=d,
                padding=0 if padding == PaddingMode.NONE else p,
                output_padding=op,
                transpose=transpose,
            )
    if not paddle.is_floating_point(x=data):
        result = result.round_()
        result = result.clip_(min=float(data.min()), max=float(data.max()))
    result = result.astype(dtype=data.dtype)
    return result


def conv1d(
    data: paddle.Tensor,
    kernel: paddle.Tensor,
    dim: int = -1,
    stride: int = 1,
    dilation: int = 1,
    padding: Union[PaddingMode, str, int] = None,
    output_padding: Optional[int] = None,
    transpose: bool = False,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    """Convolve data with 1-dimensional kernel along specified dimension."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("conv1d() 'data' must be paddle.Tensor")
    if data.ndim < 3:
        raise ValueError("conv1d() 'data' must have shape (N, C, ..., X)")
    if not isinstance(kernel, paddle.Tensor):
        raise TypeError("conv1d() 'kernel' must be of type paddle.Tensor")
    if kernel.ndim != 1:
        raise ValueError("conv1d() 'kernel' must be 1-dimensional")
    if dtype is None:
        dtype = data.dtype
    if is_float_dtype(dtype):
        kernel = kernel.astype(dtype)
    elif not is_float_dtype(kernel.dtype):
        kernel = kernel.astype("float32")
    if isinstance(padding, int):
        margin = padding
        padding = PaddingMode.ZEROS
    else:
        padding = PaddingMode.from_arg(padding)
        if padding is PaddingMode.NONE:
            margin = 0
        else:
            margin = same_padding(tuple(kernel.shape), dilation)
    result = data.astype(kernel.dtype)
    result = move_dim(result, dim, -1)
    shape_ = result.shape
    result = result.reshape([shape_[0], -1, shape_[-1]])
    groups = result.shape[1]
    weight = kernel.expand([groups, 1, kernel.shape[-1]])
    result = result.reshape(shape_[0], groups, shape_[-1])
    if margin and padding is not PaddingMode.ZEROS:
        result = paddle_aux._FUNCTIONAL_PAD(
            pad=(margin, margin), mode=padding.pad_mode(1), x=result
        )
        margin = 0
    conv_fn = (
        paddle.nn.functional.conv1d_transpose
        if transpose
        else paddle.nn.functional.conv1d
    )
    kwargs = dict(stride=stride, dilation=dilation, padding=margin, groups=groups)
    if transpose:
        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride)
        kwargs["output_padding"] = output_padding
    result = conv_fn(result, weight, **kwargs)
    result = result.reshape(shape_[0:-1] + list(result.shape)[-1:])
    result = move_dim(result, -1, dim)
    if not is_float_dtype(dtype):
        result = result.round_()
        result = result.clip_(min=float(data.min()), max=float(data.max()))
    result = result.astype(dtype)
    return result


def dot_batch(
    a: paddle.Tensor, b: paddle.Tensor, weight: Optional[paddle.paddle.Tensor] = None
) -> paddle.Tensor:
    """Weighted dot product between batches of image batch tensors.

    Args:
        a: Image data tensor of shape ``(N, C, ..., X)``.
        b: Image data tensor of shape ``(N, C, ..., X)``.

    Returns:
        paddle.Tensor of shape ``(N,)`` containing batchwise dot products.

    """
    return dot_channels(a, b, weight=weight).sum(axis=1)


def dot_channels(
    a: paddle.Tensor, b: paddle.Tensor, weight: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    """Weighted dot product between channels of image batch tensors.

    Args:
        a: Image data tensor of shape ``(N, C, ..., X)``.
        b: Image data tensor of shape ``(N, C, ..., X)``.

    Returns:
        paddle.Tensor of shape ``(N, C)`` containing channelwise dot products.

    """
    if not isinstance(a, paddle.Tensor):
        raise TypeError("dot_channels() 'a' and 'b' must be tensors")
    if a.ndim < 4:
        raise ValueError("dot_channels() 'a' and 'b' must have shape (N, C, ..., X)")
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError("dot_channels() 'a' and 'b' must have identical shape")
    c = a * b
    if weight is not None:
        c *= weight
    return c.view(tuple(c.shape)[0], tuple(c.shape)[1], -1).sum(axis=2)


def downsample(
    data: paddle.Tensor,
    levels: int = 1,
    dims: Optional[Sequence[SpatialDimArg]] = None,
    sigma: Optional[Union[Scalar, Array]] = None,
    mode: Optional[Union[Sampling, str]] = None,
    min_size: int = 0,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Downsample images after optional convolution with truncated Gaussian kernel.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        levels: Number of times the image size is halved. If zero, a reference to the
            unmodified input ``data`` tensor is returned. If negative, the images are
            upsampled instead.
        dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
        sigma: Standard deviation of Gaussian used for each downsampling step.
            If a scalar or 1-element sequence is given, an isotropic Gaussian
            kernel is used. Otherwise, the first value is the standard deviation
            of the 1-dimensional Gaussian applied to the first grid dimension, which
            is the last ``data`` tensor dimension, e.g., ``(sx, sy)``. If ``sigma=0``,
            no low-pass filter is applied. If ``None``, a default value is used.
        min_size: Required minimum grid size.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Downsampled image data.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("downsample() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("downsample() 'data' must have shape (N, C, ..., X)")
    if not paddle.is_floating_point(x=data):
        raise TypeError("downsample() 'data' must have floating point dtype")
    try:
        levels = int(levels)
    except TypeError:
        raise TypeError("downsample() 'levels' must be scalar of type int")
    if levels == 0:
        return data
    if levels < 0:
        return upsample(
            data,
            levels=-levels,
            dims=dims,
            sigma=sigma,
            mode=mode,
            align_corners=align_corners,
        )
    grid = Grid(shape=tuple(data.shape)[2:])
    if not dims:
        dims = tuple(dim for dim in range(grid.ndim))
    dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
    grid = grid.downsample(
        levels, dims=dims, min_size=min_size, align_corners=align_corners
    )
    if sigma is None:
        sigma = 0.7355
    sigma: paddle.Tensor = paddle.atleast_1d(as_tensor(sigma, dtype="float32"))
    if sigma.ndim > 1:
        raise ValueError("downsample() 'sigma' must be scalar or 1-dimensional tensor")
    if sigma.not_equal(y=paddle.to_tensor(0.0)).astype("bool").any():
        if levels > 1:
            var = paddle.zeros(shape=tuple(sigma.shape), dtype=sigma.dtype)
            for level in range(levels):
                var += sigma.mul(2**level).pow(y=2)
            sigma = var.sqrt()
        if tuple(sigma.shape)[0] == 1:
            sigma = sigma.repeat(grid.ndim)
            for i in range(grid.ndim):
                if i in dims:
                    continue
                sigma[i] = 0
        kernels = []
        kernels_ = {}
        for i in range(grid.ndim):
            std = float(sigma[i] if i < len(sigma) else 0)
            if std > 0 and (
                tuple(grid.size())[i] != tuple(data.shape)[grid.ndim - i + 1]
            ):
                kernel = kernels_.get(std)
                if kernel is None:
                    kernel = gaussian1d(std, dtype="float32", device=data.place)
                    kernels_[std] = kernel
                kernels.append(kernel)
            else:
                kernels.append(None)
        data = conv(data, reversed(kernels))
    mode = Sampling.from_arg(mode).interpolate_mode(grid.ndim)
    return paddle.nn.functional.interpolate(
        x=data, size=tuple(grid.shape), mode=mode, align_corners=align_corners
    )


def upsample(
    data: paddle.Tensor,
    levels: int = 1,
    dims: Optional[Sequence[SpatialDimArg]] = None,
    sigma: Optional[Union[Scalar, Array]] = None,
    mode: Optional[Union[Sampling, str]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Upsample images and opitonally deconvolve with truncated Gaussian kernel.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        levels: Number of times the image size is doubled. If zero, a reference to the
            unmodified input ``data`` tensor is returned. If negative, the images are
            downsampled instead.
        dims: Spatial dimensions along which to upsample. If not specified, consider all spatial dimensions.
        sigma: Standard deviation of Gaussian used for each upsampling step.
            If a scalar or 1-element sequence is given, an isotropic Gaussian
            kernel is used. Otherwise, the first value is the standard deviation
            of the 1-dimensional Gaussian applied to the first grid dimension, which
            is the last ``data`` tensor dimension, e.g., ``(sx, sy)``. If ``sigma=0``
            or ``None``, no transposed convolution is applied.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Upsampled image data.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("upsample() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("upsample() 'data' must have shape (N, C, ..., X)")
    if not paddle.is_floating_point(x=data):
        raise TypeError("upsample() 'data' must have floating point dtype")
    try:
        levels = int(levels)
    except TypeError:
        raise TypeError("upsample() 'levels' must be scalar of type int")
    if levels == 0:
        return data
    if levels < 0:
        return downsample(
            data,
            levels=-levels,
            dims=dims,
            sigma=sigma,
            mode=mode,
            align_corners=align_corners,
        )
    grid = Grid(shape=tuple(data.shape)[2:], align_corners=align_corners)
    if not dims:
        dims = tuple(dim for dim in range(grid.ndim))
    dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
    grid = grid.upsample(levels, dims=dims)
    mode = Sampling.from_arg(mode).interpolate_mode(grid.ndim)
    result: paddle.Tensor = paddle.nn.functional.interpolate(
        x=data, size=tuple(grid.shape), mode=mode, align_corners=align_corners
    )
    if sigma is not None:
        sigma: paddle.Tensor = paddle.atleast_1d(as_tensor(sigma, dtype="float32"))
        if sigma.ndim > 1:
            raise ValueError(
                "upsample() 'sigma' must be scalar or 1-dimensional tensor"
            )
        if levels > 1:
            var = paddle.zeros(shape=tuple(sigma.shape), dtype=sigma.dtype)
            for level in range(levels):
                var += sigma.mul(2**level).pow(y=2)
            sigma = var.sqrt()
        if tuple(sigma.shape)[0] == 1:
            sigma = sigma.repeat(grid.ndim)
            for i in range(grid.ndim):
                if i in dims:
                    continue
                sigma[i] = 0
        kernels = []
        kernels_ = {}
        for i in range(grid.ndim):
            std = float(sigma[i] if i < len(sigma) else 0)
            if std > 0:
                kernel = kernels_.get(std)
                if kernel is None:
                    kernel = gaussian1d(std, dtype="float32", device=result.place)
                    kernels_[std] = kernel
                kernels.append(kernel)
            else:
                kernels.append(None)
        result = conv(result, reversed(kernels), transpose=True)
    return result


def gaussian_pyramid(
    data: paddle.Tensor,
    levels: int,
    start: int = 0,
    dims: Optional[Sequence[SpatialDimArg]] = None,
    sigma: Optional[Union[Scalar, Array]] = None,
    mode: Optional[Union[Sampling, str]] = None,
    min_size: int = 0,
    align_corners: bool = ALIGN_CORNERS,
) -> Dict[int, paddle.Tensor]:
    """Create Gaussian image resolution pyramid.

    Args:
        data: Image data tensor of shape ``(N, C, ..., X)``.
        levels: Coarsest resolution level.
        start: Finest resolution level, where 0 corresponds to the original resolution.
        dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
        sigma: Standard deviation of Gaussian filter applied at each downsampling level.
        mode: Interpolation mode for resampling image data on downsampled grid.
        min_size: Minimum grid size.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Dictionary of downsampled image tensors with keys corresponding to level indices.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("gaussian_pyramid() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("gaussian_pyramid() 'data' must have shape (N, C, ..., X)")
    if not isinstance(levels, int):
        raise TypeError("gaussian_pyramid() 'levels' must be of type int")
    if levels < 0:
        raise ValueError("gaussian_pyramid() 'levels' must be positive")
    if not isinstance(start, int):
        raise TypeError("gaussian_pyramid() 'start' must be of type int")
    if start < 0:
        raise ValueError("gaussian_pyramid() 'start' must not be negative")
    pyramid = {(0): data}
    if start == 0:
        start = 1
        levels -= 1
    for i, level in enumerate(range(start, start + levels)):
        data = downsample(
            data,
            levels=level if i == 0 else 1,
            dims=dims,
            sigma=sigma,
            mode=mode,
            min_size=min_size,
            align_corners=align_corners,
        )
        pyramid[level] = data
    return pyramid


def crop(
    data: paddle.Tensor,
    margin: Optional[Union[int, Array]] = None,
    num: Optional[Union[int, Array]] = None,
    mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
    value: Scalar = 0,
) -> paddle.Tensor:
    """Crop or pad images at border.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        margin: Number of spatial grid points to remove (positive) or add (negative) at each border.
            Use instead of ``num`` in order to symmetrically crop the input ``data`` tensor, e.g.,
            ``(nx, ny, nz)`` is equivalent to ``num=(nx, nx, ny, ny, nz, nz)``.
        num: Number of spatial gird points to remove (positive) or add (negative) at each border,
            where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
            ``(nx, nx, ny, ny)``. If a scalar is given, the input is cropped equally at all borders.
            Otherwise, the given sequence must have an even length.
        mode: Image extrapolation mode.
        value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

    Returns:
        Cropped or padded image batch data of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("crop() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("crop() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if num is not None and margin is not None:
        raise AssertionError("crop() 'margin' and 'num' are mutually exclusive")
    if isinstance(num, int):
        pad_ = (-num,) * 2 * D
    elif num is not None:
        pad_ = tuple(-int(n) for n in num)
        if len(pad_) % 2 == 1:
            raise ValueError("crop() 'num' must be int or have even length")
    elif isinstance(margin, int):
        pad_ = (-margin,) * 2 * D
    elif margin is not None:
        pad_ = tuple(-int(n) for nn in ((n, n) for n in margin) for n in nn)
    else:
        raise AssertionError("crop() either 'margin' or 'num' is required")
    if all(n == 0 for n in pad_):
        return data
    mode = PaddingMode.from_arg(mode)
    if mode == PaddingMode.ZEROS:
        mode = PaddingMode.CONSTANT
        value = 0
    else:
        value = float(value)
    mode = mode.pad_mode(D)
    return paddle_aux._FUNCTIONAL_PAD(pad=pad_, mode=mode, value=value, x=data)


def pad(
    data: paddle.Tensor,
    margin: Optional[Union[int, Array]] = None,
    num: Optional[Union[int, Array]] = None,
    mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
    value: Scalar = 0,
) -> paddle.Tensor:
    """Pad or crop images at border.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        margin: Number of spatial grid points to add (positive) or remove (negative) at each border,
            Use instead of ``num`` in order to symmetrically pad the input ``data`` tensor.
        num: Number of spatial gird points to add (positive) or remove (negative) at each border,
            where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
            ``(nx, ny, nz)``. If a scalar is given, the input is padded equally at all borders.
            Otherwise, the given sequence must have an even length.
        mode: Image extrapolation mode.
        value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

    Returns:
        Padded or cropped image batch data of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("pad() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("pad() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if num is not None and margin is not None:
        raise AssertionError("pad() 'margin' and 'num' are mutually exclusive")
    if isinstance(num, int):
        pad_ = (num,) * 2 * D
    elif num is not None:
        pad_ = tuple(int(n) for n in num)
        if len(pad_) % 2 == 1:
            raise ValueError("pad() 'num' must be int or have even length")
    elif isinstance(margin, int):
        pad_ = (margin,) * 2 * D
    elif margin is not None:
        pad_ = tuple(int(n) for nn in ((n, n) for n in margin) for n in nn)
    else:
        raise AssertionError("pad() either 'pad' or 'margin' is required")
    if all(n == 0 for n in pad_):
        return data
    mode = PaddingMode.from_arg(mode)
    if mode == PaddingMode.ZEROS:
        mode = PaddingMode.CONSTANT
        value = 0
    else:
        value = float(value)
    mode = mode.pad_mode(D)
    return paddle_aux._FUNCTIONAL_PAD(pad=pad_, mode=mode, value=value, x=data)


def center_crop(data: paddle.Tensor, size: Union[int, Sequence[int]]) -> paddle.Tensor:
    """Crop image tensor to specified maximum size.

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.
        size: Maximum output size, where the size of the last tensor
            dimension must be given first, i.e., ``(X, ...)``.
            If an ``int`` is given, all spatial output dimensions
            are cropped to this maximum size. If the length of size
            is less than the spatial dimensions of the ``data`` tensor,
            then only the last ``len(size)`` dimensions are modified.

    Returns:
        Output tensor of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("center_crop() 'data' must be paddle.Tensor")
    if data.dim() < 4:
        raise ValueError("center_crop() 'data' must be tensor of shape (N, C, ..., X)")
    sdim = data.dim() - 2
    if isinstance(size, int):
        shape = (size,) * sdim
    elif len(size) == 0:
        return data
    elif len(size) > sdim:
        raise ValueError(
            "center_crop() 'data' has fewer spatial dimensions than output 'size'"
        )
    else:
        shape = tuple(data.shape)[2:][: sdim - len(size)] + tuple(reversed(size))
    crop = [max(0, m - n) for m, n in zip(tuple(data.shape)[2:], shape)]
    if sum(crop) == 0:
        return data
    crop = (n // 2 for n in crop)
    crop = (slice(0, tuple(data.shape)[0]), slice(0, tuple(data.shape)[1])) + tuple(
        slice(i, i + n) for i, n in zip(crop, shape)
    )
    return data[crop]


def center_pad(
    data: paddle.Tensor,
    size: Union[int, Sequence[int]],
    mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
    value: Scalar = 0,
) -> paddle.Tensor:
    """Pad image tensor to specified minimum size.

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.
        size: Minimum output size, where the size of the last tensor
            dimension must be given first, i.e., ``(X, ...)``.
            If an ``int`` is given, all spatial output dimensions
            are cropped or padded to this size. If the length of size
            is less than the spatial dimensions of the ``data`` tensor,
            then only the last ``len(size)`` dimensions are modified.
        mode: PaddingMode mode (cf. ``paddle.nn.functional.pad()``).
        value: Value for padding mode "constant".

    Returns:
        Output tensor of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("center_pad() 'data' must be paddle.Tensor")
    if data.dim() < 4:
        raise ValueError("center_pad() 'data' must be tensor of shape (N, C, ..., X)")
    sdim = data.dim() - 2
    if isinstance(size, int):
        shape = (size,) * sdim
    elif len(size) == 0:
        return data
    elif len(size) > sdim:
        raise ValueError(
            "center_pad() 'data' has fewer spatial dimensions than output 'size'"
        )
    else:
        shape = tuple(data.shape)[2:][: sdim - len(size)] + tuple(reversed(size))
    pad = [max(0, n - m) for m, n in zip(tuple(data.shape)[2:], shape)]
    pad = [(n // 2, (n + 1) // 2) for n in reversed(pad)]
    pad = [n for x in pad for n in x]
    if sum(pad) == 0:
        return data
    mode = PaddingMode.from_arg(mode)
    if mode == PaddingMode.ZEROS:
        mode = PaddingMode.CONSTANT
        value = 0
    mode = mode.pad_mode(sdim)
    return paddle_aux._FUNCTIONAL_PAD(pad=pad, mode=mode, value=value, x=data)


def region_of_interest(
    data: paddle.Tensor,
    start: ScalarOrTuple[int],
    size: ScalarOrTuple[int],
    padding: Union[PaddingMode, str, float] = PaddingMode.CONSTANT,
    value: float = 0,
) -> paddle.Tensor:
    """Extract region of interest from image tensor.

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.
        start: Indices of lower left corner of region of interest, e.g., ``(x, y, z)``.
        size: Size of region of interest, e.g., ``(nx, ny, nz)``.
        padding: Padding mode to use when extrapolating input image or constant fill value.
        value: Fill value to use when ``padding=Padding.CONSTANT``.

    Returns:
        paddle.Tensor of shape ``(N, C, ..., X)`` with spatial size equal to ``size``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("region_of_interest() 'data' must be paddle.Tensor")
    if data.dim() < 4:
        raise ValueError(
            "region_of_interest() 'data' must be tensor of shape (N, C, ..., X)"
        )
    sdim = data.dim() - 2
    num = []
    if isinstance(start, int):
        start = (start,) * sdim
    elif not isinstance(start, Sequence) or not all(isinstance(n, int) for n in start):
        raise TypeError("region_of_interest() 'start' must be int or sequence of ints")
    elif len(start) != 3:
        raise ValueError(
            f"region_of_interest() 'start' must be int or sequence of length {sdim}"
        )
    if isinstance(size, int):
        size = (size,) * sdim
    elif not isinstance(size, Sequence) or not all(isinstance(n, int) for n in size):
        raise TypeError("region_of_interest() 'size' must be int or sequence of ints")
    elif len(size) != 3:
        raise ValueError(
            f"region_of_interest() 'size' must be int or sequence of length {sdim}"
        )
    if isinstance(padding, (PaddingMode, str)):
        mode = PaddingMode.from_arg(padding)
        value = value
    elif isinstance(padding, (int, float)):
        mode = PaddingMode.CONSTANT
        value = padding
    else:
        raise TypeError(
            "region_of_interest() 'padding' must be str, Padding, or fill value"
        )
    num = [
        [start[i], tuple(data.shape)[data.ndim - 1 - i] - (start[i] + size[i])]
        for i in range(sdim)
    ]
    num = [n for nn in num for n in nn]
    return crop(data, num=num, mode=mode, value=value)


def fill_border(
    data: paddle.Tensor,
    margin: ScalarOrTuple[int],
    value: float = 0,
    inplace: bool = False,
) -> paddle.Tensor:
    """Fill image border with specified value."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("fill_border() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("fill_border() 'data' must have shape (N, C, ..., X)")
    if isinstance(margin, int):
        margin = (margin,) * (data.ndim - 2)
    if len(margin) > data.ndim - 2:
        raise ValueError(
            f"fill_border() 'margin' must be at most {data.ndim - 2}-dimensional"
        )
    if not inplace:
        data = data.clone()
    for i, m in enumerate(margin):
        dim = data.ndim - i - 1
        idx = slice(0, m)
        idx = tuple(idx if j == dim else slice(None) for j in range(data.ndim))
        data[idx] = value
        idx = slice(tuple(data.shape)[dim] - m, tuple(data.shape)[dim])
        idx = tuple(idx if j == dim else slice(None) for j in range(data.ndim))
        data[idx] = value
    return data


def flatten_channels(data: paddle.Tensor) -> paddle.Tensor:
    """Flatten image tensor channels.


    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.

    Returns:
        paddle.Tensor of shape ``(C, N * ... * X)``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("flatten_channels() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("flatten_channels() 'data' must have shape (N, C, ..., X)")
    C = data.shape[1]
    axis_order = (1, 0) + tuple(range(2, data.ndim))
    transposed = data.transpose(perm=axis_order)
    return transposed.view(C, -1)


def grid_resample(
    data: paddle.Tensor,
    in_spacing: Union[float, Array],
    out_spacing: Union[float, Array],
    *args: float,
    mode: Union[Sampling, str] = None,
    padding: Union[PaddingMode, str, Scalar] = None,
) -> paddle.Tensor:
    """Interpolate image on minimum bounding grid with specified spacing.

    Args:
        data: Image batch tensor with shape ``(N, C, ..., X)``.
        in_spacing: Current grid spacing.
        out_spacing: Spacing of grid on which to sample images, where the spacing
            of the first grid dimension, which is the last ``data`` dimension
            must be given first, e.g., ``(sx, sy, sz)``. If a scalar value is
            given, the images are resampled to this isotropic spacing.
        mode: Image data interpolation mode.
        padding: Image data extrapolation mode.

    Returns:
        This image batch with given spacing and interpolated image tensor.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("grid_resample() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("grid_resample() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    in_spacing = cat_scalars(in_spacing, *args, num=D, device=data.place)
    out_spacing = cat_scalars(out_spacing, *args, num=D, device=data.place)
    mode = Sampling.from_arg(mode).interpolate_mode(D)
    input_grid = Grid(shape=tuple(data.shape)[2:], spacing=in_spacing)
    output_grid = input_grid.resample(out_spacing)
    if tuple(output_grid.shape) == tuple(input_grid.shape):
        return data
    align_corners = input_grid.align_corners()
    axes = Axes.from_align_corners(align_corners)
    coords = output_grid.coords(align_corners=align_corners, device=data.place)
    coords = grid_transform_points(coords, output_grid, axes, input_grid, axes)
    return grid_sample(
        data, coords, mode=mode, padding=padding, align_corners=align_corners
    )


def grid_reshape(
    data: paddle.Tensor,
    shape: Union[int, Array, Shape],
    *args: int,
    mode: Union[Sampling, str] = Sampling.LINEAR,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Interpolate image with specified spatial image tensor shape.

    Args:
        data: Image batch tensor with shape ``(N, C, ..., X)``.
        shape: Size of spatial image dimensions, where the size of the first grid
            dimension, which is the last ``data`` dimension, must be given last,
            e.g., ``(nz, ny, nx)``.
        mode: Image data interpolation mode.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Interpolated image data of specified ``size`` of spatial dimensions.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("grid_resample() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("grid_reshape() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    shape_ = cat_scalars(shape, *args, num=D)
    return grid_resize(
        data, shape_.flip(axis=0), mode=mode, align_corners=align_corners
    )


def grid_resize(
    data: paddle.Tensor,
    size: Union[int, Array, Shape],
    *args: int,
    mode: Union[Sampling, str] = Sampling.LINEAR,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Interpolate image with specified spatial image tensor shape.

    Args:
        data: Image batch tensor with shape ``(N, C, ..., X)``.
        size: Size of spatial image dimensions, where size of first grid dimension, which
            is the last ``data`` dimension, must be given first, e.g., ``(nx, ny, nz)``.
        mode: Image data interpolation mode.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Interpolated image data of specified ``size`` of spatial dimensions.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("grid_resample() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("grid_resize() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    size_ = cat_scalars(size, *args, num=D)
    mode_ = Sampling.from_arg(mode).interpolate_mode(D)
    grid = Grid(shape=tuple(data.shape)[2:], align_corners=align_corners)
    grid = grid.resize(size_)
    if tuple(grid.shape) == tuple(data.shape)[2:]:
        return data
    if mode_ in ("area", "nearest", "nearest-exact"):
        align_corners = None
    res_shape = (grid.shape[-1],) + grid.shape[:-1]
    return paddle.nn.functional.interpolate(
        x=data, size=res_shape, mode=mode_, align_corners=align_corners
    )


def check_sample_grid(
    func: str, data: paddle.Tensor, grid: paddle.Tensor
) -> paddle.Tensor:
    """Normalize sample grid tensor."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError(f"{func}() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError(f"{func}() 'data' must have shape (N, C, ..., X)")
    if not isinstance(grid, paddle.Tensor):
        raise TypeError(f"{func}() 'grid' must be paddle.Tensor")
    N = tuple(data.shape)[0]
    D = data.ndim - 2
    if not paddle.is_floating_point(x=grid):
        raise TypeError("{func}() 'grid' must have floating point dtype")
    if tuple(grid.shape)[-1] != D:
        raise ValueError(
            f"Last {func}() 'grid' dimension size must match number of spatial image dimensions"
        )
    if grid.ndim == data.ndim - 1:
        grid = grid.unsqueeze(axis=0)
    elif grid.ndim != data.ndim:
        raise ValueError(
            f"{func}() expected 'grid' tensor with {data.ndim - 1} or {data.ndim} dimensions"
        )
    if N == 1 and tuple(grid.shape)[0] > 1:
        N = tuple(grid.shape)[0]
    elif N > 1 and tuple(grid.shape)[0] == 1:
        grid = grid.expand(shape=[N, *tuple(grid.shape)[1:]])
    if tuple(grid.shape)[0] != N:
        msg = f"{func}() expected tensor 'grid' of shape (..., X, {D})"
        msg += f" or (1, ..., X, {D})" if N == 1 else f" or (1|{N}, ..., X, {D})"
        raise ValueError(msg)
    return grid


def grid_sample(
    data: paddle.Tensor,
    grid: paddle.Tensor,
    mode: Optional[Union[Sampling, str]] = None,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Sample data at grid points.

    Args:
        data: Image batch tensor of shape ``(1, C, ..., X)`` or ``(N, C, ..., X)``.
        grid: Grid points tensor of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or ``(N, ..., X, D)``.
            Coordinates of points at which to sample ``data`` must be with respect to ``Axes.CUBE``.
        mode: Image interpolate mode.
        padding: Image extrapolation mode or constant by which to pad input ``data``.
        align_corners: Whether ``grid`` extrema ``(-1, 1)`` refer to the grid boundary
            edges (``align_corners=False``) or corner points (``align_corners=True``).

    Returns:
        Image batch tensor of sampled data with spatial shape determined by ``grid``, and batch
        size ``N`` based on ``data.shape[0]`` or ``grid.shape[0]``, respectively. The data type
        of the returned tensor is ``data.dtype`` if it is a floating point type or ``mode="nearest"``.
        Otherwise, the output data type matches ``grid.dtype``, which must be a floating point type.

    """
    grid = check_sample_grid("grid_sample", data, grid)
    if str(data.place) != str(grid.place):
        raise ValueError(
            "grid_sample() 'data' and 'grid' tensors must be on same device"
        )
    N = tuple(grid.shape)[0]
    D = tuple(grid.shape)[-1]
    if tuple(data.shape)[0] != N:
        data = data.expand(shape=[N, *tuple(data.shape)[1:]])
    mode = Sampling.from_arg(mode).grid_sample_mode(D)
    if isinstance(padding, (PaddingMode, str)):
        padding_mode = PaddingMode.from_arg(padding).grid_sample_mode(D)
        padding_value = 0
    else:
        padding_mode = "zeros"
        padding_value = float(padding or 0)
    out = data.astype(dtype=grid.dtype)
    if padding_value != 0:
        if out.data_ptr() == data.data_ptr():
            out = out.sub(padding_value)
        else:
            out = out.subtract_(y=paddle.to_tensor(padding_value))
    out = paddle.nn.functional.grid_sample(
        x=out,
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if padding_mode == "zeros" and padding_value != 0:
        out = out.add_(y=paddle.to_tensor(padding_value))
    if mode == "nearest" or data.is_floating_point():
        out = out.astype(dtype=data.dtype)
    return out


def grid_sample_mask(
    data: paddle.Tensor,
    grid: paddle.Tensor,
    threshold: float = 0,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Sample binary mask at grid points.

    Args:
        data: Image batch tensor of shape ``(N, 1, ..., X)``.
        grid: Grid points tensor of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or``(N, ..., X, D)``.
            Coordinates of points at which to sample ``data`` must be with respect to ``Axes.CUBE``.
        threshold: Scalar value used to binarize input mask. Values above this threshold are assigned
            value 1, and values below this threshold are assigned value 0.
        align_corners: Whether ``grid`` extrema ``(-1, 1)`` refer to the grid boundary edges (``False``)
            or corner points (``True``), respectively.

    Returns:
        Batch tensor of sampled mask with spatial shape determined by ``grid``, and floating point values
        in the closed interval ``[0, 1]`` as obtained by linear interpolation of the binarized input mask.

    """
    grid = check_sample_grid("grid_sample_mask", data, grid)
    if tuple(data.shape)[1] != 1:
        raise ValueError("grid_sample_mask() 'data' must have single channel")
    mask = data if data.dtype == "bool" else data > threshold
    mask = mask.to(dtype="float32", device=data.place)
    return grid_sample(
        mask,
        grid,
        mode=Sampling.LINEAR,
        padding=PaddingMode.ZEROS,
        align_corners=align_corners,
    )


def rand_sample(
    data: Union[paddle.Tensor, Sequence[paddle.Tensor]],
    num_samples: int,
    mask: Optional[paddle.Tensor] = None,
    replacement: bool = False,
    generator: Optional[paddle.Generator] = None,
) -> Union[paddle.Tensor, Sequence[paddle.Tensor]]:
    """Random sampling of voxels within an image.

    Args:
        data: One or more image batch tensors with shape ``(N, C, ... X)`` to sample
            from at the same random image grid positions, i.e., voxel indices. Note that
            all input tensors must have the same number and size of spatial dimensions.
        num_samples: Number of spatial samples to draw.
        mask: Optional mask to use for spatially weighted sampling.
        replacement: Whether to sample with or without replacement.
        generator: Random number generator to use.

    Returns:
        paddle.Tensor of shape ``(N, C, num_samples)`` with input ``data`` values at randomly
        sampled spatial grid points. When ``data`` is a sequence of tensors, a list
        of tensors with order matching the input data is returned.

    """
    input: Sequence[paddle.Tensor]
    if isinstance(data, paddle.Tensor):
        input = [data]
    elif isinstance(data, Sequence) and all(isinstance(x, paddle.Tensor) for x in data):
        input = data
    else:
        raise TypeError(
            "rand_sample() 'data' must be paddle.Tensor or Sequence[paddle.Tensor]"
        )
    if not input:
        return []
    if any(x.ndim < 3 for x in input):
        raise ValueError(
            "rand_sample() 'data' must be one or more tensors of shape (N, C, ..., X)"
        )
    shape = tuple(input[0].shape)
    if any(tuple(x.shape)[2:] != shape[2:] for x in input):
        raise ValueError(
            "rand_sample() 'data' tensors must have identical spatial shape"
        )
    numel = shape[2:].size
    if not replacement and num_samples > numel:
        raise ValueError(
            "rand_sample() 'num_samples' is greater than number of spatial points"
        )
    input = [x.flatten(start_axis=2) for x in input]
    if mask is None:
        if replacement:
            index = paddle.randint(
                low=0, high=numel, shape=(shape[0], num_samples), dtype="int64"
            )
        else:
            perm = paddle.empty(shape=numel, dtype="int64")
            index = paddle.empty(shape=(shape[0], num_samples), dtype="int64")
            for row in index:
                paddle.assign(paddle.randperm(n=numel), output=perm)
                start_16 = perm.shape[0] + 0 if 0 < 0 else 0
                paddle.assign(
                    paddle.slice(perm, [0], [start_16], [start_16 + num_samples]),
                    output=row,
                )
    else:
        if mask.ndim < 3:
            raise ValueError(
                "rand_sample() 'mask' must be tensor of shape (N, C, ..., X)"
            )
        if tuple(mask.shape)[2:] != shape[2:]:
            raise ValueError(
                "rand_sample() 'mask' has different spatial shape than 'data'"
            )
        if tuple(mask.shape)[1] != 1:
            raise ValueError("rand_sample() 'mask' must be scalar image tensor")
        mask = (
            mask.flatten(start_axis=2).squeeze(axis=1).expand(shape=[shape[0], numel])
        )
        index = multinomial(
            mask, num_samples, replacement=replacement, generator=generator
        )
    index = index.unsqueeze(axis=1).repeat(1, shape[1], 1)
    out = [x.take_along_axis(axis=2, indices=index) for x in input]
    if len(out) == 1 and isinstance(data, paddle.Tensor):
        return out[0]
    return out


def image_slice(data: paddle.Tensor, offset: Optional[int] = None) -> paddle.Tensor:
    """Get slice from image tensor.

    Args:
        data: Image data tensor of shape ``(N, C, ..., Y, X)``.
        offset: Slice offset. If ``None``, use ``Z // 2``. This argument is
            ignored when the input image is 2-dimensional.

    Returns:
        View of image tensor slice with shape ``(N, C, Y, X)``.

    """
    if data.ndim == 4:
        return data
    if data.ndim < 4 or data.ndim > 5:
        raise ValueError("image_slice() 'data' must be 4- or 5-dimensional")
    if offset is None:
        offset = tuple(data.shape)[2] // 2
    start_17 = data.shape[2] + offset if offset < 0 else offset
    return paddle.slice(data, [2], [start_17], [start_17 + 1]).squeeze(axis=2)


def normalize_image(
    data: paddle.Tensor,
    mode: str = "unit",
    min: Optional[float] = None,
    max: Optional[float] = None,
    inplace: bool = False,
) -> paddle.Tensor:
    """Normalize image intensities in [min, max].

    Args:
        data: Input image data.
        mode: How to normalize image values:
            - ``center``: Linearly rescale to [-0.5, 0.5]
            - ``unit``: Linearly rescale to [0, 1].
            - ``zscore``: Linearly  rescale to zero mean and unit variance.
        min: Minimum intensity at which to clamp input.
        max: Maximum intensity at which to clamp input.
        inplace: Whether to modify ``data`` in place.

    Returns:
        Normalized image data.

    Raises:
        TypeError: When ``inplace=True`` and ``data.dtype`` is not a floating point data type.

    """
    if inplace:
        if not data.is_floating_point():
            raise AssertionError(
                "normalize_image() 'data.dtype' must be float when inplace=True"
            )

        def add_fn(data: paddle.Tensor, a: float) -> paddle.Tensor:
            return data.add_(y=paddle.to_tensor(a))

        def sub_fn(data: paddle.Tensor, a: float) -> paddle.Tensor:
            return data.subtract_(y=paddle.to_tensor(a))

        def mul_fn(data: paddle.Tensor, a: float) -> paddle.Tensor:
            return data.multiply_(y=paddle.to_tensor(a))

        def clamp_fn(data: paddle.Tensor, a: float, b: float) -> paddle.Tensor:
            return data.clip_(min=a, max=b)

    else:
        data = data.astype(dtype="float32")

        def add_fn(data: paddle.Tensor, a: float) -> paddle.Tensor:
            return data.add(a)

        def sub_fn(data: paddle.Tensor, a: float) -> paddle.Tensor:
            return data.sub(a)

        def mul_fn(data: paddle.Tensor, a: float) -> paddle.Tensor:
            return data.mul(a)

        def clamp_fn(data: paddle.Tensor, a: float, b: float) -> paddle.Tensor:
            return data.clip(min=a, max=b)

    if mode in ("zscore", "z-score"):
        data = clamp_fn(data, min, max)
        stdev, mean = tuple(
            [
                paddle.std(data, axis=None, unbiased=True, keepdim=False),
                paddle.mean(data, axis=None, keepdim=False),
            ]
        )
        data = sub_fn(data, mean)
        if stdev > 1e-15:
            data = mul_fn(data, 1 / stdev)
    elif mode in ("unit", "center"):
        if min is None:
            min = float(data.min())
        if max is None:
            max = float(data.max())
        dif = max - min
        mul = 1 if abs(dif) < 1e-09 else 1 / dif
        add = -mul * min
        if mode == "center":
            add -= 0.5
        if mul != 1:
            data = mul_fn(data, mul)
        if add != 0:
            data = add_fn(data, add)
        if mode == "center":
            data = clamp_fn(data, -0.5, 0.5)
        else:
            data = clamp_fn(data, 0, 1)
    return data


def rescale(
    data: paddle.Tensor,
    min: Optional[Scalar] = None,
    max: Optional[Scalar] = None,
    data_min: Optional[Scalar] = None,
    data_max: Optional[Scalar] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    """Linearly rescale values to specified output interval.

    Args:
        data: Input tensor.
        min: Minimum value of output tensor. Use ``data_min`` if ``None``.
        max: Maximum value of output tensor. Use ``data_max`` if ``None``.
        data_min: Minimum value of input ``data``. Use ``data.min()`` if ``None``.
        data_max: Maximum value of input ``data``. Use ``data.max()`` if ``None``.
        dtype: Cast rescaled values to specified output data type. If ``None``,
            use ``data.dtype`` if it is a floating point type, otherwise ``paddle.float``.

    Returns:
        paddle.Tensor of same shape as ``data`` with specified ``dtype`` and values in closed interval ``[min, max]``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("rescale() 'data' must be paddle.Tensor")
    if dtype is None:
        dtype = data.dtype
        if not is_float_dtype(dtype):
            dtype = "float32"
    if dtype.is_floating_point:
        interim_dtype = dtype
    elif data.dtype.is_floating_point:
        interim_dtype = data.dtype
    else:
        interim_dtype = "float32"
    data = data.astype(interim_dtype)
    if data_min is None:
        data_min = data.min()
    data_min = float(data_min)
    if data_max is None:
        data_max = data.max()
    data_max = float(data_max)
    min = data_min if min is None else float(min)
    max = data_max if max is None else float(max)
    norm = data_max - data_min
    if norm < 1e-15:
        result = paddle.empty(shape=tuple(data.shape), dtype=data.dtype).fill_(
            value=min
        )
    else:
        scale = (max - min) / norm
        result = min + scale * (data - data_min)
    if not dtype.is_floating_point:
        result = result.round_()
    result = result.clip_(min=min, max=max)
    result = result.astype(dtype)
    return result


def sample_image(
    data: paddle.Tensor,
    coords: paddle.Tensor,
    mode: Optional[Union[Sampling, str]] = None,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Sample images at given points.

    This function samples a batch of images at spatial points. The ``coords`` tensor can be of any shape,
    including ``(N, M, D)``, i.e., a batch of N point sets with cardianality M, and ``(N, ..., X, D)`` ,
    i.e., a (deformed) regular sampling grid (cf. ``grid_sample()``).

    Args:
        data: Batch of images as tensor of shape ``(N, C, ..., X)``. If batch size is one,
            but the batch size of ``coords`` is greater than one, this single image is sampled
            at the different sets of points.
        coords: Normalized coordinates of points given as tensor of shape ``(N, ..., D)``
            or ``(1, ..., D)``. If batch size is one, all images are sampled at the same points.
        align_corners: Whether point coordinates are with respect to ``Axes.CUBE`` (False)
            or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to the
            ``grid_sample()`` function used to sample the images at the given points.

    Returns:
        Sampled image data as tensor of shape ``(N, C, *coords.shape[1:-1])``.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("sample_image() 'data' must be of type paddle.Tensor")
        data = data.as_subclass(paddle.Tensor)
    if data.ndim < 4:
        raise ValueError("sample_image() 'data' must be at least 4-dimensional tensor")
    if not isinstance(coords, paddle.Tensor):
        raise TypeError("sample_image() 'coords' must be of type paddle.Tensor")
    if coords.ndim < 2:
        raise ValueError(
            "sample_image() 'coords' must be at least 2-dimensional tensor"
        )
    G = tuple(data.shape)[0]
    N = tuple(coords.shape)[0] if G == 1 else G
    D = data.ndim - 2
    if tuple(coords.shape)[0] not in (1, N):
        raise ValueError(f"sample_image() 'coords' must be batch of length 1 or {N}")
    if tuple(coords.shape)[-1] != D:
        raise ValueError(
            f"sample_image() 'coords' must be tensor of {D}-dimensional points"
        )
    x = coords.expand(shape=(N,) + tuple(coords.shape)[1:])
    data = data.expand(shape=(N,) + tuple(data.shape)[1:])
    grid = x.reshape((N,) + (1,) * (data.ndim - 3) + (-1, D))
    data = grid_sample(
        data, grid, mode=mode, padding=padding, align_corners=align_corners
    )
    return data.reshape(tuple(data.shape)[:2] + tuple(coords.shape)[1:-1])


def spatial_derivatives(
    data: paddle.Tensor,
    mode: str = "central",
    which: Optional[Union[str, Sequence[str]]] = None,
    order: Optional[int] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
) -> Dict[str, paddle.Tensor]:
    """Calculate spatial image derivatives.

    Args:
        data: Image data tensor of shape ``(N, C, ..., X)``.
        mode: Method to use for approximating spatial image derivative.
            If ``forward``, ``backward``, or ``central``, the respective finite difference
            scheme is used to approximate the image derivative, optionally after smoothing
            the input image with a Gaussian kernel. If ``gaussian``, the image derivative
            is computed by convolving the image with a derivative of Gaussian kernel.
            If ``None``, a central difference scheme is used by default.
        which: String codes of spatial deriviatives to compute. See ``SpatialDerivativeKeys``.
        order: Order of spatial derivative. If zero, the input ``data`` is returned.
        sigma: Standard deviation of Gaussian kernel in grid units. If ``None`` or zero,
            no Gaussian smoothing is used for calculation of finite differences, and a
            default standard deviation of 0.4 is used when ``mode="gaussian"``.
        spacing: Physical spacing between image grid points, e.g., ``(sx, sy, sz)``.
            When a scalar is given, the same spacing is used for each image and spatial dimension.
            If a sequence is given, it must be of length equal to the number of spatial dimensions ``D``,
            and specify a separate spacing for each dimension in the order ``(x, ...)``. In order to
            specify a different spacing for each image in the input ``data`` batch, a 2-dimensional
            tensor must be given, where the size of the first dimension is equal to ``N``. The second
            dimension can have either size 1 for an isotropic spacing, or ``D`` in case of an
            anisotropic grid spacing.

    Returns:
        Mapping from spatial derivative keys to corresponding tensors of the respective spatial
        image derivatives of shape ``(N, C, ..., X)``. The keys are sequences of letters identifying
        the spatial dimensions along which a derivative was taken (cf. ``SpatialDerivativeKeys``).

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("spatial_derivatives() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("spatial_derivatives() 'data' must have shape (N, C, ..., X)")
    N = tuple(data.shape)[0]
    D = data.ndim - 2
    if spacing is None:
        spacing = paddle.ones(shape=(N, D), dtype="float32")
    else:
        spacing = as_tensor(
            spacing, dtype="float32", device=str("cpu").replace("cuda", "gpu")
        )
        spacing = paddle.atleast_1d(spacing)
        if spacing.ndim == 1:
            spacing = spacing.unsqueeze(axis=0)
        if (
            spacing.ndim != 2
            or tuple(spacing.shape)[0] not in (1, N)
            or tuple(spacing.shape)[1] not in (1, D)
        ):
            raise ValueError(
                f"spatial_derivatives() 'spacing' must be scalar, {D}-dimensional vector, or 2-dimensional array of shape (1, {D}), ({N}, 1), or ({N}, {D})"
            )
        spacing = spacing.expand(shape=[N, D])
    if isinstance(which, str):
        which = (which,)
    if which is None:
        if order is None:
            order = 1
        which = SpatialDerivativeKeys.all(ndim=D, order=order)
    elif order is not None:
        which = [arg for arg in which if len(arg) == order]
    unique_keys = SpatialDerivativeKeys.unique(which)
    max_order = SpatialDerivativeKeys.max_order(which)
    derivs = {}
    if not which:
        return derivs
    if not data.is_floating_point():
        data = data.astype("float32")
    if mode is None:
        mode = "central"
    if mode in ("forward", "backward", "central", "prewitt", "sobel"):
        if sigma and sigma > 0:
            blur = gaussian1d(sigma, dtype="float32", device=data.place)
            data = conv(data, blur, padding=PaddingMode.ZEROS)
        if mode in ("prewitt", "sobel"):
            avg_kernel = paddle.to_tensor(
                data=[1, 1 if mode == "prewitt" else 2, 1], dtype=data.dtype
            )
            avg_kernel /= avg_kernel.sum()
            avg_kernel = avg_kernel.to(data.place)
            fd_mode = "central"
        else:
            avg_kernel = None
            fd_mode = mode
        for i in range(max_order):
            for code in unique_keys:
                key = code[: i + 1]
                if i < len(code) and key not in derivs:
                    sdim = SpatialDim.from_arg(code[i])
                    result = data if i == 0 else derivs[code[:i]]
                    if avg_kernel is not None:
                        for d in (d for d in range(D) if d != sdim):
                            dim = SpatialDim(d).tensor_dim(result.ndim)
                            result = conv1d(
                                result,
                                avg_kernel,
                                dim=dim,
                                padding=len(avg_kernel) // 2,
                            )
                    fd_spacing = spacing[:, (sdim)]
                    result = finite_differences(
                        result, sdim, mode=fd_mode, spacing=fd_spacing
                    )
                    derivs[key] = result
        derivs = {key: derivs[SpatialDerivativeKeys.sorted(key)] for key in which}
    elif mode == "gaussian":
        if not sigma:
            sigma = 0.4
        kernel_0 = gaussian1d(sigma, normalize=False, dtype="float32")
        kernel_1 = gaussian1d_I(sigma, normalize=False, dtype="float32")
        norm = kernel_0.sum()
        kernel_0 = kernel_0.divide_(y=paddle.to_tensor(norm)).to(data.place)
        kernel_1 = kernel_1.divide_(y=paddle.to_tensor(norm)).to(data.place)
        for i in range(max_order):
            for code in unique_keys:
                key = code[: i + 1]
                if i < len(code) and key not in derivs:
                    sdim = SpatialDim.from_arg(code[i])
                    result = data if i == 0 else derivs[code[:i]]
                    for d in range(D):
                        dim = SpatialDim(d).tensor_dim(result.ndim)
                        kernel = kernel_1 if sdim == d else kernel_0
                        result = conv1d(
                            result, kernel, dim=dim, padding=len(kernel) // 2
                        )
                    derivs[key] = result
        derivs = {key: derivs[SpatialDerivativeKeys.sorted(key)] for key in which}
    else:
        raise ValueError(
            "spatial_derivatives() 'mode' must be 'forward', 'backward', 'central', or 'gaussian'"
        )
    return derivs


def finite_differences(
    data: paddle.Tensor,
    sdim: SpatialDimArg,
    mode: str = "central",
    order: int = 1,
    dilation: int = 1,
    spacing: Union[float, Sequence[float]] = 1,
) -> paddle.Tensor:
    """Calculate spatial image derivative using finite differences.

    Args:
        data: Image data tensor of shape ``(N, C, ..., X)``.
        sdim: Spatial dimension along which to compute spatial derivative.
        mode: Finite differences to use for approximating spatial derivative.
        order: Order of spatial derivative. If zero, the input ``data`` is returned.
        dilation: Step size for finite differences.
        spacing: Physical spacing between image grid points along dimension ``sdim``.
            When a scalar is given, the same spacing is used for all images in the
            input ``data`` batch. Otherwise, a separate spacing must be specified for
            each image as sequence of float values.

    Returns:
        paddle.Tensor of spatial derivative with respect to specified spatial dimension.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("finite_differences() 'data' must be paddle.Tensor")
    if data.ndim < 4:
        raise ValueError("finite_differences() 'data' must have shape (N, C, ..., X)")
    if not isinstance(order, int):
        raise TypeError("finite_differences() 'order' must be int")
    if order < 0:
        raise ValueError("finite_differences() 'order' must be non-negative")
    if not isinstance(dilation, int):
        raise TypeError("finite_differences() 'dilation' must be int")
    if dilation < 1:
        raise ValueError("finite_differences() 'dilation' must be positive")
    spatial_dim = SpatialDim.from_arg(sdim)
    dim = spatial_dim.tensor_dim(data.ndim)
    if mode not in ("forward", "backward", "central"):
        raise ValueError(
            "finite_differences() 'mode' must be 'forward', 'backward', or 'central'"
        )
    if order == 0:
        return data
    if order == 1:
        if mode == "forward":
            i = slice(0, tuple(data.shape)[dim] - dilation, 1)
            j = slice(dilation, tuple(data.shape)[dim], 1)
            p = 0, dilation
        elif mode == "backward":
            i = slice(dilation, tuple(data.shape)[dim], 1)
            j = slice(0, tuple(data.shape)[dim] - dilation, 1)
            p = dilation, 0
        else:
            i = slice(0, tuple(data.shape)[dim] - 2 * dilation, 1)
            j = slice(2 * dilation, tuple(data.shape)[dim], 1)
            p = dilation, dilation
    else:
        raise NotImplementedError(f"finite_differences(..., order={order})")
    i = tuple(
        i if d == dim else slice(0, n, 1) for d, n in enumerate(tuple(data.shape))
    )
    j = tuple(
        j if d == dim else slice(0, n, 1) for d, n in enumerate(tuple(data.shape))
    )
    N = tuple(data.shape)[0]
    data = data.astype(dtype="float32")
    deriv = data[j].sub(data[i])
    denom: paddle.Tensor = paddle.atleast_1d(
        as_tensor(spacing, dtype=data.dtype, device=data.place)
    )
    if denom.ndim > 1 or tuple(denom.shape)[0] not in (1, N):
        raise ValueError(
            f"finite_differences() 'spacing' must be scalar or sequence of length {N}"
        )
    denom = denom.mul((2 if mode == "central" else 1) * dilation)
    denom = denom.reshape((tuple(denom.shape)[0],) + (1,) * (deriv.ndim - 1))
    deriv = deriv.div(denom)
    pad = [(p if d == spatial_dim else (0, 0)) for d in range(data.ndim - 2)]
    pad = [n for v in pad for n in v]
    return paddle_aux._FUNCTIONAL_PAD(pad=pad, mode="constant", value=0, x=deriv)


def _image_size(
    fn_name: str,
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    ndim: Optional[int] = None,
) -> list:
    """Parse 'size' and/or 'shape' argument of image creation function."""
    if size is None and shape is None:
        raise AssertionError(f"{fn_name}() 'size' or 'shape' required")
    if isinstance(size, Grid):
        size = tuple(size.shape)
    if size is not None and shape is not None and size != tuple(reversed(shape)):
        raise AssertionError(f"{fn_name}() mismatch between 'size' and 'shape'")
    if size is None:
        if ndim and len(shape) != ndim:
            raise ValueError(f"{fn_name}() 'shape' must be tuple of length {ndim}")
        size = tuple(reversed(shape))
    elif isinstance(size, int):
        size = size, size
    elif ndim and len(size) != 2:
        raise ValueError(f"{fn_name}() 'size' must be tuple of length {ndim}")
    return tuple(size)


def circle_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    center: Optional[Sequence[int]] = None,
    radius: Optional[float] = None,
    sigma: float = 0,
    x_max: Optional[Union[float, Sequence[float]]] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Synthetic image of a circle.

    Args:
        size: Spatial size in the order ``(X, Y)``.
        shape: Spatial size in the order ``(Y, X)``.
        num: Number ``N`` of images in batch.
        center: Coordinates of center pixel in the order ``(x, y)``.
        radius: Radius of circle in pixel units.
        sigma: Standard deviation of isotropic Gaussian blurring kernel in pixel units.
        x_max: Maximum ``x`` pixel index at which to clamp image to zero.
            This can be used to create partial circles such as a half circle.
        dtype: Data type of output tensor. Use ``paddle.uint8`` if ``None``.
            When the output data type is a floating point type, the output tensor
            values are in the interval ``[0, 1]``. Otherwise, the output values are
            in the interval ``[0, 255]``.
        device: Device on which to create image tensor.

    Returns:
        Image tensor of shape ``(N, 1, Y, X)``.

    """
    size = _image_size("circle_image", size, shape, ndim=2)
    if center is None:
        center = tuple((n - 1) / 2 for n in size)
    center = tuple(float(x) for x in center)
    grid = Grid(size=size)
    if radius is None:
        radius = max(0, min(center) - 1 - math.ceil(2 * sigma))
    _dtype = "float32"
    _device = str("cpu").replace("cuda", "gpu")
    c = as_tensor(center, dtype=_dtype, device=_device)
    x = grid.coords(normalize=False, dtype=_dtype, device=_device)
    x = x.reshape(num or 1, 1, *tuple(x.shape)) - c
    data = paddle.linalg.norm(x=x, axis=-1) <= radius
    if x_max:
        x_threshold = as_tensor(x_max, dtype=_dtype, device=_device)
        if x_threshold.ndim == 0:
            data &= x[..., 0] <= x_threshold
        else:
            data &= (x <= x_threshold).astype("bool").all(axis=-1)
    data = data.astype(_dtype)
    if sigma > 0:
        kernel = gaussian1d(sigma, dtype=data.dtype, device=_device)
        data = conv(data, kernel / kernel.sum())
    if dtype is None:
        dtype = "uint8"
    if not dtype.is_floating_point:
        data = 255 * data / data.max()
    return data.to(dtype=dtype, device=device)


def cshape_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    center: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    width: Optional[float] = None,
    sigma: float = 0,
    x_max: Union[float, Sequence[float]] = 5,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Synthetic C-shaped image.

    Args:
        size: Spatial size in the order ``(X, Y)``.
        shape: Spatial size in the order ``(Y, X)``.
        num: Number ``N`` of images in batch.
        center: Coordinates of center pixel in the order ``(y, x)``.
        radius: Radius of circle in pixel units.
        sigma: Standard deviation of isotropic Gaussian blurring kernel in pixel units.
        x_max: Maximum ``x`` pixel index at which to clamp image to zero.
            This can be used to create partial circles such as a half circle.
        dtype: Data type of output tensor. Use ``paddle.uint8`` if ``None``.
            When the output data type is a floating point type, the output tensor
            values are in the interval ``[0, 1]``. Otherwise, the output values are
            in the interval ``[0, 255]``.
        device: Device on which to create image tensor.

    Returns:
        Image tensor of shape ``(N, 1, Y, X)``.

    """
    size = _image_size("cshape_image", size, shape, ndim=2)
    if dtype is None:
        dtype = "uint8"
    if radius is None:
        center = tuple(float(x) for x in center)
        radius = max(0, min(center) - 1 - math.ceil(2 * sigma))
    if width is None:
        width = radius // 2
    outer = circle_image(size, center=center, radius=radius, x_max=x_max, sigma=0)
    inner = circle_image(size, center=center, radius=radius - width, sigma=0)
    image = (outer - inner).astype("float32")
    if sigma > 0:
        kernel = gaussian1d(sigma, dtype="float32", device=image.place)
        image = conv(image, kernel)
        if dtype.is_floating_point:
            image /= image.max()
            image.clip_(min=0, max=1)
        else:
            image *= 255 / image.max()
            image.clip_(min=0, max=255)
    if num and num > 1:
        image = image.expand(shape=(1,) + tuple(image.shape)[1:])
    return image.to(dtype=dtype, device=device)


def empty_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    channels: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Create new batch of uninitalized image data.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch.
        channels: Number of channels per image.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Uninitialized image batch tensor.

    """
    size = _image_size("empty_image", size, shape)
    shape = (num or 1, channels or 1) + tuple(reversed(size))
    return paddle.empty(shape=shape, dtype=dtype)


def grid_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    stride: Optional[Union[int, Sequence[int]]] = None,
    inverted: bool = False,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Create batch of regularly spaced grid images.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch. When ``shape`` is not a ``Grid``, must
            match the size of the first dimension in ``shape`` if not ``None``.
        stride: Spacing between grid lines. To draw in-plane grid lines on a
            D-dimensional image where ``D>2``, specify a sequence of two stride
            values, where the first stride applies to the last tensor dimension,
            which corresponds to the first spatial grid dimension.
        inverted: Whether to draw grid lines in black (0) over white (1) background.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image tensor of shape ``(N, 1, ..., X)``. The default number of channels is 1.

    """
    size = _image_size("grid_image", size, shape)
    data = empty_image(size, num=1, channels=1, dtype=dtype, device=device)
    data.fill_(value=1 if inverted else 0)
    if stride is None:
        stride = 4
    if isinstance(stride, int):
        stride = (stride,) * (data.ndim - 2)
    if len(stride) > data.ndim - 2:
        raise ValueError(
            "grid_image() 'stride' length must not be greater than number of spatial dimensions"
        )
    start = data.ndim - len(stride)
    for dim, step in zip(range(start, data.ndim), reversed(stride)):
        n = tuple(data.shape)[dim]
        index = paddle.arange(start=n % step // 2, end=n, step=step, dtype="int64")
        data.index_fill_(axis=dim, index=index, value=0 if inverted else 1)
    return data.expand(shape=[num or 1, *tuple(data.shape)[1:]])


def ones_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    channels: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Create new batch of image data filled with ones.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch.
        channels: Number of channels per image.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image batch tensor filled with ones.

    """
    size = _image_size("ones_image", size, shape)
    data = empty_image(size, num=num, channels=channels, dtype=dtype, device=device)
    return data.fill_(value=1)


def zeros_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    channels: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Create new batch of image data filled with zeros.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch.
        channels: Number of channels per image.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image batch tensor filled with zeros.

    """
    size = _image_size("zeros_image", size, shape)
    data = empty_image(size, num=num, channels=channels, dtype=dtype, device=device)
    return data.fill_(value=0)
