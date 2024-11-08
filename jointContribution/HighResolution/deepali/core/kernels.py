import math
from typing import Optional
from typing import Sequence
from typing import Union

import paddle

from .linalg import tensordot
from .tensor import as_tensor
from .tensor import cat_scalars
from .types import Array
from .types import Device
from .types import Scalar


def bspline1d(stride: int, order: int = 4) -> paddle.Tensor:
    """B-spline kernel of given order for specified control point spacing.

    Implementation adopted from AirLab:
    https://github.com/airlab-unibas/airlab/blob/80c9d487c012892c395d63c6d937a67303c321d1/airlab/utils/kernelFunction.py#L218

    This function computes the kernel recursively by convolving with a box filter (cf. Cox-de Boor's recursion formula).
    The resulting kernel differs from the analytic B-spline function. This may be due to the box filter having extend
    to the borders of the pixels, where it should drop to zero at pixel centers rather.

    The exact B-spline kernel of order 4 is computed by ``cubic_bspline1d()``.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        order: Order of B-spline kernel, where the degree of the spline polynomials is order minus 1.

    Returns:
        B-spline convolution kernel.

    """
    kernel = kernel_ones = paddle.ones(shape=[1, 1, stride], dtype="float32")
    for _ in range(1, order + 1):
        kernel = (
            paddle.nn.functional.conv1d(
                x=kernel, weight=kernel_ones, padding=stride - 1
            )
            / stride
        )
    return kernel.reshape(-1)


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    """Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    if t >= 2:
        return 0
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t**2
        return -((t - 2) ** 3) / 6
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline(
    stride: Union[int, Sequence[int]],
    *args: int,
    derivative: int = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
):
    """Get n-dimensional cubic B-spline kernel.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    stride_ = cat_scalars(
        stride,
        *args,
        derivative=derivative,
        dtype="int32",
        device=str("cpu").replace("cuda", "gpu"),
    ).tolist()
    D = len(stride_)
    if D == 1:
        return cubic_bspline1d(
            stride_, derivative=derivative, dtype=dtype, device=device
        )
    if D == 2:
        return cubic_bspline2d(
            stride_, derivative=derivative, dtype=dtype, device=device
        )
    if D == 3:
        return cubic_bspline3d(
            stride_, derivative=derivative, dtype=dtype, device=device
        )
    raise NotImplementedError(f"cubic_bspline() {D}-dimensional kernel")


def cubic_bspline1d(
    stride: Union[int, Sequence[int]],
    derivative: int = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = "float32"
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = paddle.ones(shape=4 * stride - 1, dtype="float32")
    radius = tuple(kernel.shape)[0] // 2
    for i in range(tuple(kernel.shape)[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.place
    return kernel.to(device)


def cubic_bspline2d(
    stride: Union[int, Sequence[int]],
    *args: int,
    derivative: int = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = "float32"
    stride_ = cat_scalars(
        stride, *args, num=2, dtype="int32", device=str("cpu").replace("cuda", "gpu")
    )
    kernel = paddle.ones(shape=(4 * stride_ - 1).tolist(), dtype=dtype)
    radius = [(n // 2) for n in tuple(kernel.shape)]
    for j in range(tuple(kernel.shape)[1]):
        w_j = cubic_bspline_value((j - radius[1]) / stride[1], derivative=derivative)
        for i in range(tuple(kernel.shape)[0]):
            w_i = cubic_bspline_value(
                (i - radius[0]) / stride[0], derivative=derivative
            )
            kernel[j, i] = w_i * w_j
    if device is None:
        device = kernel.place
    return kernel.to(device)


def cubic_bspline3d(
    stride: Union[int, Sequence[int]],
    *args: int,
    derivative: int = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = "float32"
    stride_ = cat_scalars(
        stride, *args, num=3, dtype="int32", device=str("cpu").replace("cuda", "gpu")
    )
    kernel = paddle.ones(shape=(4 * stride_ - 1).tolist(), dtype="float32")
    radius = [(n // 2) for n in tuple(kernel.shape)]
    for k in range(tuple(kernel.shape)[2]):
        w_k = cubic_bspline_value((k - radius[2]) / stride[2], derivative=derivative)
        for j in range(tuple(kernel.shape)[1]):
            w_j = cubic_bspline_value(
                (j - radius[1]) / stride[1], derivative=derivative
            )
            for i in range(tuple(kernel.shape)[0]):
                w_i = cubic_bspline_value(
                    (i - radius[0]) / stride[0], derivative=derivative
                )
                kernel[k, j, i] = w_i * w_j * w_k
    if device is None:
        device = kernel.place
    return kernel.to(device)


def gaussian_kernel_radius(
    sigma: Union[Scalar, Array], factor: Scalar = 3
) -> paddle.Tensor:
    """Radius of truncated Gaussian kernel.

    Args:
        sigma: Standard deviation in grid units.
        factor: Number of standard deviations at which to truncate.

    Returns:
        Radius of truncated Gaussian kernel in grid units.

    """
    sigma = as_tensor(sigma, dtype="float32", device="cpu")
    is_scalar = sigma.ndim == 0
    if is_scalar:
        sigma = sigma.unsqueeze(axis=0)
    if sigma.ndim != 1:
        raise ValueError("gaussian() 'sigma' must be scalar or sequence")
    if tuple(sigma.shape)[0] == 0:
        raise ValueError("gaussian() 'sigma' must be scalar or non-empty sequence")
    if sigma.less_than(y=paddle.to_tensor(0.0)).astype("bool").any():
        raise ValueError("Gaussian standard deviation must be non-negative")
    factor = as_tensor(factor, dtype=sigma.dtype, device=sigma.place)
    radius = sigma.mul(factor).floor().astype("int64")
    if is_scalar:
        radius = radius
    return radius


def gaussian(
    sigma: Union[Scalar, Array],
    *args: Scalar,
    normalize: bool = True,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
):
    """Get n-dimensional Gaussian kernel."""
    sigma_ = cat_scalars(
        sigma, *args, dtype=dtype, device=str("cpu").replace("cuda", "gpu")
    )
    if not paddle.is_floating_point(x=sigma_):
        if dtype is not None:
            raise TypeError("Gaussian kernel dtype must be floating point type")
        sigma_ = sigma_.astype("float32")
    if sigma_.ndim == 0:
        sigma_ = sigma_.unsqueeze(axis=0)
    if sigma_.ndim != 1:
        raise ValueError("gaussian() 'sigma' must be scalar or sequence")
    if tuple(sigma_.shape)[0] == 0:
        raise ValueError("gaussian() 'sigma' must be scalar or non-empty sequence")
    kernel = gaussian1d(sigma_[0], normalize=False, dtype="float64")
    for std in sigma_[1:]:
        other = gaussian1d(std, normalize=False, dtype="float64")
        kernel = tensordot(kernel, other, dims=0)
    if normalize:
        kernel /= kernel.sum()
    return kernel.to(dtype=sigma_.dtype, device=device)


def gaussian1d(
    sigma: Scalar,
    radius: Optional[Union[int, paddle.Tensor]] = None,
    scale: Optional[Scalar] = None,
    normalize: bool = True,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Get 1-dimensional Gaussian kernel."""
    sigma = as_tensor(sigma, device="cpu")
    if sigma.ndim != 0:
        raise ValueError("gaussian1d() 'sigma' must be scalar")
    if sigma.less_than(y=paddle.to_tensor(0.0)):
        raise ValueError("Gaussian standard deviation must be non-negative")
    if dtype is not None and dtype not in ("float16", "float32", "float64"):
        raise TypeError("Gaussian kernel dtype must be floating point type")
    if radius is None:
        radius = gaussian_kernel_radius(sigma)
    radius = int(radius)
    if radius > 0:
        size = 2 * radius + 1
        x = paddle.linspace(start=-radius, stop=radius, num=size, dtype=dtype)
        sigma = sigma.to(dtype=dtype, device=device)
        kernel = paddle.exp(x=-0.5 * (x / sigma) ** 2)
        if scale is None:
            scale = 1 / sigma.mul(math.sqrt(2 * math.pi))
        else:
            scale = as_tensor(scale, dtype=dtype, device=device)
        kernel *= scale
        if normalize:
            kernel /= kernel.sum()
    else:
        if scale is None:
            scale = 1
        kernel = as_tensor(scale, dtype=dtype, device=device)
    return kernel


def gaussian1d_I(
    sigma: Scalar,
    normalize: bool = True,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Get 1st order derivative of 1-dimensional Gaussian kernel."""
    if paddle.is_tensor(x=sigma):
        sigma_ = as_tensor(sigma)
        if sigma_.ndim != 0:
            raise ValueError("gaussian1d() 'sigma' must be scalar")
        sigma = sigma_.item()
    if sigma < 0:
        raise ValueError("Gaussian standard deviation must be non-negative")
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("Gaussian kernel dtype must be floating point type")
    radius = int(gaussian_kernel_radius(sigma).item())
    if radius > 0:
        size = 2 * radius + 1
        x = paddle.linspace(start=-radius, stop=radius, num=size, dtype=dtype)
        norm = paddle.to_tensor(
            data=1 / (sigma * math.sqrt(2 * math.pi)), dtype=dtype, place=device
        )
        var = sigma**2
        kernel = norm * paddle.exp(x=-0.5 * x**2 / var) * (x / var)
        if normalize:
            kernel /= (norm * paddle.exp(x=-0.5 * x**2 / var)).sum()
    else:
        kernel = paddle.to_tensor(data=[1], dtype=dtype, place=device)
    return kernel


def gaussian2d(sigma: Union[Scalar, Array], *args: Scalar, **kwargs) -> paddle.Tensor:
    """Get 2-dimensional Gaussian kernel."""
    sigma = cat_scalars(sigma, *args, num=2, device=str("cpu").replace("cuda", "gpu"))
    return gaussian(sigma, **kwargs)


def gaussian3d(sigma: Union[Scalar, Array], *args: Scalar, **kwargs) -> paddle.Tensor:
    """Get 3-dimensional Gaussian kernel."""
    sigma = cat_scalars(sigma, *args, num=3, device=str("cpu").replace("cuda", "gpu"))
    return gaussian(sigma, **kwargs)
