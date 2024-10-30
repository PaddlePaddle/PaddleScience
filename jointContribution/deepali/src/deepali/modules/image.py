r"""Convoluational modules with fixed parameters."""

import math
from numbers import Number
from typing import Optional
from typing import Union

import deepali.utils.paddle_aux  # noqa
import paddle
from deepali.core import functional as U
from deepali.core.enum import PaddingMode
from deepali.core.kernels import gaussian1d
from deepali.core.typing import ScalarOrTuple
from paddle import Tensor
from pkg_resources import parse_version


class FilterImage(paddle.nn.Layer):
    r"""Convoles an image with a predefined filter kernel."""

    def __init__(
        self, kernel: Optional[paddle.Tensor], padding: Optional[Union[PaddingMode, str]] = None
    ):
        r"""Initialize parameters.

        Args:
            kernel: Predefined convolution kernel.
            padding: Image extrapolation mode.

        """
        super().__init__()
        self.padding = PaddingMode.CONSTANT if padding is None else PaddingMode.from_arg(padding)
        self.register_buffer(name="kernel", tensor=kernel)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        r"""Convolve input image with predefined filter kernel."""
        kernel: Optional[paddle.Tensor] = self.kernel
        if self.kernel is None or kernel.size < 2:
            return x
        return U.conv(x, kernel, padding=self.padding)

    def extra_repr(self) -> str:
        return f"padding={repr(self.padding.value)}"


class BlurImage(FilterImage):
    r"""Blurs an image by a predefined Gaussian low-pass filter."""

    def __init__(self, sigma: float, padding: Optional[Union[PaddingMode, str]] = None):
        r"""Initialize parameters.

        Args:
            sigma: Standard deviation of isotropic Gaussian kernel in grid units (pixel, voxel).
            padding: Image extrapolation mode.

        """
        sigma = float(sigma)
        kernel = gaussian1d(sigma) if sigma > 0 else None
        super().__init__(kernel=kernel, padding=padding)
        self.sigma = sigma

    def extra_repr(self) -> str:
        return f"sigma={repr(self.sigma)}, " + super().extra_repr()


class GaussianConv(paddle.nn.Layer):
    r"""Blurs an image by a predefined Gaussian low-pass filter."""

    def __init__(
        self, channels: int, kernel_size: ScalarOrTuple[int], sigma: float, dim: int = 3
    ) -> None:
        r"""Initialize Gaussian convolution kernel.

        Args:
            channels (int, sequence): Number of channels of the input and output tensors.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigma (float, sequence): Standard deviation of the gaussian kernel.
            dim (int, optional): The number of dimensions of the data.

        """
        if dim < 2 or dim > 3:
            raise ValueError(f"Only 2 and 3 dimensions are supported, got: {dim}")
        super().__init__()
        if isinstance(kernel_size, Number):
            kernel_size = (kernel_size,) * dim
        if isinstance(sigma, Number):
            sigma = (sigma,) * dim
        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = paddle.to_tensor(data=1, dtype="float32", place="cpu")
        mgrids = [paddle.arange(dtype="float32", end=n) for n in kernel_size]
        if parse_version(paddle.__version__) < parse_version("1.10"):
            mgrids = paddle.meshgrid(mgrids)
        else:
            mgrids = paddle.meshgrid(mgrids)
        norm = math.sqrt(2 * math.pi)
        for size, std, mgrid in zip(kernel_size, sigma, mgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * norm) * paddle.exp(x=-(((mgrid - mean) / std) ** 2) / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel.divide_(y=paddle.to_tensor(kernel.sum()))
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.tile(repeat_times=[channels, *((1,) * (kernel.dim() - 1))])
        self.register_buffer(name="kernel", tensor=kernel, persistable=True)
        self.groups = channels
        # Padding for output to equal input dimensions
        self.pad = (kernel_size[0] // 2,) * (2 * dim)
        self.conv = paddle.nn.functional.conv2d if dim == 2 else paddle.nn.functional.conv3d

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        r"""Convolve input with Gaussian kernel."""
        # Use mode='replicate' to ensure image boundaries are not degraded in output for background channel
        kernel: Tensor = self.kernel
        data = paddle.nn.functional.pad(
            x=x, pad=self.pad, mode="replicate", pad_from_left_axis=False
        )
        data = self.conv(data, kernel, groups=self.groups)
        return data
