import math
from numbers import Number
from typing import Optional
from typing import Union

import paddle
from pkg_resources import parse_version

from ..core import functional as U
from ..core.enum import PaddingMode
from ..core.kernels import gaussian1d
from ..core.types import ScalarOrTuple
from ..utils import paddle_aux


class FilterImage(paddle.nn.Layer):
    """Convoles an image with a predefined filter kernel."""

    def __init__(
        self,
        kernel: Optional[paddle.Tensor],
        padding: Optional[Union[PaddingMode, str]] = None,
    ):
        """Initialize parameters.

        Args:
            kernel: Predefined convolution kernel.
            padding: Image extrapolation mode.

        """
        super().__init__()
        self.padding = (
            PaddingMode.CONSTANT if padding is None else PaddingMode.from_arg(padding)
        )
        self.register_buffer(name="kernel", tensor=kernel)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Convolve input image with predefined filter kernel."""
        kernel: Optional[paddle.Tensor] = self.kernel
        if self.kernel is None or kernel.size < 2:
            return x
        return U.conv(x, kernel, padding=self.padding)

    def extra_repr(self) -> str:
        return f"padding={repr(self.padding.value)}"


class BlurImage(FilterImage):
    """Blurs an image by a predefined Gaussian low-pass filter."""

    def __init__(self, sigma: float, padding: Optional[Union[PaddingMode, str]] = None):
        """Initialize parameters.

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
    """Blurs an image by a predefined Gaussian low-pass filter."""

    def __init__(
        self, channels: int, kernel_size: ScalarOrTuple[int], sigma: float, dim: int = 3
    ) -> None:
        """Initialize Gaussian convolution kernel.

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
        kernel = paddle.to_tensor(data=1, dtype="float32", place="cpu")
        mgrids = [paddle.arange(dtype="float32", end=n) for n in kernel_size]
        if parse_version(paddle.__version__) < parse_version("1.10"):
            mgrids = paddle.meshgrid(mgrids)
        else:
            mgrids = paddle.meshgrid(mgrids)
        norm = math.sqrt(2 * math.pi)
        for size, std, mgrid in zip(kernel_size, sigma, mgrids):
            mean = (size - 1) / 2
            kernel *= (
                1 / (std * norm) * paddle.exp(x=-(((mgrid - mean) / std) ** 2) / 2)
            )
        kernel = kernel.divide_(y=paddle.to_tensor(kernel.sum()))
        kernel = kernel.view(1, 1, *tuple(kernel.shape))
        kernel = kernel.repeat(channels, *((1,) * (kernel.dim() - 1)))
        self.register_buffer(name="kernel", tensor=kernel, persistable=True)
        self.groups = channels
        self.pad = (kernel_size[0] // 2,) * (2 * dim)
        self.conv = (
            paddle.nn.functional.conv2d if dim == 2 else paddle.nn.functional.conv3d
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Convolve input with Gaussian kernel."""
        kernel: paddle.Tensor = self.kernel
        data = paddle_aux._FUNCTIONAL_PAD(pad=self.pad, mode="replicate", x=x)
        data = self.conv(data, kernel, groups=self.groups)
        return data
