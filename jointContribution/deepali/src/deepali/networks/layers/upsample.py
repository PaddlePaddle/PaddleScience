# Upsample and SubpixelUpsample layers are adapted from MONAI project
# (https://github.com/Project-MONAI/MONAI/blob/db8f7877da06a9b3710071c626c0488676716be1/monai/networks/blocks/upsample.py)
#
# Copyright 2020 - 2021 MONAI Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
from deepali.core.enum import PaddingMode
from deepali.core.enum import Sampling
from deepali.core.grid import ALIGN_CORNERS
from deepali.core.nnutils import upsample_output_padding
from deepali.core.nnutils import upsample_padding
from deepali.core.typing import ScalarOrTuple
from deepali.modules import Pad
from deepali.utils import paddle_aux
from paddle import Tensor

from .conv import convolution
from .pool import pooling

__all__ = "Upsample", "UpsampleMode", "SubpixelUpsample"


class UpsampleMode(Enum):
    DECONV = "deconv"
    INTERPOLATE = "interpolate"
    PIXELSHUFFLE = "pixelshuffle"


class Upsample(paddle.nn.Sequential):
    r"""Upsamples data by `scale_factor`.

    This module is adapted from the MONAI project. Supported modes are:

        - "deconv": uses a transposed convolution.
        - "interpolate": uses :py:class:`paddle.nn.Upsample`.
        - "pixelshuffle": uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    This module can optionally apply a convolution prior to upsampling interpolation
    (e.g., used to map the number of features from `in_channels` to `out_channels`).

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        scale_factor: Union[Sequence[Union[int, float]], Union[int, float]] = 2,
        kernel_size: Optional[ScalarOrTuple[int]] = None,
        mode: Union[UpsampleMode, str] = "default",
        pre_conv: Optional[Union[paddle.nn.Layer, str, int]] = "default",
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        align_corners: bool = ALIGN_CORNERS,
        apply_pad_pool: bool = True,
        padding_mode: Union[PaddingMode, str] = PaddingMode.ZEROS,
        bias: bool = True,
        init: str = "default",
    ) -> None:
        r"""Initialize upsampling layer.

        Args:
            spatial_dims: Number of spatial dimensions of input tensor.
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor. Defaults to `in_channels`.
            scale_factor: Multiplier for spatial size. Has to match input size if it is a tuple.
            kernel_size: Kernel size of transposed convolution in "deconv" mode or default ``pre_conv`` otherwise.
                The default kernel size in "deconv" mode is equal to the specified ``scale_factor``. The default
                ``pre_conv`` kernel size in "interpolate" mode is 1. In "pixelshuffle" mode, the default is 3.
            mode: Upsampling mode: "deconv", "interpolate", or "pixelshuffle".
            pre_conv: A conv block applied before upsampling. When ``conv_block`` is ``"default"``, one reserved
                conv layer will be utilized. This argument is only used for "interpolate" or "pixelshuffle" mode.
            sampling: Interpolation mode used for ``paddle.nn.Upsample`` in "interpolate" mode.
            align_corners: See `paddle.nn.Upsample`.
            apply_pad_pool: If True the upsampled tensor is padded then average pooling is applied with a kernel the
                size of `scale_factor` with a stride of 1. Only used in the pixelshuffle mode.
            padding_mode: Padding mode to use for default ``pre_conv`` and pixelshuffle ``apply_pad_pool``.
            bias: Whether to have a bias term in the default preconv and deconv layers.
            init: How to initialize default ``conv_block`` weights (cf. ``convolution()``). In case of "pixelshuffle" mode,
                if value is "icnr" or "default", ICNR initialization is used. Use "uniform" for initialization without ICNR.

        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        upsample_mode = UpsampleMode.DECONV if mode == "default" else UpsampleMode(mode)
        padding_mode = PaddingMode(padding_mode).conv_mode(spatial_dims)

        if upsample_mode == UpsampleMode.DECONV:
            if not in_channels:
                raise ValueError(
                    f"{type(self).__name__}() 'in_channels' required in {upsample_mode.value!r} mode"
                )

            if isinstance(scale_factor, (int, float)):
                scale_factor = (scale_factor,) * spatial_dims
            elif len(scale_factor) != spatial_dims:
                raise ValueError(
                    f"{type(self).__name__}() 'scale_factor' must be scalar or sequence of length {spatial_dims}"
                )
            try:
                scale_factor = tuple(int(s) for s in scale_factor)
            except TypeError:
                raise TypeError(
                    f"{type(self).__name__}() 'scale_factor' must be ints for {upsample_mode.value!r} mode"
                )

            if kernel_size is None:
                kernel_size = scale_factor
            elif isinstance(kernel_size, (int, float)):
                kernel_size = (kernel_size,) * spatial_dims
            elif len(kernel_size) != spatial_dims:
                raise ValueError(
                    f"{type(self).__name__}() 'kernel_size' must be scalar or sequence of length {spatial_dims}"
                )
            if any(k < s for k, s in zip(kernel_size, scale_factor)):
                raise ValueError(
                    f"{type(self).__name__}() 'kernel_size' must be greater than or equal to 'scale_factor'"
                )

            # Output size should be scale factor times input size, thus:
            # 2 * padding - output_padding == kernel_size - scale_factor
            padding = upsample_padding(kernel_size, scale_factor)
            output_padding = upsample_output_padding(kernel_size, scale_factor, padding)

            deconv = convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=padding,
                output_padding=output_padding,
                dilation=1,
                init=init,
                bias=bias,
                transposed=True,
            )
            self.add_sublayer(name="deconv", sublayer=deconv)

        elif upsample_mode == UpsampleMode.INTERPOLATE:
            if pre_conv == "default":
                if in_channels is None or in_channels == out_channels:
                    pre_conv = None
                else:
                    if kernel_size is None:
                        kernel_size = 1
                    if isinstance(kernel_size, int):
                        kernel_size = (kernel_size,) * spatial_dims
                    elif not isinstance(kernel_size, Sequence):
                        raise TypeError(
                            f"{type(self).__name__}() 'kernel_size' must be int or Sequence[int]"
                        )
                    elif len(kernel_size) != spatial_dims:
                        raise ValueError(
                            f"{type(self).__name__}() 'kernel_size' must be int or {spatial_dims}-tuple"
                        )
                    if any(k < 1 for k in kernel_size):
                        raise ValueError(f"{type(self).__name__}() 'kernel_size' must be positive")
                    if any(ks % 2 == 0 for ks in kernel_size):
                        padding = tuple(((ks - 1) // 2, ks // 2) for ks in kernel_size)
                        padding = tuple(p for a, b in reversed(padding) for p in (a, b))
                        pre_pad = Pad(padding=padding, mode=padding_mode)
                        self.add_sublayer(name="prepad", sublayer=pre_pad)
                        padding = 0
                    else:
                        padding = tuple(ks // 2 for ks in kernel_size)
                    pre_conv = convolution(
                        spatial_dims=spatial_dims,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                        init=init,
                        bias=bias,
                    )
            if pre_conv is not None:
                if not isinstance(pre_conv, paddle.nn.Layer):
                    raise TypeError(
                        f"{type(self).__name__}() 'preconv' must be string 'default' or Module"
                    )
                self.add_sublayer(name="preconv", sublayer=pre_conv)

            mode = Sampling(sampling).interpolate_mode(spatial_dims)
            upsample = paddle.nn.Upsample(
                scale_factor=scale_factor, mode=mode, align_corners=align_corners
            )
            self.add_sublayer(name="interpolate", sublayer=upsample)

        elif upsample_mode == UpsampleMode.PIXELSHUFFLE:
            try:
                scale_factor_ = int(scale_factor)
            except TypeError:
                raise TypeError(
                    f"{type(self).__name__}() 'scale_factor' must be int for {upsample_mode.value!r} mode"
                )

            if kernel_size is None:
                kernel_size = 3

            module = SubpixelUpsample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=scale_factor_,
                conv_block=pre_conv,
                apply_pad_pool=apply_pad_pool,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                init=init,
                bias=bias,
            )
            self.add_sublayer(name="pixelshuffle", sublayer=module)

        else:
            raise NotImplementedError(f"{type(self).__name__}() mode={mode!r} not implemented")


class SubpixelUpsample(paddle.nn.Layer):
    r"""Upsample using a subpixel CNN.

    This module is adapted from the MONAI project and supports 1D, 2D and 3D input images.
    The module consists of two parts. First of all, a convolutional layer is employed
    to increase the number of channels into: ``in_channels * (scale_factor ** spatial_dims)``.
    Secondly, a pixel shuffle manipulation is utilized to aggregate the feature maps from
    low resolution space and build the super resolution space. The first part of the module
    is not fixed, a sequential layer can be used to replace the default single layer.

    See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
    Using a nEfficient Sub-Pixel Convolutional Neural Network."

    See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

    The idea comes from:
    https://arxiv.org/abs/1609.05158

    The pixel shuffle mechanism refers to:
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/PixelShuffle.cpp
    and:
    https://github.com/pytorch/pytorch/pull/6340/files

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: Optional[int],
        out_channels: Optional[int] = None,
        scale_factor: Union[int, float] = 2,
        conv_block: Optional[Union[paddle.nn.Layer, str, int]] = "default",
        apply_pad_pool: bool = True,
        kernel_size: ScalarOrTuple[int] = 3,
        padding_mode: Union[PaddingMode, str] = PaddingMode.ZEROS,
        init: str = "default",
        bias: Union[bool, str] = True,
    ) -> None:
        r"""Initialize upsampling layer.

        Args:
            spatial_dims: Number of spatial dimensions of the input image.
            in_channels: Number of channels of the input image.
            out_channels: Optional number of channels of the output image.
            scale_factor: Multiplier for spatial size. Must be castable to ``int``. Defaults to 2.
            conv_block: A conv block to extract feature maps before upsampling.

                - When ``"default"``, one reserved conv layer will be utilized.
                - When ``paddle.nn.Layer``, the output number of channels must be divisible by ``(scale_factor ** spatial_dims)``.

            apply_pad_pool: If True the upsampled tensor is padded then average pooling is applied with a kernel the
                size of `scale_factor` with a stride of 1. This implements the nearest neighbour resize convolution
                component of subpixel convolutions described in Aitken et al.
            kernel_size: Size of default ``conv_block`` kernel. Defaults to 3.
            padding_mode: Padding mode to use for default ``conv_block`` and ``apply_pad_pool``.
            init: How to initialize default ``conv_block`` weights (cf. ``convolution()``). If value is "icnr" or
                "default", ICNR initialization is used. Use "uniform" for standard initialization without ICNR.
            bias: Whether to have a bias term in the default conv_block. When a string is given, it specifies how
                the bias term is initialized (cf. ``convolution()``).

        """
        super().__init__()

        try:
            scale_factor = int(scale_factor)
        except TypeError:
            raise TypeError("SubpixelUpsample() 'scale_factor' must be int")
        if scale_factor < 1:
            raise ValueError("SubpixelUpsample() 'scale_factor' must be a positive integer")

        if init in ("icnr", "ICNR"):
            init = "default"

        self.spatial_dims = spatial_dims
        self.scale_factor = scale_factor

        if conv_block == "default":
            if not in_channels:
                raise ValueError("SubpixelUpsample() 'in_channels' required")
            out_channels = out_channels or in_channels
            conv_out_channels = out_channels * (scale_factor**spatial_dims)
            if kernel_size % 2 == 0:
                padding = ((kernel_size - 1) // 2, kernel_size // 2) * spatial_dims
                pre_pad = Pad(padding=padding, mode=padding_mode)
                padding = 0
            else:
                pre_pad = None
                padding = kernel_size // 2
            conv_block = convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                init=init,
                bias=bias,
            )
            if init == "default":
                icnr_init(conv_block.weight, scale_factor)
            if pre_pad is not None:
                conv_block = paddle.nn.Sequential(pre_pad, conv_block)
        elif conv_block is None:
            conv_block = paddle.nn.Identity()
        elif not isinstance(conv_block, paddle.nn.Layer):
            raise ValueError(
                "SubpixelUpsample() 'conv_block' must be string 'default', Module, or None"
            )
        self.conv_block = conv_block

        if apply_pad_pool:
            pad_pool = paddle.nn.Sequential(
                Pad(padding=(scale_factor - 1, 0) * spatial_dims, mode=padding_mode, value=0),
                pooling("avg", spatial_dims=spatial_dims, kernel_size=scale_factor, stride=1),
            )
        else:
            pad_pool = paddle.nn.Identity()
        self.pad_pool = pad_pool

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """

        Args:
            x: Tensor in shape (batch, channel, spatial_1[, spatial_2, ...).

        """
        x = self.conv_block(x)
        x = pixelshuffle(x, self.spatial_dims, self.scale_factor)
        x = self.pad_pool(x)
        return x


def icnr_init(
    weight: paddle.Tensor, upsample_factor: int, init=paddle.nn.initializer.KaimingNormal
) -> None:
    r"""ICNR initialization for 2D/3D kernels.

    Adapted from MONAI project and based on Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

    """
    out_channels, in_channels, *dims = weight.shape
    scale_factor = upsample_factor ** len(dims)

    oc2 = int(out_channels / scale_factor)

    kernel = paddle.zeros(shape=[oc2, in_channels] + dims)
    kernel: Tensor = init(kernel)
    kernel = kernel.transpose(perm=paddle_aux.transpose_aux_func(kernel.ndim, 0, 1))
    kernel = kernel.reshape(oc2, in_channels, -1)
    kernel = kernel.tile(repeat_times=[1, 1, scale_factor])
    kernel = kernel.reshape([in_channels, out_channels] + dims)
    kernel = kernel.transpose(perm=paddle_aux.transpose_aux_func(kernel.ndim, 0, 1))
    weight.data.copy_(kernel)


def pixelshuffle(x: paddle.Tensor, spatial_dims: int, scale_factor: int) -> paddle.Tensor:
    r"""Apply pixel shuffle to the tensor `x` with spatial dimensions `spatial_dims` and scaling factor `scale_factor`.

    See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network."

    See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

    Args:
        x: Input tensor
        spatial_dims: number of spatial dimensions, typically 2 or 3 for 2D or 3D
        scale_factor: factor to rescale the spatial dimensions by, must be >=1

    Returns:
        Reshuffled version of `x`.

    Raises:
        ValueError: When input channels of `x` are not divisible by (scale_factor ** spatial_dims)

    """

    dim, factor = spatial_dims, scale_factor
    input_size = list(x.size())
    batch_size, channels = input_size[:2]
    scale_divisor = factor**dim

    if channels % scale_divisor != 0:
        raise ValueError(
            f"pixelshuffle() number of input channels ({channels}) must be evenly divisible by scale_factor ** spatial_dims ({factor}**{dim}={scale_divisor})"
        )

    org_channels = channels // scale_divisor
    output_size = [batch_size, org_channels] + [(d * factor) for d in input_size[2:]]

    indices = tuple(range(2, 2 + 2 * dim))
    indices_factor, indices_dim = indices[:dim], indices[dim:]
    permute_indices = (0, 1) + sum(zip(indices_dim, indices_factor), ())

    x = x.reshape(batch_size, org_channels, *([factor] * dim + input_size[2:]))
    x = x.transpose(perm=permute_indices).reshape(output_size)
    return x
