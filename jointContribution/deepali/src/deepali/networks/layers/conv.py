r"""Convolutional layers."""

# TODO: ConvLayer() support use of dropout, by default between N and A.

import math
from numbers import Integral
from typing import Any
from typing import Optional
from typing import Union

import paddle
from deepali.core.enum import PaddingMode
from deepali.core.nnutils import same_padding
from deepali.core.nnutils import stride_minus_kernel_padding
from deepali.core.typing import ScalarOrTuple
from deepali.core.typing import ScalarOrTuple1d
from deepali.core.typing import ScalarOrTuple2d
from deepali.core.typing import ScalarOrTuple3d
from deepali.modules import ReprWithCrossReferences
from deepali.utils import paddle_aux  # noqa
from paddle.nn import Conv1D as _Conv1d
from paddle.nn import Conv1DTranspose as _ConvTranspose1d
from paddle.nn import Conv2D as _Conv2d
from paddle.nn import Conv2DTranspose as _ConvTranspose2d
from paddle.nn import Conv3D as _Conv3d
from paddle.nn import Conv3DTranspose as _ConvTranspose3d

from .acti import ActivationArg
from .acti import activation
from .norm import NormArg
from .norm import normalization

__all__ = (
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "ConvLayer",
    "convolution",
    "conv_module",
    "is_convolution",
    "is_conv_module",
    "is_transposed_convolution",
    "same_padding",
    "stride_minus_kernel_padding",
)


class _ConvInit(object):
    r"""Mix-in for initialization of convolutional layer parameters."""

    def reset_parameters(self) -> None:
        # Initialize weights
        if self.weight_init in ("default", "uniform"):
            init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
                negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
            )
            init_KaimingUniform(self.weight)
        elif self.weight_init == "xavier":
            init_XavierUniform = paddle.nn.initializer.XavierUniform()
            init_XavierUniform(self.weight)
        elif self.weight_init == "constant":
            init_Constant = paddle.nn.initializer.Constant(value=0.1)
            init_Constant(self.weight)
        elif self.weight_init == "zeros":
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.weight)
        else:
            raise AssertionError(
                f"{type(self).__name__}.reset_parameters() invalid 'init' value: {self.weight_init!r}"
            )
        # Initialize bias
        if self.bias is not None:
            if self.bias_init in ("default", "uniform"):
                fan_in, _ = paddle_aux._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init_Uniform = paddle.nn.initializer.Uniform(low=-bound, high=bound)
                init_Uniform(self.bias)
            elif self.bias_init == "constant":
                init_Constant = paddle.nn.initializer.Constant(value=0.1)
                init_Constant(self.bias)
            elif self.bias_init == "zeros":
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(self.bias)
            else:
                raise AssertionError(
                    f"{type(self).__name__}.reset_parameters() invalid 'bias' value: {self.bias_init!r}"
                )


class Conv1d(_ConvInit, paddle.nn.Conv1D):
    r"""Convolutional layer with custom initialization of learnable parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple1d[int],
        stride: ScalarOrTuple1d[int] = 1,
        padding: ScalarOrTuple1d[int] = 0,
        dilation: ScalarOrTuple1d[int] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        padding_mode: str = "zeros",
        init: str = "default",
    ):
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = "uniform" if init == "default" else init
        bias_attr = None if bool(bias) else False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
            padding_mode=padding_mode,
        )


class Conv2d(_ConvInit, paddle.nn.Conv2D):
    r"""Convolutional layer with custom initialization of learnable parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple2d[int],
        stride: ScalarOrTuple2d[int] = 1,
        padding: ScalarOrTuple2d[int] = 0,
        dilation: ScalarOrTuple2d[int] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        padding_mode: str = "zeros",
        init: str = "default",
    ):
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = "uniform" if init == "default" else init
        bias_attr = None if bool(bias) else False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
            padding_mode=padding_mode,
        )


class Conv3d(_ConvInit, paddle.nn.Conv3D):
    r"""Convolutional layer with custom initialization of learnable parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple3d[int],
        stride: ScalarOrTuple3d[int] = 1,
        padding: ScalarOrTuple3d[int] = 0,
        dilation: ScalarOrTuple3d[int] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        padding_mode: str = "zeros",
        init: str = "uniform",
    ):
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = "uniform" if init == "default" else init
        bias_attr = None if bool(bias) else False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
            padding_mode=padding_mode,
        )


class ConvTranspose1d(_ConvInit, paddle.nn.Conv1DTranspose):
    r"""Transposed convolution in 1D with custom initialization of learnable parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple1d[int],
        stride: ScalarOrTuple1d[int] = 1,
        padding: ScalarOrTuple1d[int] = 0,
        output_padding: ScalarOrTuple1d[int] = 0,
        dilation: ScalarOrTuple1d[int] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        padding_mode: str = "zeros",
        init: str = "default",
    ):
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = "uniform" if init == "default" else init
        bias_attr = None if bool(bias) else False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
            padding_mode=padding_mode,
        )


class ConvTranspose2d(_ConvInit, paddle.nn.Conv2DTranspose):
    r"""Transposed convolution in 2D with custom initialization of learnable parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple2d[int],
        stride: ScalarOrTuple2d[int] = 1,
        padding: ScalarOrTuple2d[int] = 0,
        output_padding: ScalarOrTuple2d[int] = 0,
        dilation: ScalarOrTuple2d[int] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        padding_mode: str = "zeros",
        init: str = "default",
    ):
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = "uniform" if init == "default" else init
        bias_attr = None if bool(bias) else False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
            # padding_mode=padding_mode,
        )


class ConvTranspose3d(_ConvInit, paddle.nn.Conv3DTranspose):
    r"""Transposed convolution in 3D with custom initialization of learnable parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple3d[int],
        stride: ScalarOrTuple3d[int] = 1,
        padding: ScalarOrTuple3d[int] = 0,
        output_padding: ScalarOrTuple3d[int] = 0,
        dilation: ScalarOrTuple3d[int] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        padding_mode: str = "zeros",
        init: str = "default",
    ):
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = "uniform" if init == "default" else init
        bias_attr = None if bool(bias) else False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
            # padding_mode=padding_mode,
        )


class ConvLayer(ReprWithCrossReferences, paddle.nn.Sequential):
    r"""Convolutional layer with optional pre- or post-convolution normalization and/or activation."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple[int],
        stride: ScalarOrTuple[int] = 1,
        padding: Optional[ScalarOrTuple[int]] = None,
        output_padding: Optional[ScalarOrTuple[int]] = None,
        padding_mode: Union[PaddingMode, str] = "zeros",
        dilation: ScalarOrTuple[int] = 1,
        groups: int = 1,
        init: str = "default",
        bias: Optional[Union[bool, str]] = None,
        norm: NormArg = None,
        acti: ActivationArg = None,
        order: str = "CNA",
        transposed: bool = False,
        conv: Optional[paddle.nn.Layer] = None,
    ) -> None:
        if spatial_dims < 0 or spatial_dims > 3:
            raise ValueError("ConvLayer() 'spatial_dims' must be 1, 2, or 3")
        if isinstance(kernel_size, Integral):
            kernel_size = (int(kernel_size),) * spatial_dims
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        # Order of layer operations
        if not isinstance(order, str):
            raise TypeError("ConvLayer() 'order' must be str")
        order = order.upper()
        if "C" not in order:
            raise ValueError("ConvLayer() 'order' must contain 'C' for convolution")
        if "D" in order:
            raise NotImplementedError("ConvLayer() 'order' has 'D' for not implemented dropout")
        if len(set(order)) != len(order) or any(c not in {"A", "C", "N"} for c in order):
            raise ValueError(
                f"ConvLayer() 'order' must be permutation of characters 'A' (activation), 'N' (norm), and 'C' (convolution), got {order!r}"
            )
        # Non-linear activation
        if acti is None or "A" not in order:
            acti_fn = None
        else:
            acti_fn = activation(acti)
        # Normalization layer
        norm_after_conv = False
        if norm is None or "N" not in order:
            norm_layer = None
        else:
            norm_after_conv = order.index("C") < order.index("N")
            num_features = out_channels if norm_after_conv else in_channels
            norm_layer = normalization(norm, spatial_dims=spatial_dims, num_features=num_features)
        # Convolution layer
        if bias is None:
            bias = True
            if norm_after_conv and isinstance(norm_layer, paddle.nn.Layer):
                bias = not getattr(norm_layer, "affine", False)
        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride)
        if conv is None:
            conv = convolution(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                bias=bias,
                init=init,
                transposed=transposed,
            )
        # Initialize module
        modules = {"A": ("acti", acti_fn), "N": ("norm", norm_layer), "C": ("conv", conv)}
        modules = [modules[k] for k in order if modules[k][1] is not None]
        super().__init__(*modules)
        if acti_fn is None:
            self.acti = None
        if norm_layer is None:
            self.norm = None
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = order

    def has_norm_after_conv(self) -> bool:
        r"""Whether this layer has a normalization layer after the convolution."""
        names = tuple(
            name for name, _ in self.named_sublayers(include_self=True) if name in ("conv", "norm")
        )
        return len(names) == 2 and names[-1] == "norm"


def convolution(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: Optional[ScalarOrTuple[int]] = 0,
    output_padding: Optional[ScalarOrTuple[int]] = 0,
    padding_mode: Union[PaddingMode, str] = "zeros",
    dilation: ScalarOrTuple[int] = 1,
    groups: int = 1,
    init: str = "default",
    bias: Union[bool, str] = True,
    transposed: bool = False,
) -> paddle.nn.Layer:
    r"""Create convolution module for specified number of spatial input tensor dimensions."""
    if in_channels < 1:
        raise ValueError(f"convolution() 'in_channels' ({in_channels}) must be positive")
    if out_channels < 1:
        raise ValueError(f"convolution() 'out_channels' ({out_channels}) must be positive")
    padding_mode = PaddingMode.from_arg(padding_mode)
    if padding_mode is PaddingMode.NONE:
        padding = 0
    elif padding is None:
        padding = same_padding(kernel_size, dilation)
    kwargs = dict(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        init=init,
    )
    # Specify non-default padding_mode only if used so conv.extra_repr() does not show it
    if any(n != 0 for n in ((padding,) if isinstance(padding, int) else padding)):
        kwargs["padding_mode"] = padding_mode.conv_mode(spatial_dims)
    if transposed:
        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride)
        kwargs["output_padding"] = output_padding
    if spatial_dims == 1:
        conv_type = ConvTranspose1d if transposed else Conv1d
    elif spatial_dims == 2:
        conv_type = ConvTranspose2d if transposed else Conv2d
    elif spatial_dims == 3:
        conv_type = ConvTranspose3d if transposed else Conv3d
    else:
        raise ValueError("convolution() 'spatial_dims' must be 1, 2, or 3")
    return conv_type(in_channels, out_channels, **kwargs)


def conv_module(*args, **kwargs) -> paddle.nn.Layer:
    r"""Create convolution layer, see ``convolution()``."""
    return convolution(*args, **kwargs)


def is_convolution(arg: Any) -> bool:
    r"""Whether given module is a learnable convolution."""
    types = (
        _Conv1d,
        _Conv2d,
        _Conv3d,
        _ConvTranspose1d,
        _ConvTranspose2d,
        _ConvTranspose3d,
    )
    return isinstance(arg, types)


def is_conv_module(arg: Any) -> bool:
    r"""Whether given module is a learnable convolution."""
    return is_convolution(arg)


def is_transposed_convolution(arg: Any) -> bool:
    r"""Whether given module is a learnable transposed convolution."""
    types = _ConvTranspose1d, _ConvTranspose2d, _ConvTranspose3d
    return isinstance(arg, types)
