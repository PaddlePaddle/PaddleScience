r"""Blocks of network layers with residual skip connections."""
from __future__ import annotations  # noqa

from typing import Any
from typing import Mapping
from typing import Optional
from typing import Union

import paddle
from deepali.core.enum import PaddingMode
from deepali.core.typing import ScalarOrTuple

from ..layers.acti import ActivationArg
from ..layers.acti import activation
from ..layers.conv import ConvLayer
from ..layers.conv import convolution
from ..layers.conv import same_padding
from ..layers.join import JoinLayer
from ..layers.norm import NormArg
from .skip import SkipConnection
from .skip import SkipFunc

# fmt: off
__all__ = (
    "ResidualUnit",
)
# fmt: on


class ResidualUnit(SkipConnection):
    r"""Sequence of convolutional layers with a residual skip connection.

    Implements a number of variants of residual blocks as described in:
    - He et al., 2015, Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385
    - He et al., 2016, Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027
    - Xie et al., 2017, Aggregated Residual Transformations for Deep Neural Networks, https://arxiv.org/abs/1611.05431

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_channels: Optional[int] = None,
        kernel_size: ScalarOrTuple[int] = 3,
        stride: ScalarOrTuple[int] = 1,
        padding: Optional[ScalarOrTuple[int]] = None,
        padding_mode: Union[PaddingMode, str] = "zeros",
        dilation: ScalarOrTuple[int] = 1,
        groups: int = 1,
        init: str = "default",
        bias: Optional[Union[bool, str]] = False,
        norm: NormArg = "batch",
        acti: ActivationArg = "relu",
        skip: Optional[Union[SkipFunc, str, Mapping[str, Any]]] = "identity | conv1 | conv",
        num_layers: Optional[int] = None,
        order: str = "cna",
        pre_conv: str = "conv1",
        post_conv: str = "conv1",
        other: Optional[ResidualUnit] = None,
    ) -> None:
        r"""Initialize residual unit.

        Args:
            spatial_dims: Number of spatial input and output dimensions.
            in_channels: Number of input feature maps.
            out_channels: Number of output feature maps.
            num_channels: Number of input feature maps to spatial convolutions.
                If ``None`` or equal to ``out_channels``, the residual branch consists of ``num_layers``
                of convolutions with specified ``kernel_size`` and pre- or post-activation (and normalization).
                Otherwise, the first and last convolution has kernel size 1 in order to match the
                specified number of input and output channels of the shortcut connection.
                When the interim number of channels is smaller than the input and output channels,
                this residual unit is a so-called bottleneck block.
            kernel_size: Size of residual convolutions.
            pre_conv: Type of pre-convolution when residual unit is a bottleneck block.
                The kernel size is set to 1 if "conv1", and ``kernel_size`` otherwise.
            post_conv: Type of post-convolution when residual unit is a bottleneck block. See ``pre_conv``.
            stride: Stride of first residual convolution. All subsequent convolutions have stride 1.
                In case of a bottleneck block, the first convolution with kernel size 1 also has stride 1,
                and the specified stride is applied on the second convolution instead.
            padding: Padding used by residual convolutions with specified ``kernel_size``.
                If specified, must result in a same padding such that spatial tensor shape remains unchanged.
            padding_mode: Padding mode used for same padding of input to each convolutional layers.
            dilation: Dilation of convolution kernel.
            groups: Number of groups into which to split each convolution. Note that combined with a bottleneck,
                i.e., where ``num_channels`` is smaller than ``out_channels``, this is equivalent to increasing
                the number of residual paths also referred to as cardinality (cf. ResNeXt). For example, setting
                ``in_channels=256``, out_channels=256``, ``num_channels=128``, and ``groups=32`` is equivalent to
                a residual block consisting of 32 paths with 4 channels in each path which are concatentated prior
                to the final convolution with kernel size 1 before the resulting residuals are added to the result
                of the (identity) skip connection (cf. Xie et al., 2017, Fig. 3).
            init: Mode used to initialize weights of convolution kernels.
            bias: Whether to use bias terms of convolutional filters.
            norm: Normalization layer to use in each convolutional layer.
            acti: Activation function to use in each convolutional layer.
            skip: Function along the shortcut connection. If specified, must ensure that the
                shape of the output along this shortcut connection is identical to the shape of
                the output from the sequence of convolutional layers in this block. If ``None``,
                the shortcut connection is the identity map. By default, the identity map is used when the
                output shape matches the input shape. Otherwise, a convolution with kernel size 1 is used if
                ``in_channels != out_channels``, or a convolution with specified ``kernel_size``. The argument
                can be a string specifying which of these, i.e., "identity", "conv1", or "conv", to consider.
                Multiple options can be allowed by using the "|" operator. The default is "identity | conv1 | conv".
                In order to always use a convolution with kernel size 1, even in case of a stride > 1, use
                "identity | conv1". To force the use of the identity, set argument to ``None`` or "identity".
            num_layers: Number of convolutional layers in residual block.
            order: Order of convolution, normalization, and activation in each layer.
                In case of post-convolution activation (post-activiation), the final activation
                is applied after the addition of the layer output with the shortcut connection.
                Otherwise, the addition is of input and output is the last operation of this block.
            other: If specified, convolutions in convolutional layers are reused. The given residual unit
                must have created with the same parameters as this residual unit. Note that the resulting
                and the other residual unit will share references to the same convolution modules.

        """
        if spatial_dims < 0 or spatial_dims > 3:
            raise ValueError("ResidualUnit() 'spatial_dims' must be 1, 2, or 3")
        order = order.upper()
        if len(set(order)) != len(order) or "C" not in order or "A" not in order:
            raise ValueError(
                f"ResidualUnit() 'order' must be permutation of unique characters 'a|A' (activation), 'n|N' (norm, optional), and 'c|C' (convolution), got {order!r}"
            )
        # Pre- vs. post-activation
        if order.index("C") < order.index("A"):
            post_acti = activation(acti)
        else:
            post_acti = None
        # Add convolutional layers
        if out_channels is None:
            out_channels = in_channels
        if num_channels is None:
            num_channels = out_channels
        is_bottleneck = num_channels != out_channels
        if num_layers is None:
            num_layers = 3 if is_bottleneck else 2
        if not isinstance(num_layers, int):
            raise TypeError("ResidualUnit() 'num_layers' must be int or None")
        if is_bottleneck and num_layers < 3:
            raise ValueError(
                "ResidualUnit() 'num_layers' must be at least 3 in case of a bottleneck block"
            )
        elif num_layers < 1:
            raise ValueError("ResidualUnit() 'num_layers' must be positive")
        residual = paddle.nn.Sequential()
        if is_bottleneck:
            name = f"layer_{len(residual)}"
            if other is None:
                other_conv = None
            else:
                other_layer = getattr(other.residual, name)
                assert isinstance(other_layer, ConvLayer)
                other_conv = other_layer.conv
                assert isinstance(other_conv, paddle.nn.Layer)
            if pre_conv == "conv":
                pre_kernel_size = kernel_size
            elif pre_conv == "conv1":
                pre_kernel_size = 1
            else:
                raise ValueError("ResidualUnit() 'pre_conv' must be 'conv' or 'conv1'")
            conv = ConvLayer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=pre_kernel_size,
                padding_mode=padding_mode,
                init=init,
                bias=bias,
                norm=norm,
                acti=acti,
                order=order,
                conv=other_conv,
            )
            residual.add_sublayer(name=name, sublayer=conv)
        for i in range(1 if is_bottleneck else 0, num_layers - 1 if is_bottleneck else num_layers):
            name = f"layer_{len(residual)}"
            is_first_layer = i == 0
            is_last_layer = i == num_layers - 1
            if other is None:
                other_conv = None
            else:
                other_layer = getattr(other.residual, name)
                assert isinstance(other_layer, ConvLayer)
                other_conv = other_layer.conv
                assert isinstance(other_conv, paddle.nn.Layer)
            conv = ConvLayer(
                spatial_dims=spatial_dims,
                in_channels=in_channels if is_first_layer else num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride if is_first_layer else 1,
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                init=init,
                bias=bias,
                norm=norm,
                acti=None if is_last_layer and post_acti is not None else acti,
                order=order,
                conv=other_conv,
            )
            residual.add_sublayer(name=name, sublayer=conv)
        if is_bottleneck:
            name = f"layer_{len(residual)}"
            if other is None:
                other_conv = None
            else:
                other_layer = getattr(other.residual, name)
                assert isinstance(other_layer, ConvLayer)
                other_conv = other_layer.conv
                assert isinstance(other_conv, paddle.nn.Layer)
            if post_conv == "conv":
                post_kernel_size = kernel_size
            elif post_conv == "conv1":
                post_kernel_size = 1
            else:
                raise ValueError("ResidualUnit() 'post_conv' must be 'conv' or 'conv1'")
            conv = ConvLayer(
                spatial_dims=spatial_dims,
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=post_kernel_size,
                padding_mode=padding_mode,
                init=init,
                bias=bias,
                norm=norm,
                acti=acti if post_acti is None else None,
                order=order,
                conv=other_conv,
            )
            residual.add_sublayer(name=name, sublayer=conv)
        # Apply convolution along skip connection to match shape of residual tensor
        has_strided_conv = (
            not paddle.to_tensor(data=stride, dtype="int32", place="cpu").prod().equal(y=1)
        )
        if skip in (None, ""):
            skip = "identity"
        if isinstance(skip, str):
            skip_names = [s.strip() for s in skip.lower().split("|")]
            if has_strided_conv or in_channels != out_channels:
                if "conv" in skip_names and (has_strided_conv or "conv1" not in skip_names):
                    skip = {}
                elif "conv1" in skip_names:
                    skip = {"kernel_size": 1, "padding": 0}
                elif skip == "identity":
                    raise ValueError(
                        f"ResidualUnit() cannot use 'identity' skip connection (has_strided_conv={has_strided_conv}, in_channels={in_channels}, out_channels={out_channels})"
                    )
                else:
                    raise ValueError(f"ResidualUnit() invalid 'skip' value {skip!r}")
            elif "identity" in skip_names:
                skip = paddle.nn.Identity()
            elif "conv1" in skip_names:
                skip = {"kernel_size": 1, "padding": 0}
            elif "conv" in skip_names:
                skip = {}
            else:
                raise ValueError(f"ResidualUnit() invalid 'skip' value {skip!r}")
        if isinstance(skip, dict):
            if "padding" in skip and "kernel_size" not in skip:
                raise ValueError("ResidualUnit() 'skip' specifies 'padding' but not 'kernel_size'")
            conv_args = dict(
                kernel_size=kernel_size if has_strided_conv else 1,
                stride=stride,
                dilation=1,
                init=init,
                bias=bias is not False,
            )
            conv_args.update(skip)
            if "padding" not in conv_args:
                conv_args["padding"] = same_padding(conv_args["kernel_size"], conv_args["dilation"])
            skip = convolution(spatial_dims, in_channels, out_channels, **conv_args)
        elif not callable(skip):
            raise TypeError("ResidualUnit() 'skip' must be str, dict, callable, or None")
        # Function used to combine input with residual tensor
        add = JoinLayer("add")
        if post_acti is None:
            join = add
        else:
            join = paddle.nn.Sequential()
            join.add_sublayer(name="add", sublayer=add)
            join.add_sublayer(name="acti", sublayer=post_acti)
        # Initialize parent class and set module attributes
        super().__init__(residual, name="residual", skip=skip, join=join)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels

    def is_bottleneck(self) -> bool:
        r"""Whether this residual unit is a "bottleneck" type ResNet block."""
        return self.out_channels != self.num_channels

    @property
    def last_conv_layer(self) -> ConvLayer:
        r"""Get last residual convolutional layer."""
        return self.func[-1]

    def zero_init_residual(self) -> None:
        r"""Zero-initialize normalization layer in residual branch.

        This is so that the residual branch starts with zeros, and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677.

        """
        conv_layer = self.last_conv_layer
        if conv_layer.has_norm_after_conv():
            weight = getattr(conv_layer.norm, "weight", None)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(weight)
