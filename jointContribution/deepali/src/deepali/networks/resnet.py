r"""Convolutional network with residual shortcut connections.

For further reference implementations of 2D and 3D ResNet, see for example:

- https://github.com/pytorch/vision/blob/1703e4ca4f879f509e797ea816670ee72fae55dd/torchvision/models/resnet.py
- https://github.com/kenshohara/3D-ResNets-PyTorch/blob/540a0ea1abaee379fa3651d4d5afbd2d667a1f49/models/resnet.py

"""
from __future__ import annotations  # noqa

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import paddle
import paddlenlp
from deepali.core.config import DataclassConfig
from deepali.core.enum import PaddingMode
from deepali.core.itertools import zip_longest_repeat_last
from deepali.core.typing import ScalarOrTuple
from deepali.modules import ReprWithCrossReferences

from .blocks import ResidualUnit
from .layers import ActivationArg
from .layers import ConvLayer
from .layers import Linear
from .layers import NormArg
from .layers import Upsample
from .layers import is_batch_norm
from .layers import is_convolution
from .layers import is_group_norm
from .layers import pooling

# fmt: off
__all__ = (
    "ResidualUnit",
    "ResNet",
    "ResNetConfig",
)
# fmt: on
ModuleFactory = Union[Callable[..., paddle.nn.Layer], Type[paddle.nn.Layer]]
MODEL_DEPTHS = {10, 18, 34, 50, 101, 152, 200}


@dataclass
class ResNetConfig(DataclassConfig):
    r"""Configuration of residual network architecture."""

    # Input shape
    spatial_dims: int
    in_channels: int = 1

    # cf. ResNet and ResidualUnit
    stride: ScalarOrTuple[int] = (1, 2, 2, 2)
    num_blocks: ScalarOrTuple[int] = (3, 4, 6, 3)
    num_channels: Sequence[int] = (64, 128, 256, 512)
    num_layers: int = 2
    num_classes: Optional[int] = None
    kernel_size: int = 3
    expansion: float = 1
    padding_mode: Union[PaddingMode, str] = "zeros"
    pre_conv: bool = False
    post_deconv: bool = False
    recursive: ScalarOrTuple[bool] = False
    bias: bool = False
    norm: NormArg = "batch"
    acti: ActivationArg = "relu"
    order: str = "cna"
    skip: str = "identity | conv1"
    residual_pre_conv: str = "conv1"
    residual_post_conv: str = "conv1"

    @classmethod
    def from_depth(
        cls, model_depth: int, spatial_dims: int, in_channels: int = 1, **kwargs
    ) -> ResNetConfig:
        r"""Get default ResNet configuration for given depth."""
        config = cls(spatial_dims=spatial_dims, in_channels=in_channels, **kwargs)
        if model_depth == 10:
            config.num_blocks = (1, 1, 1, 1)
            config.num_layers = 2
            config.expansion = 1
        elif model_depth == 18:
            config.num_blocks = (2, 2, 2, 2)
            config.num_layers = 2
            config.expansion = 1
        elif model_depth == 34:
            config.num_blocks = (3, 4, 6, 3)
            config.num_layers = 2
            config.expansion = 1
        elif model_depth == 50:
            config.num_blocks = (3, 4, 6, 3)
            config.num_layers = 3
            config.expansion = 4
        elif model_depth == 101:
            config.num_blocks = (3, 4, 23, 3)
            config.num_layers = 3
            config.expansion = 4
        elif model_depth == 152:
            config.num_blocks = (3, 8, 36, 3)
            config.num_layers = 3
            config.expansion = 4
        elif model_depth == 200:
            config.num_blocks = (3, 24, 36, 3)
            config.num_layers = 3
            config.expansion = 4
        else:
            raise ValueError("ResNetConfig.from_depth() 'model_depth' must be in {MODEL_DEPTHS!r}")
        return config


def classification_head(
    spatial_dims: int, in_channels: int, num_classes: int, **kwargs
) -> paddle.nn.Layer:
    r"""Image classification head for ResNet model."""
    pool = pooling("AdaptiveAvg", spatial_dims=spatial_dims, output_size=1)
    fc = Linear(in_channels, num_classes)
    return paddle.nn.Sequential(pool, paddle.nn.Flatten(), fc)


def conv_layer(
    level: int, is_first: bool, spatial_dims: int, in_channels: int, out_channels: int, **kwargs
) -> paddle.nn.Layer:
    r"""Convolutional layer before/between residual blocks when ``pre_conv=True``."""
    norm = kwargs.get("norm")
    acti = kwargs.get("acti")
    order = kwargs.get("order", "cna").upper()
    if is_first:
        if "N" in order and order.index("N") < order.index("C"):
            norm = None
        if "A" in order and order.index("A") < order.index("C"):
            acti = None
    kwargs.update(dict(norm=norm, acti=acti, order=order))
    return ConvLayer(spatial_dims, in_channels, out_channels, **kwargs)


def input_layer(
    spatial_dims: int, in_channels: int, out_channels: int, **kwargs
) -> paddle.nn.Layer:
    r"""First convolutional ResNet layer."""
    order = kwargs.get("order", "cna").upper()
    norm = None if "N" in order and order.index("N") < order.index("C") else kwargs.get("norm")
    kwargs.update(dict(kernel_size=7, padding=3, stride=2, norm=norm, acti=None, order=order))
    conv = ConvLayer(spatial_dims, in_channels, out_channels, **kwargs)
    pool = pooling("max", spatial_dims=spatial_dims, kernel_size=3, stride=2, padding=1)
    return paddle.nn.Sequential(conv, pool)


class ResNet(ReprWithCrossReferences, paddle.nn.Sequential):
    r"""Residual network.

    Note that unlike ``torchvision.models.ResNet``, the ``__init__`` function of this class
    does not initialize the parameters of the model, other than the standard initialization
    for each module type. In order to apply the initialization of the torchvision ResNet, call
    functions ``init_conv_modules()``, ``init_norm_layers()``, and ``zero_init_residuals()``
    (in this order!) after constructing the ResNet model.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        num_channels: ScalarOrTuple[int] = (64, 128, 256, 512),
        num_blocks: ScalarOrTuple[int] = (3, 4, 6, 3),
        num_layers: int = 2,
        num_classes: Optional[int] = None,
        kernel_size: int = 3,
        stride: ScalarOrTuple[int] = (1, 2, 2, 2),
        expansion: float = 1,
        padding_mode: Union[PaddingMode, str] = "zeros",
        recursive: ScalarOrTuple[bool] = False,
        pre_conv: bool = False,
        post_deconv: bool = False,
        bias: Union[bool, str] = False,
        norm: NormArg = "batch",
        acti: ActivationArg = "relu",
        order: str = "cna",
        skip: str = "identity | conv1",
        residual_pre_conv: str = "conv1",
        residual_post_conv: str = "conv1",
        conv_layer: ModuleFactory = conv_layer,
        deconv_layer: ModuleFactory = Upsample,
        resnet_block: ModuleFactory = ResidualUnit,
        input_layer: Optional[ModuleFactory] = input_layer,
        output_layer: Optional[ModuleFactory] = None,
    ) -> None:
        r"""Initialize layers.

        Args:
            spatial_dims: Number of spatial tensor dimensions.
            in_channels: Number of input channels.
            num_channels: Number of feature channels at each level.
            num_blocks: Number of residual blocks at each level.
            num_layers: Number of convolutional layers in each residual block.
            num_classes: Number of output class probabilities of ``output_layer``.
            kernel_size: Size of convolutional filters in residual blocks.
            stride: Stride of initial convolution at each level. Subsequent convolutions have stride 1.
            expansion: Expansion factor of ``num_channels``. If specified, the number of input and output
                feature maps for each residual block are equal to ``expansion * num_channels``, and the
                bottleneck convolutional layers after an initial convolution with kernel size 1 operate
                on feature maps with ``num_channels`` each, which are subsequently expanded again by the
                specified expansion level by another convolution with kernel size 1.
            padding_mode: Padding mode for convolutional layers with kernel size greater than 1.
            recursive: Whether residual blocks at each level are applied recursively. If ``True``, all residual
                blocks at a given level share their convolutional modules. Other modules such as normalization
                layers are not shared. When ``recursive=False``, a new residual block without any shared modules
                is created each time. When ``recursive=True`` and the number of feature channels or spatial size
                of a given residual block does not match the number of output channels or spatial size of the
                preceeding block, respectively, a pre-convolutional layer which adjusts the number of channels
                and/or spatial size is inserted between these residual blocks even when ``pre_conv=False``.
            pre_conv: Always insert a separate convolutional layer between levels. When ``recursive=True``,
                this is also the case when ``pre_conv=False`` if the output and input tensor shapes of final
                and first residual block in subsequent levels do not match.
            post_deconv: Whether to place upsampling layers after the sequence of residual blocks for levels
                with a ``stride`` that is less than 1. By default, upsampling is performed as part of the
                ``pre_conv``. If ``post_deconv=False``, a pre-upsampling layer is always inserted when a
                level has an initial stride of less than 1 regardless of the ``pre_conv`` setting.
            bias: Whether to use bias terms of convolutional layers. Can be either a boolean, or a string
                indicating the function used to initialize these bias terms (cf. ``ConvLayer``).
            norm: Type of normalization to use in convolutional layers. Use no normalization if ``None``.
            acti: Type of non-linear activation function to use in each convolutional layer.
            order: Order of convolution (C), normalization (N), and non-linear activation (A) in each
                convolutional layer. If this string does not contain the character ``n|N``, no normalization
                is performed regardless of the setting of ``norm`` (cf. ``ConvLayer``).
            skip: Type(s) of shortcut connections (cf. ``ResidualUnit``).
            residual_pre_conv: Type of pre-convolution when residual unit is a bottleneck block.
                The kernel size is set to 1 if "conv1", and ``kernel_size`` otherwise.
            residual_post_conv: Type of post-convolution when residual unit is a bottleneck block.
            conv_layer: Type or callable used to create convolutional layers (cf. ``pre_conv``).
            deconv_layer: Type or callable used to create upsampling layers.
            resnet_block: Type or callable used to create residual blocks.
            input_layer: Type or callable used to create initial layer which receives the input tensor.
            output_layer: Type or callable used to create an output layer (head). If ``None`` and
                ``num_classes`` is specified, a default ``classification_head`` is added.

        """
        super().__init__()
        padding_mode = PaddingMode(padding_mode)
        order = order.upper()
        if "C" not in order:
            raise ValueError("ResNet() 'order' must contain 'C' for convolution")
        if isinstance(acti, str):
            nonlinearity = acti
        elif isinstance(acti, Sequence) and len(acti) == 2:
            nonlinearity = acti[0]
            if not isinstance(nonlinearity, str):
                raise TypeError("ResNet() 'acti[0]' must be str")
        else:
            raise ValueError("ResNet() 'acti' must be str or 2-tuple")
        self.nonlinearity = nonlinearity  # used by self.init_conv_modules()
        # Number of residual blocks determined by longest sequence (channels, blocks, stride)
        if isinstance(num_channels, int):
            num_channels = (num_channels,)
        if isinstance(num_blocks, int):
            num_blocks = (num_blocks,)
        if isinstance(stride, (float, int)):
            stride = (stride,)
        # Initial layer (optional)
        if input_layer is None:
            channels = in_channels
        else:
            channels = max(1, int(num_channels[0] * expansion + 0.5))
            layer = input_layer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                bias=bias,
                norm=norm,
                acti=acti,
                order=order,
            )
            self.add_sublayer(name="input_layer", sublayer=layer)
        # Cascade of residual blocks
        for i, (m, b, s) in enumerate(zip_longest_repeat_last(num_channels, num_blocks, stride)):
            if m < 1:
                raise ValueError(f"ResNet() 'num_channels' must be positive, got {m}")
            n = max(1, int(m * expansion + 0.5))
            deconv = None
            if s < 1:
                deconv_in_channels = n if post_deconv else channels
                deconv_out_channels = deconv_in_channels
                if pre_conv and not post_deconv:
                    deconv_out_channels = n
                    channels = n
                deconv = deconv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=deconv_in_channels,
                    out_channels=deconv_out_channels,
                    scale_factor=1 / s,
                    bias=bias,
                )
                if not post_deconv:
                    self.add_sublayer(name=f"deconv_{i}", sublayer=deconv)
                s = 1
            with_pre_conv = pre_conv and (i > 0 or input_layer is None)
            with_pre_conv = with_pre_conv or recursive and (s != 1 or channels != n)
            with_pre_conv = with_pre_conv and (deconv is None or post_deconv)
            if with_pre_conv:
                is_first = input_layer is None and i == 0
                conv = conv_layer(
                    level=i,
                    is_first=is_first,
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=n,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    stride=s,
                    bias=bias,
                    norm=norm,
                    acti=acti,
                    order=order,
                )
                self.add_sublayer(name=f"conv_{i}", sublayer=conv)
                channels = n
                s = 1
            block = None
            for j in range(b):
                block = resnet_block(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=n,
                    num_channels=m,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    pre_conv=residual_pre_conv,
                    post_conv=residual_post_conv,
                    padding_mode=padding_mode,
                    stride=s,
                    bias=bias,
                    norm=norm,
                    acti=acti,
                    order=order,
                    skip=skip,
                    other=block if recursive else None,
                )
                channels = n
                s = 1
                self.add_sublayer(name=f"block_{i}_{j}", sublayer=block)
            if deconv is not None and post_deconv:
                self.add_sublayer(name=f"deconv_{i}", sublayer=deconv)
        # Optional output layer (e.g., classification head)
        if num_classes and output_layer is None:
            output_layer = classification_head
        if output_layer is not None:
            head_kwargs = dict(
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                bias=bias,
                norm=norm,
                acti=acti,
                order=order,
            )
            if num_classes:
                head_kwargs["num_classes"] = num_classes
            head = output_layer(spatial_dims, channels, **head_kwargs)
            self.add_sublayer(name="output_layer", sublayer=head)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = channels
        self.num_classes = num_classes or None

    @classmethod
    def from_config(cls: Type[ResNet], config: ResNetConfig) -> ResNet:
        return cls(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            num_channels=config.num_channels,
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            kernel_size=config.kernel_size,
            stride=config.stride,
            expansion=config.expansion,
            padding_mode=config.padding_mode,
            recursive=config.recursive,
            pre_conv=config.pre_conv,
            post_deconv=config.post_deconv,
            bias=config.bias,
            norm=config.norm,
            acti=config.acti,
            order=config.order,
            skip=config.skip,
            residual_pre_conv=config.residual_pre_conv,
            residual_post_conv=config.residual_post_conv,
        )

    @classmethod
    def from_dict(cls: Type[ResNet], config: Mapping[str, Any]) -> ResNet:
        config = ResNetConfig.from_dict(config)
        return cls.from_config(config)

    @classmethod
    def from_depth(
        cls: Type[ResNet], model_depth: int, spatial_dims: int, in_channels: int = 1, **kwargs
    ) -> ResNet:
        config = ResNetConfig.from_depth(model_depth, spatial_dims, in_channels, **kwargs)
        return cls.from_config(config)

    def init_conv_modules(self) -> ResNet:
        r"""Initialize parameters of convolutions."""
        init_conv_modules(self, nonlinearity=self.nonlinearity)
        return self

    def init_norm_layers(self) -> ResNet:
        r"""Initialize normalization layer weights and biases."""
        init_norm_layers(self)
        return self

    def zero_init_residuals(self) -> ResNet:
        r"""Zero-initialize the last normalization layer in each residual branch."""
        zero_init_residuals(self)
        return self


def init_conv_modules(network: paddle.nn.Layer, nonlinearity: str = "relu") -> paddle.nn.Layer:
    r"""Initialize parameters of convolutions."""
    # https://github.com/pytorch/vision/blob/5905de086dcebff606f9110d15ebd3d1de18fc2e/torchvision/models/resnet.py#L189-L190
    nonlinearity = nonlinearity.lower()
    if nonlinearity == "lrelu":
        nonlinearity = "leaky_relu"
    if nonlinearity in ("relu", "leaky_relu"):
        for module in network.sublayers():
            if is_convolution(module):
                paddlenlp.utils.initializer.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity=nonlinearity
                )
                if module.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(module.bias)
    return network


def init_norm_layers(network: paddle.nn.Layer) -> paddle.nn.Layer:
    r"""Initialize batch and group norm layers weights to one and biases to zero."""
    # https://github.com/pytorch/vision/blob/5905de086dcebff606f9110d15ebd3d1de18fc2e/torchvision/models/resnet.py#L191-L193
    for module in network.sublayers():
        if is_batch_norm(module) or is_group_norm(module):
            init_Constant = paddle.nn.initializer.Constant(value=1)
            init_Constant(module.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(module.bias)
    return network


def zero_init_residuals(network: paddle.nn.Layer) -> paddle.nn.Layer:
    r"""Zero-initialize the last normalization layer in each residual branch."""
    # https://github.com/pytorch/vision/blob/5905de086dcebff606f9110d15ebd3d1de18fc2e/torchvision/models/resnet.py#L195-L203
    for module in network.sublayers():
        if isinstance(module, ResidualUnit):
            module.zero_init_residual()
    return network
