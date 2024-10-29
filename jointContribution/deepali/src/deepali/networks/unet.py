r"""U-net model architectures."""
from __future__ import annotations  # noqa

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import paddle
from deepali.core.config import DataclassConfig
from deepali.core.enum import PaddingMode
from deepali.core.image import crop
from deepali.core.itertools import repeat_last
from deepali.core.nnutils import as_immutable_container
from deepali.core.typing import ListOrTuple
from deepali.core.typing import ScalarOrTuple
from deepali.modules import GetItem
from deepali.modules import ReprWithCrossReferences
from paddle import Tensor
from paddle.nn import Identity

from .blocks import ResidualUnit
from .layers import ActivationArg
from .layers import ConvLayer
from .layers import JoinLayer
from .layers import NormArg
from .layers import PoolLayer
from .layers import Upsample
from .layers import UpsampleMode
from .utils import module_output_size

__all__ = (
    "SequentialUNet",
    "UNet",
    "UNetConfig",
    "UNetDecoder",
    "UNetDecoderConfig",
    "UNetDownsampleConfig",
    "UNetEncoder",
    "UNetEncoderConfig",
    "UNetLayerConfig",
    "UNetOutputConfig",
    "UNetUpsampleConfig",
    "unet_conv_block",
)
ModuleFactory = Union[Callable[..., paddle.nn.Layer], Type[paddle.nn.Layer]]
NumChannels = ListOrTuple[Union[int, Sequence[int]]]
NumBlocks = Union[int, Sequence[int]]
NumLayers = Optional[Union[int, Sequence[int]]]


def reversed_num_channels(num_channels: NumChannels) -> NumChannels:
    r"""Reverse order of per-block/-stage number of feature channels."""
    rev_channels = tuple(
        tuple(reversed(c)) if isinstance(c, Sequence) else c for c in reversed(num_channels)
    )
    return rev_channels


def decoder_num_channels_from_encoder_num_channels(num_channels: NumChannels) -> NumChannels:
    r"""Get default UNetDecoderConfig.num_channels from UNetEncoderConfig.num_channels."""
    num_channels = list(reversed_num_channels(num_channels))
    if isinstance(num_channels[0], Sequence):
        num_channels[0] = num_channels[0][0]
    return tuple(num_channels)


def first_num_channels(num_channels: NumChannels) -> int:
    r"""Get number of feature channels of first block."""
    nc = num_channels[0]
    if isinstance(nc, Sequence):
        if not nc:
            raise ValueError("first_num_channels() 'num_channels[0]' must not be empty")
        nc = nc[0]
    return nc


def last_num_channels(num_channels: NumChannels) -> int:
    r"""Get number of feature channels of last block."""
    nc = num_channels[-1]
    if isinstance(nc, Sequence):
        if not nc:
            raise ValueError("last_num_channels() 'num_channels[-1]' must not be empty")
        nc = nc[-1]
    return nc


@dataclass
class UNetLayerConfig(DataclassConfig):
    kernel_size: ScalarOrTuple[int] = 3
    dilation: ScalarOrTuple[int] = 1
    padding: Optional[ScalarOrTuple[int]] = None
    padding_mode: Union[PaddingMode, str] = "zeros"
    init: str = "default"
    bias: Union[str, bool, None] = None
    norm: NormArg = "instance"
    acti: ActivationArg = "lrelu"
    order: str = "cna"

    def __post_init__(self):
        self._join_kwargs_in_sequence("acti")
        self._join_kwargs_in_sequence("norm")


@dataclass
class UNetDownsampleConfig(DataclassConfig):
    mode: str = "conv"
    factor: Union[int, Sequence[int]] = 2
    kernel_size: Optional[ScalarOrTuple[int]] = None
    padding: Optional[ScalarOrTuple[int]] = None


@dataclass
class UNetUpsampleConfig(DataclassConfig):
    mode: Union[str, UpsampleMode] = "deconv"
    factor: Union[int, Sequence[int]] = 2
    kernel_size: Optional[ScalarOrTuple[int]] = None
    dilation: Optional[ScalarOrTuple[int]] = None
    padding: Optional[ScalarOrTuple[int]] = None


@dataclass
class UNetOutputConfig(DataclassConfig):
    channels: int = 1
    kernel_size: int = 1
    dilation: int = 1
    padding: Optional[int] = None
    padding_mode: Union[PaddingMode, str] = "zeros"
    init: str = "default"
    bias: Union[str, bool, None] = False
    norm: NormArg = None
    acti: ActivationArg = None
    order: str = "cna"

    def __post_init__(self):
        self._join_kwargs_in_sequence("acti")
        self._join_kwargs_in_sequence("norm")


@dataclass
class UNetEncoderConfig(DataclassConfig):
    num_channels: NumChannels = (8, 16, 32, 64)
    num_blocks: NumBlocks = 2
    num_layers: NumLayers = None
    conv_layer: UNetLayerConfig = field(default_factory=UNetLayerConfig)
    downsample: Union[str, UNetDownsampleConfig] = field(default_factory=UNetDownsampleConfig)
    residual: bool = False

    # When paddle.backends.cudnn.deterministic == True, then a dilated convolution at layer_1_2
    # causes a "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`".
    # This can be resolved by either setting paddle.backends.cudnn.deterministic = False, or
    # by not using a dilated convolution for layer_1_2. See also related GitHub issue reported
    # at https://github.com/pytorch/pytorch/issues/32035.

    # dilation different from conv_layer.dilation for first block in each stage
    block_1_dilation: Optional[int] = None
    # dilation different from conv_layer.dilation for first stage
    stage_1_dilation: Optional[int] = None

    @property
    def num_levels(self) -> int:
        r"""Number of spatial encoder levels."""
        return len(self.num_channels)

    @property
    def out_channels(self) -> int:
        return last_num_channels(self.num_channels)

    def __post_init__(self):
        if isinstance(self.downsample, str):
            self.downsample = UNetDownsampleConfig(self.downsample)


@dataclass
class UNetDecoderConfig(DataclassConfig):
    num_channels: NumChannels = (64, 32, 16, 8)
    num_blocks: NumBlocks = 2
    num_layers: NumLayers = None
    conv_layer: UNetLayerConfig = field(default_factory=UNetLayerConfig)
    upsample: Union[str, UNetUpsampleConfig] = field(default_factory=UNetUpsampleConfig)
    join_mode: str = "cat"
    crop_skip: bool = False
    residual: bool = False

    @property
    def num_levels(self) -> int:
        r"""Number of spatial decoder levels, including bottleneck input."""
        return len(self.num_channels)

    @property
    def in_channels(self) -> int:
        return first_num_channels(self.num_channels)

    @property
    def out_channels(self) -> int:
        return last_num_channels(self.num_channels)

    def __post_init__(self):
        if isinstance(self.upsample, (str, UpsampleMode)):
            self.upsample = UNetUpsampleConfig(self.upsample)

    @classmethod
    def from_encoder(
        cls,
        encoder: Union[UNetEncoder, UNetEncoderConfig],
        residual: Optional[bool] = None,
        **kwargs,
    ) -> UNetDecoderConfig:
        r"""Derive decoder configuration from U-net encoder configuration."""
        if isinstance(encoder, UNetEncoder):
            encoder = encoder.config
        if not isinstance(encoder, UNetEncoderConfig):
            raise TypeError(
                f"{cls.__name__}.from_encoder() argument must be UNetEncoder or UNetEncoderConfig"
            )
        if encoder.num_levels < 2:
            raise ValueError(f"{cls.__name__}.from_encoder() encoder must have at least two levels")
        if "upsample_mode" in kwargs:
            if "upsample" in kwargs:
                raise ValueError(
                    f"{cls.__name__}.from_encoder() 'upsample' and 'upsample_mode' are mutually exclusive"
                )
            kwargs["upsample"] = UNetUpsampleConfig(kwargs.pop("upsample_mode"))
        residual = encoder.residual if residual is None else residual
        num_channels = decoder_num_channels_from_encoder_num_channels(encoder.num_channels)
        num_blocks = encoder.num_blocks
        if isinstance(num_blocks, Sequence):
            num_blocks = tuple(reversed(repeat_last(num_blocks, encoder.num_levels)))
        num_layers = encoder.num_layers
        if isinstance(num_layers, Sequence):
            num_layers = tuple(reversed(repeat_last(num_layers, encoder.num_levels)))
        return cls(
            num_channels=num_channels,
            num_blocks=num_blocks,
            num_layers=num_layers,
            conv_layer=encoder.conv_layer,
            residual=residual,
            **kwargs,
        )


@dataclass
class UNetConfig(DataclassConfig):
    encoder: UNetEncoderConfig = field(default_factory=UNetEncoderConfig)
    decoder: Optional[UNetDecoderConfig] = None
    output: Optional[UNetOutputConfig] = None

    def __post_init__(self):
        if self.decoder is None:
            self.decoder = UNetDecoderConfig.from_encoder(self.encoder)


def unet_conv_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: ScalarOrTuple[int] = 3,
    stride: ScalarOrTuple[int] = 1,
    padding: Optional[ScalarOrTuple[int]] = None,
    padding_mode: Union[PaddingMode, str] = "zeros",
    dilation: ScalarOrTuple[int] = 1,
    groups: int = 1,
    init: str = "default",
    bias: Optional[Union[bool, str]] = None,
    norm: NormArg = None,
    acti: ActivationArg = None,
    order: str = "CNA",
    num_layers: Optional[int] = None,
) -> paddle.nn.Layer:
    r"""Create U-net block of convolutional layers."""
    if num_layers is None:
        num_layers = 1
    elif num_layers < 1:
        raise ValueError("unet_conv_block() 'num_layers' must be positive")

    def conv_layer(m: int, n: int, s: int, d: int) -> ConvLayer:
        return ConvLayer(
            spatial_dims=spatial_dims,
            in_channels=m,
            out_channels=n,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=s,
            dilation=d,
            groups=groups,
            init=init,
            bias=bias,
            norm=norm,
            acti=acti,
            order=order,
        )

    block = paddle.nn.Sequential()
    for i in range(num_layers):
        m = in_channels if i == 0 else out_channels
        n = out_channels
        s = stride if i == 0 else 1
        d = dilation if s == 1 else 1
        conv = conv_layer(m, n, s, d)
        block.add_sublayer(name=f"layer_{i + 1}", sublayer=conv)
    return block


class UNetEncoder(ReprWithCrossReferences, paddle.nn.Layer):
    r"""Downsampling path of U-net model."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: Optional[int] = None,
        config: Optional[UNetEncoderConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        input_layer: Optional[ModuleFactory] = None,
    ):
        super().__init__()

        if config is None:
            config = UNetEncoderConfig()
        elif not isinstance(config, UNetEncoderConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetEncoderConfig")
        if config.num_levels < 2:
            raise ValueError(
                f"{type(self).__name__} U-net must have at least two spatial resolution levels"
            )

        if config.downsample.mode == "none":
            down_stride = (1,) * config.num_levels
        if isinstance(config.downsample.factor, int):
            down_stride = (1,) + (config.downsample.factor,) * (config.num_levels - 1)
        else:
            down_stride = repeat_last(config.downsample.factor, config.num_levels)

        if conv_block is None:
            conv_block = ResidualUnit if config.residual else unet_conv_block
        elif not isinstance(conv_block, paddle.nn.Layer) and not callable(conv_block):
            raise TypeError(f"{type(self).__name__}() 'conv_block' must be Module or callable")

        num_blocks = repeat_last(config.num_blocks, config.num_levels)
        num_layers = repeat_last(config.num_layers, config.num_levels)

        num_channels = list(config.num_channels)
        channels = first_num_channels(num_channels)
        for i, (s, b, nc) in enumerate(zip(down_stride, num_blocks, num_channels)):
            if not isinstance(b, int):
                raise TypeError(f"{type(self).__name__} 'num_blocks' must be int or Sequence[int]")
            if b < 1:
                raise ValueError(f"{type(self).__name__} 'num_blocks' must be positive")
            if isinstance(nc, int):
                nc = (nc,) * b
                if s > 1:
                    if config.downsample.mode == "conv":
                        nc = (nc[0],) + nc
                    else:
                        nc = (channels,) + nc
                else:
                    nc = (nc[0],) + nc
            elif not isinstance(nc, Sequence):
                raise TypeError(
                    f"{type(self).__name__}() 'num_channels' values must be int or Sequence[int]"
                )
            if not nc:
                raise ValueError(
                    f"{type(self).__name__}() 'num_channels' must not contain empty sequence"
                )
            num_channels[i] = list(nc)
            channels = nc[-1]
        num_channels: List[List[int]] = list(list(nc) for nc in num_channels)
        channels = num_channels[0][0]

        if in_channels is None:
            in_channels = channels
        if input_layer is None:
            input_layer = ConvLayer if in_channels != channels else Identity
        elif not isinstance(input_layer, paddle.nn.Layer) and not callable(input_layer):
            raise TypeError(f"{type(self).__name__}() 'input_layer' must be Module or callable")

        stages = paddle.nn.LayerDict()
        channels = in_channels
        for i, (s, l, nc) in enumerate(zip(down_stride, num_layers, num_channels)):
            assert isinstance(nc, Sequence) and len(nc) > 0
            stage = paddle.nn.LayerDict()
            if s > 1:
                if config.downsample.mode == "conv":
                    c = nc[0]
                    k = config.downsample.kernel_size or config.conv_layer.kernel_size
                    if config.downsample.padding is None:
                        p = config.conv_layer.padding
                    else:
                        p = config.downsample.padding
                    d = 0
                    if i == 0:
                        d = config.stage_1_dilation
                    d = d or config.conv_layer.dilation
                    downsample = conv_block(
                        spatial_dims=spatial_dims,
                        in_channels=channels,
                        out_channels=c,
                        kernel_size=k,
                        stride=s,
                        dilation=d,
                        padding=p,
                        padding_mode=config.conv_layer.padding_mode,
                        init=config.conv_layer.init,
                        bias=config.conv_layer.bias,
                        norm=config.conv_layer.norm,
                        acti=config.conv_layer.acti,
                        order=config.conv_layer.order,
                        num_layers=l,
                    )
                    channels = c
                else:
                    if nc[0] != channels:
                        raise ValueError(
                            f"{type(self).__name__}() number of input channels of stage after pooling ({nc[0]}) must match number of channels of previous stage ({channels})"
                        )
                    pool_size = config.downsample.kernel_size or s
                    pool_args = dict(kernel_size=pool_size, stride=s)
                    if pool_size % 2 == 0:
                        pool_args["padding"] = pool_size // 2 - 1
                    else:
                        pool_args["padding"] = pool_size // 2
                    if config.downsample.mode == "avg":
                        pool_args["count_include_pad"] = False
                    downsample = PoolLayer(
                        config.downsample.mode, spatial_dims=spatial_dims, **pool_args
                    )
                stage["downsample"] = downsample
                s = 1
            elif i == 0:
                c = nc[0]
                stage["input"] = input_layer(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=c,
                    kernel_size=config.conv_layer.kernel_size,
                    dilation=config.stage_1_dilation or config.conv_layer.dilation,
                    padding=config.conv_layer.padding,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                )
                channels = c
            blocks = paddle.nn.Sequential()
            for j, c in enumerate(nc[1:]):
                d = 0
                if j == 0:
                    d = config.block_1_dilation
                if i == 0:
                    d = d or config.stage_1_dilation
                d = d or config.conv_layer.dilation
                block = conv_block(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=c,
                    kernel_size=config.conv_layer.kernel_size,
                    stride=1,
                    dilation=d,
                    padding=config.conv_layer.padding,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                    num_layers=l,
                )
                blocks.add_sublayer(name=f"block_{j + 1}", sublayer=block)
                channels = c
                s = 1
            # mirror full module names of UNetDecoder, e.g.,
            # encoder.stages.stage_1.blocks.block_1.layer_1.conv
            stage["blocks"] = blocks
            stages[f"stage_{i + 1}"] = stage

        config = deepcopy(config)
        config.num_channels = num_channels

        self.config = config
        self.num_channels: List[List[int]] = num_channels
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.stages = stages

    @property
    def out_channels(self) -> int:
        return self.num_channels[-1][-1]

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        r"""Calculate output size of last feature map for a tensor of given spatial input size."""
        return self.output_sizes(in_size)[-1]

    def output_sizes(self, in_size: ScalarOrTuple[int]) -> List[ScalarOrTuple[int]]:
        r"""Calculate output sizes of feature maps for a tensor of given spatial input size."""
        size = in_size
        fm_sizes = []
        for name, stage in self.stages.items():
            assert isinstance(stage, paddle.nn.LayerDict)
            if name == "stage_1":
                size = module_output_size(stage["input"], size)
            if "downsample" in stage:
                size = module_output_size(stage["downsample"], size)
            size = module_output_size(stage["blocks"], size)
            fm_sizes.append(size)
        return fm_sizes

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, ...]:
        features = []
        for name, stage in self.stages.items():
            if not isinstance(stage, paddle.nn.LayerDict):
                raise AssertionError(
                    f"{type(self).__name__}.forward() expected stage ModuleDict, got {type(stage)}"
                )
            if name == "stage_1":
                input_layer = stage["input"]
                x = input_layer(x)
            if "downsample" in stage:
                downsample = stage["downsample"]
                x = downsample(x)
            blocks = stage["blocks"]
            x = blocks(x)
            features.append(x)
        return tuple(features)


class UNetDecoder(ReprWithCrossReferences, paddle.nn.Layer):
    r"""Upsampling path of U-net model."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: Optional[int] = None,
        config: Optional[UNetDecoderConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        input_layer: Optional[ModuleFactory] = None,
        output_all: bool = False,
    ) -> None:
        super().__init__()

        if config is None:
            config = UNetDecoderConfig()
        elif not isinstance(config, UNetDecoderConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetDecoderConfig")
        if config.num_levels < 2:
            raise ValueError(
                f"{type(self).__name__} U-net must have at least two spatial resolution levels"
            )

        if not isinstance(config.num_channels, Sequence):
            raise TypeError(f"{type(self).__name__}() 'config.num_channels' must be Sequence")
        if any(isinstance(nc, Sequence) and not nc for nc in config.num_channels):
            raise ValueError(
                f"{type(self).__name__}() 'config.num_channels' contains empty sequence"
            )

        num_blocks = repeat_last(config.num_blocks, config.num_levels)
        num_layers = repeat_last(config.num_layers, config.num_levels)
        scale_factor = repeat_last(config.upsample.factor, config.num_levels - 1)
        upsample_mode = UpsampleMode(config.upsample.mode)
        join_mode = config.join_mode

        num_channels = list(config.num_channels)
        for i, (b, nc) in enumerate(zip(num_blocks, num_channels)):
            if not isinstance(b, int):
                raise TypeError(f"{type(self).__name__} 'num_blocks' must be int or Sequence[int]")
            if b < 1:
                raise ValueError(f"{type(self).__name__} 'num_blocks' must be positive")
            if isinstance(nc, int):
                nc = (nc,) * (1 if i == 0 else b + 1)
            elif isinstance(nc, Sequence):
                nc = list(nc)
            else:
                raise TypeError(
                    f"{type(self).__name__}() 'num_channels' values must be int or Sequence[int]"
                )
            if not nc:
                raise ValueError(
                    f"{type(self).__name__}() 'num_channels' must not contain empty sequence"
                )
            num_channels[i] = list(nc)
        if upsample_mode is UpsampleMode.INTERPOLATE and config.upsample.kernel_size == 0:
            for i, nc in enumerate(num_channels[:-1]):
                next_nc = num_channels[i + 1][0]
                assert isinstance(nc, List) and len(nc) > 0
                if isinstance(config.num_channels[i], int):
                    if len(nc) == 1:
                        nc.append(next_nc)
                    else:
                        nc[-1] = next_nc
                elif nc[-1] != next_nc:
                    raise ValueError(
                        f"{type(self).__name__}() 'num_channels' of last feature map in previous stage ({num_channels[i]}) must match number of channels of first feature map ({next_nc}) of next stage when upsampling mode is interpolation without preconv. Either adjust 'num_channels' or use non-zero 'upsample.kernel_size'."
                    )
        channels = num_channels[0][0]

        if conv_block is None:
            conv_block = ResidualUnit if config.residual else unet_conv_block
        elif not isinstance(conv_block, paddle.nn.Layer) and not callable(conv_block):
            raise TypeError(f"{type(self).__name__}() 'conv_block' must be Module or callable")

        if in_channels is None:
            in_channels = channels
        if input_layer is None:
            input_layer = ConvLayer if in_channels != channels else Identity
        elif not isinstance(input_layer, paddle.nn.Layer) and not callable(input_layer):
            raise TypeError(f"{type(self).__name__}() 'input_layer' must be Module or callable")

        stages = paddle.nn.LayerDict()

        stage = paddle.nn.LayerDict()
        stage["input"] = input_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=config.conv_layer.kernel_size,
            dilation=config.conv_layer.dilation,
            padding=config.conv_layer.padding,
            padding_mode=config.conv_layer.padding_mode,
            init=config.conv_layer.init,
            bias=config.conv_layer.bias,
            norm=config.conv_layer.norm,
            acti=config.conv_layer.acti,
            order=config.conv_layer.order,
        )
        blocks = paddle.nn.Sequential()
        for j, c in enumerate(num_channels[0][1:]):
            block = conv_block(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=c,
                kernel_size=config.conv_layer.kernel_size,
                dilation=config.conv_layer.dilation,
                padding=config.conv_layer.padding,
                padding_mode=config.conv_layer.padding_mode,
                init=config.conv_layer.init,
                bias=config.conv_layer.bias,
                norm=config.conv_layer.norm,
                acti=config.conv_layer.acti,
                order=config.conv_layer.order,
                num_layers=num_layers[0],
            )
            blocks.add_sublayer(name=f"block_{j + 1}", sublayer=block)
            channels = c
        stage["blocks"] = blocks
        stages["stage_1"] = stage

        for i, (s, l, nc) in enumerate(zip(scale_factor, num_layers[1:], num_channels[1:])):
            assert isinstance(nc, Sequence) and len(nc) > 1
            stage = paddle.nn.LayerDict()
            if upsample_mode is UpsampleMode.INTERPOLATE and config.upsample.kernel_size != 0:
                p = config.upsample.padding
                if p is None:
                    p = config.conv_layer.padding
                k = config.upsample.kernel_size
                if k is None:
                    k = config.conv_layer.kernel_size
                d = config.upsample.dilation or config.conv_layer.dilation
                pre_conv = ConvLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=nc[0],
                    kernel_size=k,
                    dilation=d,
                    padding=p,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                )
            else:
                pre_conv = "default"
            if s > 1:
                upsample = Upsample(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=nc[0],
                    scale_factor=s,
                    mode=upsample_mode,
                    align_corners=False,
                    pre_conv=pre_conv,
                    kernel_size=config.upsample.kernel_size,
                    padding_mode=config.conv_layer.padding_mode,
                    bias=True if config.conv_layer.bias is None else config.conv_layer.bias,
                )
                stage["upsample"] = upsample
            stage["join"] = JoinLayer(join_mode, dim=1)
            channels = (2 if join_mode == "cat" else 1) * nc[0]
            blocks = paddle.nn.Sequential()
            for j, c in enumerate(nc[1:]):
                block = conv_block(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=c,
                    kernel_size=config.conv_layer.kernel_size,
                    dilation=config.conv_layer.dilation,
                    padding=config.conv_layer.padding,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                    num_layers=l,
                )
                blocks.add_sublayer(name=f"block_{j + 1}", sublayer=block)
                channels = c
            stage["blocks"] = blocks
            stages[f"stage_{i + 2}"] = stage

        config = deepcopy(config)
        config.num_channels = num_channels

        self.config = config
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels: List[List[int]] = num_channels
        self.stages = stages
        self.output_all = output_all

    @classmethod
    def from_encoder(
        cls,
        encoder: Union[UNetEncoder, UNetEncoderConfig],
        residual: Optional[bool] = None,
        **kwargs,
    ) -> UNetDecoder:
        r"""Create U-net decoder given U-net encoder configuration."""
        config = UNetDecoderConfig.from_encoder(encoder, residual=residual, **kwargs)
        return cls(spatial_dims=encoder.spatial_dims, config=config)

    @property
    def out_channels(self) -> int:
        return self.num_channels[-1][-1]

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        r"""Calculate output size for an initial feature map of given spatial input size."""
        return self.output_sizes(in_size)[-1]

    def output_sizes(self, in_size: ScalarOrTuple[int]) -> List[ScalarOrTuple[int]]:
        r"""Calculate output sizes for an initial feature map of given spatial input size."""
        size = in_size
        out_sizes = []
        for name, stage in self.stages.items():
            if not isinstance(stage, paddle.nn.LayerDict):
                raise AssertionError(
                    f"{type(self).__name__}.out_sizes() expected stage ModuleDict, got {type(stage)}"
                )
            if name == "stage_1":
                size = module_output_size(stage["input"], size)
            if "upsample" in stage:
                size = module_output_size(stage["upsample"], size)
            size = module_output_size(stage["blocks"], size)
            out_sizes.append(size)
        return out_sizes

    def forward(
        self, features: Sequence[paddle.Tensor]
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
        if not isinstance(features, Sequence):
            raise TypeError(f"{type(self).__name__}() 'features' must be Sequence")
        features = list(features)
        if len(features) != len(self.stages):
            raise ValueError(
                f"{type(self).__name__}() 'features' must contain {len(self.stages)} tensors"
            )
        x: Tensor = features.pop()
        output: List[paddle.Tensor] = []
        for name, stage in self.stages.items():
            if not isinstance(stage, paddle.nn.LayerDict):
                raise AssertionError(
                    f"{type(self).__name__}.forward() expected stage ModuleDict, got {type(stage)}"
                )
            blocks = stage["blocks"]
            if name == "stage_1":
                input_layer = stage["input"]
                x = input_layer(x)
            else:
                skip = features.pop()
                upsample = stage["upsample"]
                join = stage["join"]
                x = upsample(x)
                if self.config.crop_skip:
                    margin = tuple(n - m for m, n in zip(tuple(x.shape)[2:], tuple(skip.shape)[2:]))
                    assert all(m >= 0 and m % 2 == 0 for m in margin)
                    margin = tuple(m // 2 for m in margin)
                    skip = crop(skip, margin=margin)
                x = join([x, skip])
                del skip
            x = blocks(x)
            if self.output_all:
                output.append(x)
        if self.output_all:
            return tuple(output)
        return x


class SequentialUNet(ReprWithCrossReferences, paddle.nn.Sequential):
    r"""Sequential U-net architecture.

    The final module of this sequential module either outputs a tuple of feature maps at the
    different resolution levels (``out_channels=None``), the final decoded feature map at the
    highest resolution level (``out_channels == config.decoder.out_channels`` and ``output_layers=None``),
    or a tensor with specified number of ``out_channels`` as produced by a final output layer otherwise.
    Note that additional layers (e.g., a custom output layer or post-output layers) can be added to the
    initialized sequential U-net using ``add_module()``.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        config: Optional[UNetConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        output_layer: Optional[ModuleFactory] = None,
        bridge_layer: Optional[ModuleFactory] = None,
    ) -> None:
        super().__init__()

        # Network configuration
        if config is None:
            config = UNetConfig()
        elif not isinstance(config, UNetConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetConfig")
        config = deepcopy(config)
        self.config = config

        # Downsampling path
        self.encoder = UNetEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            config=config.encoder,
            conv_block=conv_block,
        )
        in_channels = self.encoder.in_channels

        # Upsamling path with skip connections
        self.decoder = UNetDecoder(
            spatial_dims=spatial_dims,
            in_channels=self.encoder.out_channels,
            config=config.decoder,
            conv_block=conv_block,
            input_layer=bridge_layer,
            output_all=output_layer is None and not out_channels,
        )

        # Optional output layer
        channels = self.decoder.out_channels
        if not out_channels and config.output is not None:
            out_channels = config.output.channels
        if output_layer is None:
            if self.decoder.output_all:
                out_channels = self.decoder.num_channels
            elif out_channels:
                output_layer = ConvLayer
        if output_layer is not None:
            out_channels = out_channels or in_channels
            if config.output is None:
                config.output = UNetOutputConfig()
            output = output_layer(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=config.output.kernel_size,
                padding=config.output.padding,
                padding_mode=config.output.padding_mode,
                dilation=config.output.dilation,
                init=config.output.init,
                bias=config.output.bias,
                norm=config.output.norm,
                acti=config.output.acti,
                order=config.output.order,
            )
            self.add_sublayer(name="output", sublayer=output)
        self.out_channels: Union[int, List[int]] = out_channels

    @property
    def spatial_dims(self) -> int:
        return self.encoder.spatial_dims

    @property
    def in_channels(self) -> int:
        return self.encoder.in_channels

    @property
    def num_channels(self) -> List[List[int]]:
        return self.decoder.num_channels

    @property
    def num_levels(self) -> int:
        return len(self.num_channels)

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        r"""Calculate spatial output size given an input tensor with specified spatial size."""
        # ATTENTION: module_output_size(self, in_size) would cause an infinite recursion!
        size = in_size
        for module in self:
            size = module_output_size(module, size)
        return size


class UNet(ReprWithCrossReferences, paddle.nn.Layer):
    r"""U-net with optionally multiple output layers."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        output_modules: Optional[Mapping[str, paddle.nn.Layer]] = None,
        output_indices: Optional[Union[Mapping[str, int], int]] = None,
        config: Optional[UNetConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        bridge_layer: Optional[ModuleFactory] = None,
        output_layer: Optional[ModuleFactory] = None,
        output_name: str = "output",
    ) -> None:
        super().__init__()

        if output_modules is None:
            output_modules = {}
        if not isinstance(output_modules, Mapping):
            raise TypeError(f"{type(self).__name__}() 'output_modules' must be Mapping")

        # Network configuration
        if config is None:
            config = UNetConfig()
        elif not isinstance(config, UNetConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetConfig")
        config = deepcopy(config)
        self.config = config

        # Downsampling path
        self.encoder = UNetEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            config=config.encoder,
            conv_block=conv_block,
        )
        in_channels = self.encoder.in_channels

        # Upsamling path with skip connections
        self.decoder = UNetDecoder(
            spatial_dims=spatial_dims,
            in_channels=self.encoder.out_channels,
            config=config.decoder,
            conv_block=conv_block,
            input_layer=bridge_layer,
            output_all=True,
        )

        # Optional output layer
        channels = self.decoder.out_channels
        self.output_modules = paddle.nn.LayerDict()
        if not out_channels and config.output is not None:
            out_channels = config.output.channels
        if output_layer is None:
            if out_channels == channels and config.output is None:
                self.output_modules[output_name] = GetItem(-1)
            elif out_channels:
                output_layer = ConvLayer
            elif not output_modules:
                out_channels = self.decoder.num_channels
        if output_layer is not None:
            out_channels = out_channels or in_channels
            if config.output is None:
                config.output = UNetOutputConfig()
            output = output_layer(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=config.output.kernel_size,
                padding=config.output.padding,
                padding_mode=config.output.padding_mode,
                dilation=config.output.dilation,
                init=config.output.init,
                bias=config.output.bias,
                norm=config.output.norm,
                acti=config.output.acti,
                order=config.output.order,
            )
            output = [("input", GetItem(-1)), ("layer", output)]
            output = paddle.nn.Sequential(*output)
            self.output_modules[output_name] = output
        self.out_channels: Union[int, Sequence[int], None] = out_channels

        # Additional output layers
        if output_indices is None:
            output_indices = {}
        elif isinstance(output_indices, int):
            output_indices = {name: output_indices for name in output_modules}
        for name, output in output_modules.items():
            output_index = output_indices.get(name)
            if output_index is not None:
                if not isinstance(output_index, int):
                    raise TypeError(f"{type(self).__name__}() 'output_indices' must be int")
                output = [("input", GetItem(output_index)), ("layer", output)]
                output = paddle.nn.Sequential(*output)
            self.output_modules[name] = output

    @property
    def spatial_dims(self) -> int:
        return self.encoder.spatial_dims

    @property
    def in_channels(self) -> int:
        return self.encoder.in_channels

    @property
    def num_channels(self) -> List[List[int]]:
        return self.decoder.num_channels

    @property
    def num_levels(self) -> int:
        return len(self.num_channels)

    @property
    def num_output_layers(self) -> int:
        return len(self.output_modules)

    def output_names(self) -> Iterable[str]:
        return self.output_modules.keys()

    def output_is_dict(self) -> bool:
        r"""Whether model output is dictionary of output tensors."""
        return not (self.output_is_tensor() or self.output_is_tuple())

    def output_is_tensor(self) -> bool:
        r"""Whether model output is a single output tensor."""
        return len(self.output_modules) == 1 and bool(self.out_channels)

    def output_is_tuple(self) -> bool:
        r"""Whether model output is tuple of decoded feature maps."""
        return not self.output_modules

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        out_sizes = self.output_sizes(in_size)
        if self.output_is_tensor():
            assert len(out_sizes) == 1
            return out_sizes[0]
        if self.output_is_tuple():
            return out_sizes[-1]
        assert isinstance(out_sizes, dict) and len(out_sizes) > 0
        out_size = None
        for size in out_sizes.values():
            if out_size is None:
                out_size = size
            elif out_size != size:
                raise RuntimeError(
                    f"{type(self).__name__}.output_size() is ambiguous, use output_sizes() instead"
                )
        assert out_size is not None
        return out_size

    def output_sizes(
        self, in_size: ScalarOrTuple[int]
    ) -> Union[Dict[str, ScalarOrTuple[int]], List[ScalarOrTuple[int]]]:
        enc_out_size = self.encoder.output_size(in_size)
        dec_out_sizes = self.decoder.output_sizes(enc_out_size)
        if self.output_is_tensor():
            return dec_out_sizes[-1:]
        if self.output_is_tuple():
            return dec_out_sizes
        out_sizes = {}
        for name, module in self.output_modules.items():
            out_sizes[name] = module_output_size(module, dec_out_sizes)
        return out_sizes

    def forward(
        self, x: paddle.Tensor
    ) -> Union[paddle.Tensor, NamedTuple, Tuple[paddle.Tensor, ...]]:
        outputs = {}
        features = self.decoder(self.encoder(x))
        for name, output in self.output_modules.items():
            outputs[name] = output(features)
        if not outputs:
            return features
        if len(outputs) == 1 and self.out_channels:
            return next(iter(outputs.values()))
        return as_immutable_container(outputs)
