r"""Spatial transformer network (STN)."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import overload

import paddle
from deepali.core import Device
from deepali.core import PathStr
from deepali.core import ScalarOrTuple
from deepali.core import functional as U
from deepali.core.pathlib import unlink_or_mkdir
from deepali.networks.layers import Conv2d
from deepali.networks.layers import Conv3d
from deepali.networks.layers import ConvLayer
from deepali.networks.layers import Linear
from deepali.networks.layers import convolution
from deepali.networks.layers import pooling
from deepali.networks.utils import module_output_size
from deepali.spatial import TransformConfig
from deepali.spatial import has_affine_component
from deepali.spatial import has_nonrigid_component
from deepali.utils import paddle_aux  # noqa
from paddle import Tensor


@dataclass
class SpatialTransformerConfig(TransformConfig):
    r"""Hyperparameters of spatial transformer network (STN)."""

    batch_norm_enabled: bool = False
    bias_enabled: bool = True
    bias_init: str = "uniform"
    affine_max_angle: float = math.pi / 4
    affine_max_offset: float = 1
    affine_max_scale: float = 1
    affine_nodes: int = 32
    affine_init: str = "zeros"
    normalize_flow: bool = True
    stn_input_channels: list = None

    @staticmethod
    def section() -> str:
        r"""Common key prefix of configuration entries in configuration file."""
        return "model.stn"


class SpatialTransformEncoder(paddle.nn.Layer):
    r"""Image encoder of spatial transformer network."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        bias_enabled: bool = True,
        bias_init: str = "uniform",
    ) -> None:
        super().__init__()
        assert 2 <= spatial_dims <= 3
        acti = paddle.nn.ReLU()
        ks = 3

        def conv(m: int, n: int, **kwargs) -> paddle.nn.Layer:
            bias = bias_init if bias_enabled else False
            pad = ks // 2
            return convolution(spatial_dims, m, n, kernel_size=ks, padding=pad, bias=bias, **kwargs)

        blocks = [
            [
                ("conv_0", conv(in_channels, in_channels, stride=2)),
                ("conv_1", conv(in_channels, 16)),
                ("acti_0", acti),
                ("conv_2", conv(16, 16)),
                ("acti_1", acti),
            ],
            [("conv_0", conv(16, 16, stride=2)), ("conv_1", conv(16, 16)), ("acti", acti)],
            [("conv_0", conv(16, 32, stride=2)), ("conv_1", conv(32, 32)), ("acti", acti)],
            [("conv_0", conv(32, 32, stride=2)), ("conv_1", conv(32, 32)), ("acti", acti)],
        ]
        self.blocks = paddle.nn.LayerList(
            sublayers=[paddle.nn.Sequential(*block) for block in blocks]
        )

    @property
    def nblocks(self) -> int:
        return len(self.blocks)

    @property
    def spatial_dims(self) -> int:
        conv = self.blocks[0][0]
        if isinstance(conv, Conv2d):
            return 2
        elif isinstance(conv, Conv3d):
            return 3
        raise AssertionError(
            f"{type(self).__name__}.spatial_dims expected first convolution to be Conv2d or Conv3d"
        )

    @property
    def in_channels(self) -> int:
        r"""Number of expected input tensor channels."""
        return self.input_channels()

    @property
    def out_channels(self) -> int:
        r"""Number of channels of last output feature map."""
        return self.output_channels(-1)

    def input_channels(self) -> int:
        r"""Number of input channels."""
        conv = self.blocks[0][0]
        assert isinstance(conv, (Conv2d, Conv3d))
        return tuple(conv.weight.shape)[1]

    @overload
    def output_channels(self) -> List[int]:
        r"""Number of channels of output feature maps."""
        ...

    @overload
    def output_channels(self, index: int) -> int:
        r"""Number of channels of i-th output feature map."""
        ...

    def output_channels(self, index: Optional[int] = None) -> Union[int, List[int]]:
        r"""Number of channels of output feature map(s)."""
        if index is None:
            channels = []
            for block in self.blocks:
                conv = block[-2]
                assert isinstance(conv, (Conv2d, Conv3d))
                channels.append(tuple(conv.weight.shape)[0])
            return channels
        nblocks = self.nblocks
        if index < -nblocks or index >= nblocks:
            raise ValueError(
                f"{type(self).__name__}.output_channels() 'index' must be in [-{nblocks}, {nblocks - 1}]"
            )
        if index < 0:
            index += nblocks
        conv = self.blocks[index][-2]
        assert isinstance(conv, (Conv2d, Conv3d))
        return tuple(conv.weight.shape)[0]

    def output_size(self, in_size: ScalarOrTuple[int], index: int = -1) -> ScalarOrTuple[int]:
        r"""Spatial size of i-th output feature map."""
        nblocks = self.nblocks
        if index < -nblocks or index >= nblocks:
            raise ValueError(
                f"{type(self).__name__}.output_size() 'index' must be in [-{nblocks}, {nblocks - 1}]"
            )
        if index < 0:
            index += nblocks
        ds_factor = 2 ** (index + 1)
        if isinstance(in_size, int):
            return in_size // ds_factor
        return tuple(n // ds_factor for n in in_size)

    def output_sizes(self, in_size: ScalarOrTuple[int]) -> List[ScalarOrTuple[int]]:
        r"""Spatial size of output feature maps."""
        sizes = []
        ds_factor = 1
        for _ in self.blocks:
            ds_factor *= 2
            if isinstance(in_size, int):
                sizes.append(in_size // ds_factor)
            else:
                sizes.append(tuple(n // ds_factor for n in in_size))
        return sizes

    def output_shape(self, in_shape: Sequence[int], index: int = -1) -> list:
        r"""Shape of i-th output feature map."""
        in_channels = self.input_channels()
        if len(in_shape) != self.spatial_dims + 2 or in_shape[1] != in_channels:
            raise ValueError(
                f"{type(self).__name__}.output_shape() 'in_shape' must be of the form (N, {in_channels}, {'Y, X' if self.spatial_dims == 2 else 'Z, Y, X'})"
            )
        nblocks = self.nblocks
        if index < -nblocks or index >= nblocks:
            raise ValueError(
                f"{type(self).__name__}.output_shape() 'index' must be in [-{nblocks}, {nblocks - 1}]"
            )
        if index < 0:
            index += nblocks
        batch_size = in_shape[0]
        out_channels = self.output_channels(index)
        out_shape = tuple(reversed(self.output_size(in_shape[:1:-1])))
        out_shape = tuple((batch_size, out_channels)) + out_shape
        return out_shape

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, ...]:
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return tuple(out)


class AffineTransformDecoder(paddle.nn.Layer):
    r"""Decoder of affine transformation parameters."""

    def __init__(
        self,
        spatial_dims: int,
        features: int,
        hidden: int = 32,
        max_angle: float = math.pi / 4,
        max_offset: float = 1,
        max_scale: float = 1,
        as_matrix: bool = False,
        init: Optional[str] = "zeros",
    ) -> None:
        r"""Initialize affine transformation decoder.

        Args:
            spatial_dims: Number of spatial dimensions.
            features: Number of input features.
            hidden: Number of hidden units.
            max_angle: Maximum rotation angle with respect to unit cube domain.
            max_offset: Maximum translation with respect to unit cube domain.
            max_scale: Maximum scaling factor with respect to unit cube domain.
            as_matrix: Whether to compose and return transformation matrix.

        """
        super().__init__()
        if spatial_dims < 2 or spatial_dims > 3:
            raise ValueError(f"{type(self).__name__}() 'ndim' must be 2 or 3")
        self.spatial_dims = spatial_dims
        self.linear = paddle.nn.Sequential(Linear(features, hidden), paddle.nn.ReLU())
        self.offset = Linear(hidden, spatial_dims, bias=init, init=init)
        self.angles = Linear(hidden, 1 if spatial_dims == 2 else spatial_dims, bias=init, init=init)
        self.scales = Linear(hidden, spatial_dims, bias=init, init=init)
        self.max_angle = max_angle
        self.max_offset = max_offset
        self.max_scale_log = math.log(max_scale)
        self.as_matrix = as_matrix

    def forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        r"""Predict affine transformation parameters from encoded features.

        Args:
            x: Input feature vector as tensor of shape ``(N, M)``.

        Returns:
            When ``self.as_matrix == True``, dictionary with key 'affine' with value
            corresponding to the homogeneous coordinate transformation matrix as tensor
            of shape ``(N, D, D + 1)`` where ``D == self.spatial_dims``. Otherwise, dictionary
            keys 'angles', 'offset', and 'scales' contain the separately predicted affine
            transformation parameters.

        """
        x = self.linear(x)
        angles: Tensor = self.angles(x)
        offset: Tensor = self.offset(x)
        scales: Tensor = self.scales(x)
        angles = angles.tanh().mul(self.max_angle)
        offset = offset.tanh().mul(self.max_offset)
        scales = scales.tanh().mul(self.max_scale_log).exp()
        if not self.as_matrix:
            return dict(angles=angles, offset=offset, scales=scales)
        rotation = U.euler_rotation_matrix(angles, order="ZXZ")
        scaling = U.scaling_transform(scales)
        translation = U.translation(offset)
        matrix = U.hmm(scaling, U.hmm(rotation, translation))
        return dict(affine=matrix)


class VectorFieldDecoder(paddle.nn.Layer):
    r"""Decoder of non-rigid vector field."""

    def __init__(
        self,
        in_channels: int,
        in_size: Sequence[int],
        batch_norm_enabled: bool = False,
        bias_enabled: bool = True,
        bias_init: str = "uniform",
        normalize_flow: bool = True,
    ) -> None:
        if in_channels < 0:
            raise ValueError(f"{type(self).__name__}() 'in_channels' must be positive")
        spatial_dims = len(in_size)
        if spatial_dims < 1 or spatial_dims > 3:
            raise ValueError(f"{type(self).__name__}() 'in_size' must be sequence of length 2 or 3")
        super().__init__()
        self._in_channels = in_channels
        self._in_size = tuple(in_size)
        self.normalize_flow = normalize_flow
        interp_mode = "bilinear" if spatial_dims == 2 else "trilinear"

        def conv(m: int, n: int) -> ConvLayer:
            ks = 3
            bias = bias_init if bias_enabled else False
            norm = "batch" if batch_norm_enabled else None
            acti = "relu"
            return ConvLayer(spatial_dims, m, n, kernel_size=ks, bias=bias, norm=norm, acti=acti)

        self.blocks = paddle.nn.LayerList(
            sublayers=[conv(2 * in_channels, 32), conv(48, 16), conv(32, 16), conv(16, 8)]
        )
        self.head = convolution(spatial_dims, 8, spatial_dims, kernel_size=3, padding=1)
        self.up = paddle.nn.Upsample(scale_factor=2, mode=interp_mode, align_corners=False)

    @property
    def nblocks(self) -> int:
        return len(self.blocks)

    @property
    def spatial_dims(self) -> int:
        return len(self.in_size)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def in_size(self) -> list:
        return self._in_size

    @property
    def out_channels(self) -> int:
        return self.spatial_dims

    @property
    def out_size(self) -> list:
        r"""Spatial size of output vector field parameters."""
        return self.output_size()

    @property
    def out_shape(self) -> list:
        r"""Shape of output vector field parameters for a batch size of 1."""
        return self.output_shape()

    def input_shape(self, batch_size: int = 1) -> list:
        r"""Expected shape of input tensor for given batch size."""
        return tuple((batch_size, self.in_channels)) + self.in_size[::-1]

    def output_shape(self, in_shape: Optional[Sequence[int]] = None) -> list:
        r"""Calculate shape of output vector field parameters."""
        if not in_shape:
            batch_size = 1
            in_shape = self.input_shape(batch_size)
        else:
            batch_size = in_shape[0]
            input_shape = self.input_shape(batch_size)
            if len(in_shape) != self.spatial_dims + 2 or in_shape != input_shape:
                raise ValueError(
                    f"{type(self).__name__}.output_shape() 'in_shape' must be (N, {', '.join(str(n) for n in input_shape[1:])}"
                )
        return tuple((batch_size, self.out_channels)) + self.output_size(in_shape[:1:-1])[::-1]

    def output_size(self, in_size: Optional[ScalarOrTuple[int]] = None) -> ScalarOrTuple[int]:
        r"""Calculate spatial size of vector field parameters."""
        if in_size is not None and in_size != self.in_size:
            raise ValueError(
                f"{type(self).__name__}.output_size() 'in_size' must be None or {self.in_size}"
            )
        size = self.in_size
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                size = module_output_size(self.up, size)
            size = module_output_size(block, size)
        size = module_output_size(self.head, size)
        return size

    def forward(self, features: Sequence[paddle.Tensor]) -> paddle.Tensor:
        features: List[paddle.Tensor] = list(features)
        if len(features) != len(self.blocks):
            raise ValueError(
                f"{type(self).__name__} expected {len(self.nblocks)} input feature map{'s' if len(self.nblocks) > 1 else ''}"
            )
        x = features.pop()
        for block in self.blocks:
            if features:
                x = paddle.concat(x=[self.up(x), features.pop()], axis=1)
            x = block(x)
        x = self.head(x)
        if self.normalize_flow:
            x = U.normalize_flow(x, size=self.in_size, side_length=1, align_corners=True)
        return x


class SpatialTransformerNetwork(paddle.nn.Layer):
    r"""Spatial transformer network.

    The input to the STN is generally a concatenated pair of images with one or more channels each,
    and the output is a vector field. The predicted vectors can be either displacement vectors, velocity
    vectors, or control point coefficients of a spline representation. The interpretation of the output
    vectors depends on their subsequent use as parameters of a ``SpatialTransform``, which represents
    the actual spatial coordinate transformation used to deform an image grid or other set of points
    at which an input moving image is sampled to produce a deformed output image.

    """

    def __init__(
        self, in_channels: int, in_size: Sequence[int], config: SpatialTransformerConfig
    ) -> None:
        r"""Initialize spatial transformer modules.

        Args:
            in_channels: Number of input tensor channels.
            in_size: Number and size of spatial dimensions of input tensor.
            config: Spatial transformer network configuration.

        """
        super().__init__()
        config = deepcopy(config)
        self.config = config
        self._in_channels = in_channels
        self._in_size = tuple(in_size)
        spatial_dims = len(in_size)
        self.encoder = SpatialTransformEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            bias_enabled=config.bias_enabled,
            bias_init=config.bias_init,
        )
        fm_channels = self.encoder.output_channels(-1)
        fm_size = self.encoder.output_size(in_size)
        if has_affine_component(config.transform):
            ds_factor = 2
            features = fm_channels * tuple(n // ds_factor for n in fm_size).size
            self.pool = pooling("avg", spatial_dims=spatial_dims, kernel_size=ds_factor)
            self.affine = AffineTransformDecoder(
                spatial_dims=spatial_dims,
                features=features,
                hidden=config.affine_nodes,
                max_angle=config.affine_max_angle,
                max_offset=config.affine_max_offset,
                max_scale=config.affine_max_scale,
                as_matrix=config.affine_model == "A",
                init=config.affine_init,
            )
        else:
            self.affine = None
        if has_nonrigid_component(config.transform):
            self.vfield = VectorFieldDecoder(
                in_channels=fm_channels,
                in_size=fm_size[::-1] if config.flip_grid_coords else fm_size,
                bias_enabled=config.bias_enabled,
                bias_init=config.bias_init,
                batch_norm_enabled=config.batch_norm_enabled,
                normalize_flow=config.normalize_flow,
            )
        else:
            self.vfield = None

    @property
    def spatial_dims(self) -> int:
        r"""Number of spatial dimensions."""
        return len(self.in_size)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def in_size(self) -> list:
        return self._in_size

    @property
    def in_shape(self) -> list:
        r"""Expected input tensor shape for a batch size of 1."""
        return self.input_shape()

    def input_shape(self, batch_size: int = 1) -> list:
        r"""Expected input tensor shape for given batch size."""
        return tuple((batch_size, self.in_channels)) + self.in_size[::-1]

    @property
    def out_channels(self) -> int:
        r"""Number of output channels of vector field parameters or 0."""
        return 0 if self.vfield is None else self.spatial_dims

    @property
    def out_size(self) -> list:
        r"""Spatial size of output vector field parameters for batch size 1 or empty Size."""
        return self.output_size()

    @property
    def out_shape(self) -> list:
        r"""Shape of output vector field parameters for batch size 1 or empty Size."""
        return self.output_shape()

    def output_shape(self, in_shape: Optional[Sequence[int]] = None) -> list:
        r"""Calculate shape of output vector field parameters."""
        if not in_shape:
            batch_size = 1
        else:
            batch_size = in_shape[0]
            input_shape = self.input_shape(batch_size)
            if len(in_shape) != self.spatial_dims + 2 or in_shape != input_shape:
                raise ValueError(
                    f"{type(self).__name__}.output_shape() 'in_shape' must be (N, {', '.join(str(n) for n in input_shape[1:])}"
                )
        size = self.output_size()
        if not size:
            return ()
        return tuple((batch_size, self.spatial_dims)) + size[::-1]

    def output_size(self, in_size: Optional[ScalarOrTuple[int]] = None) -> ScalarOrTuple[int]:
        r"""Calculate spatial size of output vector field parameters."""
        if in_size and in_size != self.in_size:
            raise ValueError(
                f"{type(self).__name__}.output_size() 'in_size' must be None or {self.in_size}"
            )
        if self.vfield is None:
            return ()
        size = self.encoder.output_size(self.in_size)
        size = self.vfield.output_size(size)
        return size

    def forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        r"""Infer spatial transformation parameters given input data tensor."""
        params = {}
        features: Tensor = self.encoder(x)
        if self.affine is not None:
            z: Tensor = features[-1]
            if self.pool is not None:
                z = self.pool(z)
            params.update(self.affine(z.flatten(start_axis=1)))
        if self.vfield is not None:
            params["vfield"] = self.vfield(features)
        return params

    @classmethod
    def read(
        cls,
        path: PathStr,
        in_channels: int,
        in_size: Sequence[int],
        device: Optional[Device] = None,
    ) -> SpatialTransformerNetwork:
        r"""Create spatial transformer network from previously saved model file.

        Args:
            path: Path of saved model file written by ``self.write(path)``.
            in_channels: Number of input channels.
            in_size: Spatial size of input tensor.
            device: Device on which to load model. If ``None``, use "cpu".

        Returns:
            Spatial transformer network with parameters initialized to the previously saved state.

        """
        if device is None:
            device = paddle.CPUPlace()
        data = paddle.load(path=str(path))
        if not isinstance(data, dict):
            raise ValueError(f"{cls.__name__}.read() model file must contain dictionary")
        if "config" not in data:
            raise ValueError(f"{cls.__name__}.read() model file must contain 'config' dict entry")
        if "state" not in data:
            raise ValueError(f"{cls.__name__}.read() model file must contain 'state' dict entry")
        config = SpatialTransformerConfig.from_dict(data["config"])
        stn = cls(in_channels=in_channels, in_size=in_size, config=config)
        stn.to(device).set_state_dict(state_dict=data["state"])
        return stn

    def write(self, path: PathStr) -> None:
        r"""Save model parameters and configuration."""
        path = unlink_or_mkdir(path)
        paddle.save(obj={"config": self.config.asdict(), "state": self.state_dict()}, path=path)
