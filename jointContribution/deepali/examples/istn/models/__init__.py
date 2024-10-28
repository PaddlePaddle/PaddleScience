r"""Image and spatial transformer network."""
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
from deepali.core import DataclassConfig
from deepali.core import Grid
from deepali.core import PaddingMode
from deepali.core import Sampling
from deepali.core import functional as U
from deepali.spatial import GenericSpatialTransform
from deepali.spatial import ImageTransformer
from deepali.spatial import ParametricTransform
from paddle.nn import Layer

from .itn import ImageTransformerConfig
from .itn import ImageTransformerNetwork
from .stn import SpatialTransformerConfig
from .stn import SpatialTransformerNetwork


@dataclass
class InputConfig(DataclassConfig):
    size: Sequence[int]
    channels: int = 1

    @property
    def spatial_dims(self) -> int:
        return len(self.size)

    @property
    def shape(self) -> list:
        return tuple(reversed(self.size))


@dataclass
class ImageAndSpatialTransformerConfig(DataclassConfig):
    input: InputConfig
    itn: Optional[Union[str, ImageTransformerConfig]] = "miccai2019"
    stn: SpatialTransformerConfig = SpatialTransformerConfig()


class ImageAndSpatialTransformerNetwork(paddle.nn.Layer):
    r"""Image and spatial transformer network."""

    def __init__(
        self,
        stn: SpatialTransformerNetwork,
        itn: Optional[ImageTransformerNetwork] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str, float] = PaddingMode.BORDER,
    ) -> None:
        super().__init__()
        if stn.in_channels < 2 or stn.in_channels % 2 != 0:
            raise ValueError(
                f"{type(self).__name__}() 'stn.in_channels' must be positive even number"
            )
        self.itn = itn
        self.stn = stn
        grid = Grid(size=stn.in_size)
        transform = GenericSpatialTransform(grid, params=None, config=stn.config)
        self.warp = ImageTransformer(transform, sampling=sampling, padding=padding)

    @property
    def config(self) -> ImageAndSpatialTransformerConfig:
        stn: SpatialTransformerNetwork = self.stn
        itn: Optional[ImageTransformerNetwork] = self.itn
        return ImageAndSpatialTransformerConfig(
            input=InputConfig(size=tuple(stn.in_size), channels=stn.in_channels // 2),
            itn=None if itn is None else itn.config,
            stn=stn.config,
        )

    @property
    def transform(self) -> GenericSpatialTransform:
        r"""Reference to spatial coordinates transformation."""
        return self.warp.transform

    def forward(
        self, source_img: paddle.Tensor, target_img: paddle.Tensor, apply: bool = True
    ) -> Dict[str, paddle.Tensor]:
        output: Dict[str, paddle.Tensor] = {}
        itn: Optional[paddle.nn.Layer] = self.itn
        source_soi = source_img
        target_soi = target_img
        if itn is not None:
            with paddle.set_grad_enabled(mode=itn.training):
                source_soi = itn(source_img)
                target_soi = itn(target_img)
            output["source_soi"] = source_soi
            output["target_soi"] = target_soi
        else:
            output["source_soi"] = source_img
            output["target_soi"] = target_img
        stn: Layer = self.stn
        stn_input = paddle.concat(x=[source_soi, target_soi], axis=1)
        params: Dict[str, paddle.Tensor] = stn(stn_input)
        vfield_params: Optional[paddle.Tensor] = params.get("vfield")
        nonrigid_transform: Optional[ParametricTransform] = self.transform.get("nonrigid")
        if vfield_params is None:
            assert nonrigid_transform is None
        else:
            assert nonrigid_transform is not None
            vfield_shape = nonrigid_transform.data_shape[1:]
            vfield_params = U.grid_reshape(vfield_params, vfield_shape, align_corners=False)
            params["vfield"] = vfield_params
        self.transform.params = params
        if apply:
            output["warped_img"] = self.warp(source_img)
        return output


def create_istn(config: ImageAndSpatialTransformerConfig) -> ImageAndSpatialTransformerNetwork:
    itn = ImageTransformerNetwork(config.input.spatial_dims, config.input.channels, config.itn)
    itn_output_size = itn.output_size(config.input.size)
    stn = SpatialTransformerNetwork(2 * itn.out_channels, itn_output_size, config.stn)
    return ImageAndSpatialTransformerNetwork(itn=itn, stn=stn)
