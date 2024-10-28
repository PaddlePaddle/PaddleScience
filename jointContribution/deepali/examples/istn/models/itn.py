r"""Image-to-Image transformer network (ITN)."""
import sys
from typing import Union

from deepali.networks.unet import UNet
from deepali.networks.unet import UNetConfig as ImageTransformerConfig


class ImageTransformerNetwork(UNet):
    r"""Image transformer network."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        config: Union[str, ImageTransformerConfig] = "default",
    ) -> None:
        if isinstance(config, str):
            config = itn_config(config)
        elif not isinstance(config, ImageTransformerConfig):
            raise TypeError(
                "ImageTransformerNetwork() 'config' must be str or ImageTransformerConfig"
            )
        if spatial_dims < 2 or spatial_dims > 3:
            raise ValueError("ImageTransformerNetwork() 'spatial_dims' must be 2 or 3")
        if in_channels < 1:
            raise ValueError("ImageTransformerNetwork() 'in_channels' must be positive")
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            config=config,
        )


def itn_config(name: str = "default") -> ImageTransformerConfig:
    if name == "default":
        name = "miccai2019"
    config_factory = getattr(sys.modules[__name__], f"itn_config_{name.lower()}", None)
    if config_factory is None:
        raise ValueError(f"create_itn_model() has no 'config' named '{name}'")
    config = config_factory()
    assert isinstance(config, ImageTransformerConfig)
    return config


def itn_config_miccai2019() -> ImageTransformerConfig:
    config = ImageTransformerConfig()
    kernel_size = 3
    scale_factor = 2
    acti = "relu"
    bias = True
    norm = None
    config.encoder.num_channels = (2, 4), (8, 8), (16, 16)
    config.encoder.conv_layer.kernel_size = kernel_size
    config.encoder.conv_layer.padding = kernel_size // 2
    config.encoder.conv_layer.bias = bias
    config.encoder.conv_layer.acti = acti
    config.encoder.conv_layer.norm = norm
    config.encoder.downsample.mode = "conv"
    config.encoder.downsample.factor = scale_factor
    config.encoder.downsample.kernel_size = scale_factor
    config.encoder.downsample.padding = 0
    config.decoder.num_channels = 16, (8, 8), (4, 2, 2)
    config.decoder.conv_layer.kernel_size = kernel_size
    config.encoder.conv_layer.padding = kernel_size // 2
    config.decoder.conv_layer.bias = bias
    config.decoder.conv_layer.acti = acti
    config.decoder.conv_layer.norm = norm
    config.decoder.upsample.mode = "deconv"
    config.decoder.upsample.factor = scale_factor
    config.decoder.upsample.kernel_size = scale_factor
    config.decoder.upsample.padding = 0
    config.decoder.join_mode = "add"
    config.output.kernel_size = kernel_size
    config.output.bias = bias
    config.output.acti = acti
    config.output.norm = norm
    return config
