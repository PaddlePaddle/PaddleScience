# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code below is heavily based on [FourCastNet](https://github.com/NVlabs/FourCastNet)
"""

from functools import partial
from typing import Optional
from typing import Tuple

import paddle
import paddle.fft
import paddle.nn.functional as F
from paddle import nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.utils import initializer


def drop_path(
    x: paddle.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> paddle.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...

    Args:
        x (paddle.Tensor): The tensor to apply.
        drop_prob (float, optional):  Drop paths probability. Defaults to 0.0.
        training (bool, optional): Whether at training mode. Defaults to False.
        scale_by_keep (bool, optional): Whether upscale the output. Defaults to True.

    Returns:
        paddle.Tensor: Output tensor after apply dropout.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.full(shape, keep_prob, x.dtype)
    random_tensor = paddle.bernoulli(random_tensor)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Args:
        drop_prob (float, optional): Drop paths probability. Defaults to 0.0.
        scale_by_keep (bool, optional): Whether upscale the output. Defaults to True.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class PeriodicPad2d(nn.Layer):
    """Pad longitudinal (left-right) circular and pad latitude (top-bottom) with zeros.

    Args:
        pad (int): Number of pad.
    """

    def __init__(self, pad: int):
        super(PeriodicPad2d, self).__init__()
        self.pad = pad

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        # pad top and bottom zeros
        out = F.pad(
            out,
            (0, 0, 0, 0, self.pad, self.pad, 0, 0),
            mode="constant",
            value=0,
        )
        return out


class MLP(nn.Layer):
    """Multi layer perceptron module used in Transformer.

    Args:
        in_features (int): Number of the input features.
        hidden_features (Optional[int]): Number of the hidden size. Defaults to None.
        out_features (Optional[int]): Number of the output features. Defaults to None.
        activation (str, optional): Name of activation function. Defaults to "gelu".
        drop (float, optional): Probability of dropout the units. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: str = "gelu",
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_mod.get_activation(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Layer):
    """2D Adaptive Fourier Neural Operators.

    Args:
        hidden_size (int): Number of hidden size.
        num_blocks (int, optional): Number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
        hidden_size_factor (int, optional): The factor of hidden size. Defaults to 1.
        scale (float, optional):  The scale factor of the parameter when initialization. Defaults to 0.02.
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
        scale: float = 0.02,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(
                f"hidden_size({hidden_size}) should be divisble by num_blocks({num_blocks})."
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = scale

        self.w1 = self.create_parameter(
            shape=(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            ),
            default_initializer=nn.initializer.Normal(std=self.scale),
        )
        self.b1 = self.create_parameter(
            shape=(2, self.num_blocks, self.block_size * self.hidden_size_factor),
            default_initializer=nn.initializer.Normal(std=self.scale),
        )
        self.w2 = self.create_parameter(
            shape=(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            ),
            default_initializer=nn.initializer.Normal(std=self.scale),
        )
        self.b2 = self.create_parameter(
            shape=(2, self.num_blocks, self.block_size),
            default_initializer=nn.initializer.Normal(std=self.scale),
        )

    def forward(self, x):
        bias = x

        B, H, W, C = x.shape

        x = paddle.fft.rfft2(x, axes=(1, 2), norm="ortho")
        x = x.reshape((B, H, W // 2 + 1, self.num_blocks, self.block_size))

        o1_shape = (
            B,
            H,
            W // 2 + 1,
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
        )
        o1_real = paddle.zeros(o1_shape)
        o1_imag = paddle.zeros(o1_shape)
        o2_real = paddle.zeros(x.shape)
        o2_imag = paddle.zeros(x.shape)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        st, end = total_modes - kept_modes, total_modes + kept_modes

        o1_real[:, st:end, :kept_modes] = F.relu(
            paddle.einsum(
                "xyzbi,bio->xyzbo",
                x[:, st:end, :kept_modes].real(),
                self.w1[0],
            )
            - paddle.einsum(
                "xyzbi,bio->xyzbo",
                x[:, st:end, :kept_modes].imag(),
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[:, st:end, :kept_modes] = F.relu(
            paddle.einsum(
                "xyzbi,bio->xyzbo",
                x[:, st:end, :kept_modes].imag(),
                self.w1[0],
            )
            + paddle.einsum(
                "xyzbi,bio->xyzbo",
                x[:, st:end, :kept_modes].real(),
                self.w1[1],
            )
            + self.b1[1]
        )

        o2_real[:, st:end, :kept_modes] = (
            paddle.einsum(
                "xyzbi,bio->xyzbo",
                o1_real[:, st:end, :kept_modes],
                self.w2[0],
            )
            - paddle.einsum(
                "xyzbi,bio->xyzbo",
                o1_imag[:, st:end, :kept_modes],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, st:end, :kept_modes] = (
            paddle.einsum(
                "xyzbi,bio->xyzbo",
                o1_imag[:, st:end, :kept_modes],
                self.w2[0],
            )
            + paddle.einsum(
                "xyzbi,bio->xyzbo",
                o1_real[:, st:end, :kept_modes],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = paddle.stack([o2_real, o2_imag], axis=-1)
        x = F.softshrink(x, threshold=self.sparsity_threshold)
        x = paddle.as_complex(x)
        x = x.reshape((B, H, W // 2 + 1, C))
        x = paddle.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho")

        return x + bias


class Block(nn.Layer):
    """AFNO network block.

    Args:
        dim (int): The input tensor dimension.
        mlp_ratio (float, optional): The ratio used in MLP. Defaults to 4.0.
        drop (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        activation (str, optional): Name of activation function. Defaults to "gelu".
        norm_layer (nn.Layer, optional): Class of norm layer. Defaults to nn.LayerNorm.
        double_skip (bool, optional): Whether use double skip. Defaults to True.
        num_blocks (int, optional): The number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
        norm_layer: nn.Layer = nn.LayerNorm,
        double_skip: bool = True,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(
            dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class PatchEmbed(nn.Layer):
    """Patch embedding module.

    Args:
        img_size (Tuple[int, ...], optional): Image size. Defaults to (224, 224).
        patch_size (Tuple[int, ...], optional): Patch size. Defaults to (16, 16).
        in_channels (int, optional): The input tensor channels. Defaults to 3.
        embed_dim (int, optional): The output tensor channels. Defaults to 768.
    """

    def __init__(
        self,
        img_size: Tuple[int, ...] = (224, 224),
        patch_size: Tuple[int, ...] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2D(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        _, _, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class AFNONet(base.Arch):
    """Adaptive Fourier Neural Network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        img_size (Tuple[int, ...], optional): Image size. Defaults to (720, 1440).
        patch_size (Tuple[int, ...], optional): Path. Defaults to (8, 8).
        in_channels (int, optional): The input tensor channels. Defaults to 20.
        out_channels (int, optional): The output tensor channels. Defaults to 20.
        embed_dim (int, optional): The embedding dimension for PatchEmbed. Defaults to 768.
        depth (int, optional): Number of transformer depth. Defaults to 12.
        mlp_ratio (float, optional): Number of ratio used in MLP. Defaults to 4.0.
        drop_rate (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path_rate (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        num_blocks (int, optional): Number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
        num_timestamps (int, optional): Number of timestamp. Defaults to 1.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.AFNONet(("input", ), ("output", ))
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        img_size: Tuple[int, ...] = (720, 1440),
        patch_size: Tuple[int, ...] = (8, 8),
        in_channels: int = 20,
        out_channels: int = 20,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        num_timestamps: int = 1,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_timestamps = num_timestamps
        norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        data = paddle.zeros((1, num_patches, embed_dim))
        data = initializer.trunc_normal_(data, std=0.02)
        self.pos_embed = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            self.out_channels * self.patch_size[0] * self.patch_size[1],
            bias_attr=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                initializer.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            initializer.ones_(m.weight)
            initializer.zeros_(m.bias)
        elif isinstance(m, nn.Conv2D):
            initializer.conv_init_(m)

    def forward_tensor(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape((B, self.h, self.w, self.embed_dim))
        for block in self.blocks:
            x = block(x)

        x = self.head(x)

        b = x.shape[0]
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        h = self.img_size[0] // self.patch_size[0]
        w = self.img_size[1] // self.patch_size[1]
        c_out = x.shape[3] // (p1 * p2)
        x = x.reshape((b, h, w, p1, p2, c_out))
        x = x.transpose((0, 5, 1, 3, 2, 4))
        x = x.reshape((b, c_out, h * p1, w * p2))

        return x

    def split_to_dict(
        self, data_tensors: Tuple[paddle.Tensor, ...], keys: Tuple[str, ...]
    ):
        return {key: data_tensors[i] for i, key in enumerate(keys)}

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        x = self.concat_to_tensor(x, self.input_keys)

        y = []
        input = x
        for _ in range(self.num_timestamps):
            out = self.forward_tensor(input)
            y.append(out)
            input = out
        y = self.split_to_dict(y, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y


class PrecipNet(base.Arch):
    """Precipitation Network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        wind_model (base.Arch): Wind model.
        img_size (Tuple[int, ...], optional): Image size. Defaults to (720, 1440).
        patch_size (Tuple[int, ...], optional): Path. Defaults to (8, 8).
        in_channels (int, optional): The input tensor channels. Defaults to 20.
        out_channels (int, optional): The output tensor channels. Defaults to 1.
        embed_dim (int, optional): The embedding dimension for PatchEmbed. Defaults to 768.
        depth (int, optional): Number of transformer depth. Defaults to 12.
        mlp_ratio (float, optional): Number of ratio used in MLP. Defaults to 4.0.
        drop_rate (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path_rate (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        num_blocks (int, optional): Number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
        num_timestamps (int, optional): Number of timestamp. Defaults to 1.

    Examples:
        >>> import ppsci
        >>> wind_model = ppsci.arch.AFNONet(("input", ), ("output", ))
        >>> model = ppsci.arch.PrecipNet(("input", ), ("output", ), wind_model)
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        wind_model: base.Arch,
        img_size: Tuple[int, ...] = (720, 1440),
        patch_size: Tuple[int, ...] = (8, 8),
        in_channels: int = 20,
        out_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        num_timestamps=1,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_timestamps = num_timestamps
        self.backbone = AFNONet(
            ("input",),
            ("output",),
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2D(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=0
        )
        self.act = nn.ReLU()
        self.apply(self._init_weights)
        self.wind_model = wind_model
        self.wind_model.eval()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                initializer.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            initializer.ones_(m.weight)
            initializer.zeros_(m.bias)
        elif isinstance(m, nn.Conv2D):
            initializer.conv_init_(m)

    def forward_tensor(self, x):
        x = self.backbone.forward_tensor(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

    def split_to_dict(
        self, data_tensors: Tuple[paddle.Tensor, ...], keys: Tuple[str, ...]
    ):
        return {key: data_tensors[i] for i, key in enumerate(keys)}

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        x = self.concat_to_tensor(x, self.input_keys)

        input_wind = x
        y = []
        for _ in range(self.num_timestamps):
            with paddle.no_grad():
                out_wind = self.wind_model.forward_tensor(input_wind)
            out = self.forward_tensor(out_wind)
            y.append(out)
            input_wind = out_wind
        y = self.split_to_dict(y, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y
