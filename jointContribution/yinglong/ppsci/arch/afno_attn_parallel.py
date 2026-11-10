from functools import partial
from typing import Tuple

import numpy as np
import paddle
import paddle.fft
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Normal
from paddle.nn.initializer import TruncatedNormal

from ppsci.arch import base
from ppsci.utils import initializer

from .afno import AFNO2D
from .afno import MLP
from .afno import DropPath
from .afno import PatchEmbed
from .afno import PatchMerging
from .afno import UpSample
from .time_embedding import TimeFeatureEmbedding

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def to_2tuple(x):
    return tuple([x] * 2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape(
        [-1, H // window_size, W // window_size, window_size, window_size, C]
    )
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x


class WindowAttention(nn.Layer):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
            default_initializer=zeros_,
        )
        self.add_parameter(
            "relative_position_bias_table", self.relative_position_bias_table
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2

        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def eval(
        self,
    ):
        # this is used to re-param swin for model export
        relative_position_bias_table = self.relative_position_bias_table
        window_size = self.window_size
        index = self.relative_position_index.reshape([-1])

        relative_position_bias = paddle.index_select(
            relative_position_bias_table, index
        )
        relative_position_bias = relative_position_bias.reshape(
            [window_size[0] * window_size[1], window_size[0] * window_size[1], -1]
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1]
        )  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.unsqueeze(0)
        self.register_buffer("relative_position_bias", relative_position_bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape([B_, N, 3, self.num_heads, C // self.num_heads])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))

        if self.training or not hasattr(self, "relative_position_bias"):
            index = self.relative_position_index.reshape([-1])

            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index
            )
            relative_position_bias = relative_position_bias.reshape(
                [
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1,
                ]
            )  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1]
            )  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + self.relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape([B_, N, C])
        x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return "dim={}, window_size={}, num_heads={}".format(
            self.dim, self.window_size, self.num_heads
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


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
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        attn_channel_ratio=0.5,
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
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn_channle_ratio = attn_channel_ratio
        self.attn_dim = int(dim * attn_channel_ratio)

        self.norm1 = norm_layer(dim)

        if dim - self.attn_dim > 0:
            self.filter = AFNO2D(
                dim - self.attn_dim,
                num_blocks,
                sparsity_threshold,
                hard_thresholding_fraction,
            )
        if self.attn_dim > 0:
            self.attn = WindowAttention(
                self.attn_dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution
                Hp = int(np.ceil(H / self.window_size)) * self.window_size
                Wp = int(np.ceil(W / self.window_size)) * self.window_size
                img_mask = paddle.zeros([1, Hp, Wp, 1], dtype="float32")  # 1 Hp Wp 1
                h_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                w_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        try:
                            img_mask[:, h, w, :] = cnt
                        except:
                            pass

                        cnt += 1

                mask_windows = window_partition(
                    img_mask, self.window_size
                )  # nW, window_size, window_size, 1
                mask_windows = mask_windows.reshape(
                    [-1, self.window_size * self.window_size]
                )
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                huns = -100.0 * paddle.ones_like(attn_mask)
                attn_mask = huns * (attn_mask != 0).astype("float32")
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask)

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

    def attn_forward(self, x):
        B, H, W, C = x.shape

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C]
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp, C
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
        return x

    def afno_forward(self, x):
        x = self.filter(x)
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.norm1(x)

        if self.attn_dim == 0:
            x = self.afno_forward(x)
        elif self.attn_dim == self.dim:
            x = self.attn_forward(x)
        else:  # self.attn_dim > 0 and self.attn_dim < self.dim
            x_attn = x[:, :, :, : self.attn_dim]
            x_afno = x[:, :, :, self.attn_dim :]
            x_attn = self.attn_forward(x_attn)
            x_afno = self.afno_forward(x_afno)

            x = paddle.concat([x_attn, x_afno], axis=-1)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNOAttnParallelNet(base.Arch):
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
        window_size=7,
        num_heads=8,
        attn_channel_ratio=0.5,
        use_recompute=False,
        merge_label=False,
        merge_weights_n=None,
        merge_weights_m=None,
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
        self.window_size = window_size
        self.num_heads = num_heads
        if not isinstance(attn_channel_ratio, list):
            self.attn_channel_ratio = [attn_channel_ratio] * depth
        else:
            self.attn_channel_ratio = attn_channel_ratio
        assert len(self.attn_channel_ratio) == depth

        self.use_recompute = use_recompute
        self.merge_label = merge_label
        if merge_label is True:
            self.merge_weights_n = paddle.to_tensor(
                np.load(merge_weights_n), dtype=paddle.float32
            )
            self.merge_weights_m = paddle.to_tensor(
                np.load(merge_weights_m), dtype=paddle.float32
            )

            self.merge_weights_n = self.merge_weights_n.unsqueeze(0).unsqueeze(0)
            self.merge_weights_m = self.merge_weights_m.unsqueeze(0).unsqueeze(0)

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

        self.time_embed = TimeFeatureEmbedding(
            d_model=embed_dim, embed_type="fixed", freq="h"
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    input_resolution=(self.h, self.w),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    attn_channel_ratio=self.attn_channel_ratio[i],
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

    def forward_tensor(self, x, x_time):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed + self.time_embed(x_time, x.shape[1])
        x = self.pos_drop(x)

        x = x.reshape((B, self.h, self.w, self.embed_dim))

        for block in self.blocks:
            # x = block(x)
            if not self.use_recompute:
                x = block(x)
            else:
                x = recompute(block, x)

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

    def forward(self, x, x_time):
        if self._input_transform is not None:
            x = self._input_transform(x)
        x_tensor = self.concat_to_tensor(x, self.input_keys)

        y = []
        input = x_tensor
        for i in range(self.num_timestamps):
            out = self.forward_tensor(input, x_time[i])
            y.append(out)
            if self.merge_label:
                input = (
                    self.merge_weights_m * out
                    + self.merge_weights_n * x[f"{self.input_keys[0]}_{i}_merge"]
                )
            else:
                input = out
        y = self.split_to_dict(y, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y

    # def forward(self, x):
    #     x_time = ['2020/01/02/0']
    #     out = self.forward_tensor(x,x_time)
    #     return out


class AFNOAttnParallelUNet(base.Arch):
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
        depths=[4, 2, 2, 4],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        num_timestamps: int = 1,
        window_size=7,
        num_heads=8,
        attn_channel_ratio=0.25,
        use_recompute=False,
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
        self.window_size = window_size
        self.num_heads = num_heads
        depth = sum(depths)
        if not isinstance(attn_channel_ratio, list):
            self.attn_channel_ratio = [attn_channel_ratio] * depth
        else:
            self.attn_channel_ratio = attn_channel_ratio
        assert len(self.attn_channel_ratio) == depth

        self.use_recompute = use_recompute

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
        self.num_layers = len(depths)

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        # layer0
        self.block0 = nn.Sequential()
        for i in range(depths[0]):
            block = Block(
                dim=embed_dim,
                input_resolution=(self.h, self.w),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                attn_channel_ratio=self.attn_channel_ratio[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                num_blocks=self.num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            self.block0.add_sublayer(str(i), block)

        # layer 1
        self.block1 = nn.Sequential()
        self.patch_merger = PatchMerging(embed_dim, norm_layer)
        for i in range(depths[1]):
            i = depths[0] + i
            block = Block(
                dim=embed_dim * 2,
                input_resolution=(self.h // 2, self.w // 2),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                attn_channel_ratio=self.attn_channel_ratio[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                num_blocks=self.num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            self.block1.add_sublayer(str(i), block)

        # layer 2
        self.block2 = nn.Sequential()
        for i in range(depths[2]):
            i = depths[0] + depths[1] + i
            block = Block(
                dim=embed_dim * 2,
                input_resolution=(self.h // 2, self.w // 2),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                attn_channel_ratio=self.attn_channel_ratio[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                num_blocks=self.num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            self.block2.add_sublayer(str(i), block)

        # layer 3
        self.block3 = nn.Sequential()
        self.upsample = UpSample(self.h, self.w, embed_dim * 2, embed_dim, norm_layer)
        for i in range(depths[3]):
            i = depths[0] + depths[1] + depths[2] + i
            block = Block(
                dim=embed_dim,
                input_resolution=(self.h, self.w),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                attn_channel_ratio=self.attn_channel_ratio[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                num_blocks=self.num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            self.block3.add_sublayer(str(i), block)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim * 2,
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

        if not self.use_recompute:
            x0 = self.block0(x)
        else:
            x0 = recompute(self.block0, x)

        x1 = self.patch_merger(x0)
        if not self.use_recompute:
            x1 = self.block1(x1)
        else:
            x1 = recompute(self.block1, x1)

        if not self.use_recompute:
            x2 = self.block2(x1)
        else:
            x2 = recompute(self.block2, x1)

        x3 = self.upsample(x2)
        if not self.use_recompute:
            x3 = self.block3(x3)
        else:
            x3 = recompute(self.block3, x3)

        x = paddle.concat([x0, x3], axis=-1)

        if not self.use_recompute:
            x = self.head(x)
        else:
            x = recompute(self.head, x)

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


class BlockV2(nn.Layer):
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
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
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
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)

        self.filter = AFNO2D(
            dim,
            num_blocks,
            sparsity_threshold,
            hard_thresholding_fraction,
        )

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            img_mask = paddle.zeros([1, Hp, Wp, 1], dtype="float32")  # 1 Hp Wp 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    try:
                        img_mask[:, h, w, :] = cnt
                    except:
                        pass

                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(
                [-1, self.window_size * self.window_size]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            huns = -100.0 * paddle.ones_like(attn_mask)
            attn_mask = huns * (attn_mask != 0).astype("float32")
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.reduce = nn.Linear(2 * dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            drop=drop,
        )
        self.double_skip = double_skip

    def attn_forward(self, x):
        B, H, W, C = x.shape

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C]
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp, C
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
        return x

    def afno_forward(self, x):
        x = self.filter(x)
        return x

    def forward(self, x):
        residual = x
        x = self.norm1(x)

        x_attn = self.attn_forward(x)
        x_afno = self.afno_forward(x)

        x = paddle.concat([x_attn, x_afno], axis=-1)
        x = self.reduce(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNOAttnParallelNetV2(base.Arch):
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
        window_size=7,
        num_heads=8,
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
        self.window_size = window_size
        self.num_heads = num_heads

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
                BlockV2(
                    dim=embed_dim,
                    input_resolution=(self.h, self.w),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
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


class BlockV3(nn.Layer):
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
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
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
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)

        self.reduce_attn = nn.Linear(dim, dim // 4)
        self.reduce_afno = nn.Linear(dim, dim // 4 * 3)

        self.filter = AFNO2D(
            dim // 4 * 3,
            num_blocks,
            sparsity_threshold,
            hard_thresholding_fraction,
        )

        self.attn = WindowAttention(
            dim // 4,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            img_mask = paddle.zeros([1, Hp, Wp, 1], dtype="float32")  # 1 Hp Wp 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    try:
                        img_mask[:, h, w, :] = cnt
                    except:
                        pass

                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(
                [-1, self.window_size * self.window_size]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            huns = -100.0 * paddle.ones_like(attn_mask)
            attn_mask = huns * (attn_mask != 0).astype("float32")
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

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

    def attn_forward(self, x):
        B, H, W, C = x.shape

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C]
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp, C
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
        return x

    def afno_forward(self, x):
        x = self.filter(x)
        return x

    def forward(self, x):
        residual = x
        x = self.norm1(x)

        x_attn = self.reduce_attn(x)
        x_attn = self.attn_forward(x_attn)

        x_afno = self.reduce_afno(x)
        x_afno = self.afno_forward(x_afno)

        x = paddle.concat([x_attn, x_afno], axis=-1)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNOAttnParallelNetV3(base.Arch):
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
        window_size=7,
        num_heads=8,
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
        self.window_size = window_size
        self.num_heads = num_heads

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
                BlockV3(
                    dim=embed_dim,
                    input_resolution=(self.h, self.w),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
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
