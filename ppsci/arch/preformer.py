from typing import Optional
from typing import Tuple

import numpy as np
from paddle import nn

from ppsci.arch import base


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class ConvSC(nn.Layer):
    def __init__(self, C_in: int, C_out: int, stride: int, transpose: bool = False):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        if not transpose:
            self.conv = nn.Conv2D(
                C_in,
                C_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                weight_attr=nn.initializer.KaimingNormal(),
            )
        else:
            self.conv = nn.Conv2DTranspose(
                C_in,
                C_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=stride // 2,
                weight_attr=nn.initializer.KaimingNormal(),
            )
        self.norm = nn.GroupNorm(2, C_out)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.conv(x)
        y = self.act(self.norm(y))
        return y


class OverlapPatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(perm=[0, 2, 1])
        x = self.norm(x)

        return x, H, W


class DWConv(nn.Layer):
    def __init__(self, dim: int = 768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(perm=[0, 2, 1]).reshape([B, C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(perm=[0, 2, 1])

        return x


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Layer = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: Optional[int] = None,
        qk_scale: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: float = 1.0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape([B, N, self.num_heads, C // self.num_heads])
            .transpose(perm=[0, 2, 1, 3])
        )

        if self.sr_ratio > 1:
            x_ = x.transpose(perm=[0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(x_).reshape([B, C, -1]).transpose(perm=[0, 2, 1])
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape([B, -1, 2, self.num_heads, C // self.num_heads])
                .transpose(perm=[2, 0, 3, 1, 4])
            )
        else:
            kv = (
                self.kv(x)
                .reshape([B, -1, 2, self.num_heads, C // self.num_heads])
                .transpose(perm=[2, 0, 3, 1, 4])
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(perm=[0, 1, 3, 2])) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(perm=[0, 2, 1, 3]).reshape([B, N, C])
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: Optional[int] = None,
        qk_scale: Optional[int] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Layer = nn.GELU,
        norm_layer: nn.Layer = nn.LayerNorm,
        sr_ratio: float = 1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Encoder(nn.Layer):
    def __init__(self, C_in: int, C_hid: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S)

        self.enc0 = ConvSC(C_in, C_hid, stride=strides[0])
        self.enc1 = OverlapPatchEmbed(
            img_size=256, patch_size=7, stride=4, in_chans=C_hid, embed_dim=C_hid
        )
        self.enc2 = Block(
            dim=C_hid,
            num_heads=1,
            mlp_ratio=4,
            qkv_bias=None,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=8,
        )
        self.norm1 = nn.LayerNorm(C_hid)

    def forward(self, x):
        B = x.shape[0]
        latent = []
        x = self.enc0(x)
        latent.append(x)
        x, H, W = self.enc1(x)
        x = self.enc2(x, H, W)
        x = self.norm1(x)
        x = x.reshape([B, H, W, -1]).transpose(perm=[0, 3, 1, 2]).contiguous()
        latent.append(x)

        return latent


class MidXnet(nn.Layer):
    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        N_T: int,
        incep_ker: Tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ):
        super().__init__()

        self.N_T = N_T
        dpr = [x.item() for x in np.linspace(0, 0.1, N_T)]
        enc_layers = []
        for i in range(N_T):
            enc_layers.append(
                Block(
                    dim=channel_in,
                    num_heads=4,
                    mlp_ratio=4,
                    qkv_bias=None,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    sr_ratio=8,
                )
            )

        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # B TC H W
        x = x.reshape([B, T * C, H, W])
        # B HW TC
        x = x.flatten(2).transpose(perm=[0, 2, 1])

        # encoder
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z, H, W)

        return z


# MultiDecoder
class Decoder(nn.Layer):
    def __init__(self, C_hid: int, C_out: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2D(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)):
            hid = self.dec[i](hid)
        Y = self.readout(hid)
        return Y


class Preformer(base.Arch):
    """
    Preformer is a class that represents a Spatial-Temporal Transformer model designed for short-term precipitation forecasting with multiple meteorological variables.

    Args:
        input_keys (Tuple[str, ...]): A tuple of input keys.
        output_keys (Tuple[str, ...]): A tuple of output keys.
        shape_in (Tuple[int, ...]): The shape of the input data (T, C, H, W), where
            T is the number of time steps, C is the number of channels,
            H and W are the spatial dimensions.
        hid_S (int): The number of hidden channels in the spatial encoder.
        hid_T (int): The number of hidden units in the temporal encoder.
        N_S (int): The number of spatial transformer layers.
        N_T (int): The number of temporal transformer layers.
        incep_ker (Tuple[int, ...]): The kernel sizes used in the inception block.
        groups (int): The number of groups for grouped convolutions.
        num_classes (int): The number of predicted meteorological variables.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.Preformer(
        ...     input_keys=("input",),
        ...     output_keys=("output",),
        ...     shape_in=(6, 12, 192, 256),
        ...     hid_S=64,
        ...     hid_T=256,
        ...     N_S=4,
        ...     N_T=4,
        ...     incep_ker=(3, 5, 7, 11),
        ...     groups=8,
        ...     num_classes=4,
        ... )
        >>> input_dict = {"input": paddle.rand([8, 6, 12, 192, 256])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["output"].shape)
        [8, 6, 1, 192, 256]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        shape_in: Tuple[int, ...],
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        incep_ker: Tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
        num_classes: int = 1,
    ):
        super().__init__()

        self.input_keys = input_keys
        self.output_keys = output_keys

        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid1 = MidXnet(T * hid_S, hid_T // 2, N_T, incep_ker, groups)
        self.dec = Decoder(T * hid_S, T * num_classes, N_S)

    def forward(self, x_raw):
        x_raw = x_raw[self.input_keys[0]]

        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape([B * T, C, H, W])

        # encoded
        embed = self.enc(x)
        _, C_4, H_4, W_4 = embed[-1].shape

        # translator
        z = embed[-1].reshape([B, T, C_4, H_4, W_4])
        hid = self.hid1(z)
        hid = hid.transpose(perm=[0, 2, 1]).reshape([B, -1, H_4, W_4])

        # decoded
        Y = self.dec(hid, embed[0])
        Y = Y.reshape([B, T, 1, H, W])

        Y = nn.functional.softplus(Y)

        return {self.output_keys[0]: Y}
