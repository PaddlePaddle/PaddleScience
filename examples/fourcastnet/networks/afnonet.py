# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://github.com/NVlabs/AFNO-transformer

import warnings
import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fft
from paddle.nn.initializer import Constant, Uniform

from utils.img_utils import PeriodicPad2d

zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with paddle.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor = paddle.uniform(
        shape=tensor.shape, dtype=tensor.dtype, min=2 * l - 1, max=2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor = paddle.multiply(tensor, paddle.to_tensor(std * math.sqrt(2.)))
    tensor = paddle.add(tensor, paddle.to_tensor(mean))

    # Clamp to ensure it's in the proper range
    tensor = paddle.clip(tensor, min=a, max=b)
    return tensor


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
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
    def __init__(self,
                 hidden_size,
                 num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        w1 = self.scale * paddle.randn(
            (2, self.num_blocks, self.block_size,
             self.block_size * self.hidden_size_factor))
        self.w1 = self.create_parameter(
            shape=w1.shape,
            dtype=w1.dtype,
            default_initializer=paddle.nn.initializer.Assign(w1))
        b1 = self.scale * paddle.randn(
            (2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.b1 = self.create_parameter(
            shape=b1.shape,
            dtype=b1.dtype,
            default_initializer=paddle.nn.initializer.Assign(b1))
        w2 = self.scale * paddle.randn(
            (2, self.num_blocks, self.block_size * self.hidden_size_factor,
             self.block_size))
        self.w2 = self.create_parameter(
            shape=w2.shape,
            dtype=w2.dtype,
            default_initializer=paddle.nn.initializer.Assign(w2))
        b2 = self.scale * paddle.randn((2, self.num_blocks, self.block_size))
        self.b2 = self.create_parameter(
            shape=b2.shape,
            dtype=b2.dtype,
            default_initializer=paddle.nn.initializer.Assign(b2))

    def forward(self, x):
        bias = x

        B, H, W, C = x.shape

        x = paddle.fft.rfft2(x, axes=(1, 2), norm="ortho")
        x = x.reshape((B, H, W // 2 + 1, self.num_blocks, self.block_size))

        o1_real = paddle.zeros([
            B, H, W // 2 + 1, self.num_blocks,
            self.block_size * self.hidden_size_factor
        ])
        o1_imag = paddle.zeros([
            B, H, W // 2 + 1, self.num_blocks,
            self.block_size * self.hidden_size_factor
        ])
        o2_real = paddle.zeros(x.shape)
        o2_imag = paddle.zeros(x.shape)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            paddle.einsum('xyzbi,bio->xyzbo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real(), self.w1[0]) - \
            paddle.einsum('xyzbi,bio->xyzbo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag(), self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            paddle.einsum('xyzbi,bio->xyzbo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag(), self.w1[0]) + \
            paddle.einsum('xyzbi,bio->xyzbo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real(), self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            paddle.einsum('xyzbi,bio->xyzbo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            paddle.einsum('xyzbi,bio->xyzbo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            paddle.einsum('xyzbi,bio->xyzbo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            paddle.einsum('xyzbi,bio->xyzbo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = paddle.stack([o2_real, o2_imag], axis=-1)
        x = F.softshrink(x, threshold=self.sparsity_threshold)
        x = paddle.as_complex(x)
        x = x.reshape((B, H, W // 2 + 1, C))
        x = paddle.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho")

        return x + bias


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 double_skip=True,
                 num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold,
                             hard_thresholding_fraction)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
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


class PrecipNet(nn.Layer):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2D(
            self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0)
        self.act = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            data = trunc_normal_(m.weight, std=.02)
            m.weight = paddle.create_parameter(
                shape=m.weight.shape,
                dtype=m.weight.dtype,
                default_initializer=nn.initializer.Assign(data))
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            k = 1 / (m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
            uniform = Uniform(-k**0.5, k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class AFNONet(nn.Layer):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0, ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        zero = paddle.zeros((1, num_patches, embed_dim))
        zero = trunc_normal_(zero, std=.02)
        self.pos_embed = paddle.create_parameter(
            shape=zero.shape,
            dtype=zero.dtype,
            default_initializer=nn.initializer.Assign(zero))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                num_blocks=self.num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias_attr=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            data = trunc_normal_(m.weight, std=.02)
            m.weight = paddle.create_parameter(
                shape=m.weight.shape,
                dtype=m.weight.dtype,
                default_initializer=nn.initializer.Assign(data))
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            k = 1 / (m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
            uniform = Uniform(-k**0.5, k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape((B, self.h, self.w, self.embed_dim))
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
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


class PatchEmbed(nn.Layer):
    def __init__(self,
                 img_size=(224, 224),
                 patch_size=(16, 16),
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


def drop_path(x,
              drop_prob: float=0.,
              training: bool=False,
              scale_by_keep: bool=True):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = paddle.full(shape, keep_prob)
    random_tensor = paddle.bernoulli(random_tensor)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float=0., scale_by_keep: bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


if __name__ == "__main__":
    model = AFNONet(
        img_size=(720, 1440), patch_size=(4, 4), in_chans=3, out_chans=10)
    sample = paddle.randn(1, 3, 720, 1440)
    result = model(sample)
    print(result.shape)
    print(paddle.norm(result))
