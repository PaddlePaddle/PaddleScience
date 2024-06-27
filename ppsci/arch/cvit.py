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

from typing import Callable
from typing import Optional
from typing import Tuple

import einops

# import flax.linen as nn
import numpy as np
import paddle

# from einops import rearrange
# from einops import repeat
from paddle import nn
from paddle.nn import functional as F

from ppsci.arch import base
from ppsci.utils import initializer
from ppsci.utils import logger

# from jax.nn.initializers import normal, xavier_uniform
# from ppsci.utils.initializer import normal_
# from ppsci.utils.initializer import xavier_uniform_


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: paddle.Tensor):
    if embed_dim % 2 != 0:
        raise ValueError(f"embedding dimension({embed_dim}) must be divisible by 2")

    omega = paddle.arange(embed_dim // 2, dtype=paddle.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape([-1])  # (M,)
    out = paddle.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = paddle.sin(out)  # (M, D/2)
    emb_cos = paddle.cos(out)  # (M, D/2)

    emb = paddle.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, length: int):
    return paddle.unsqueeze(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, paddle.arange(length, dtype=paddle.float32)
        ),
        0,
    )


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int]):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        if embed_dim % 2 != 0:
            raise ValueError(f"embedding dimension({embed_dim}) must be divisible by 2")

        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = paddle.concat([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = paddle.arange(grid_size[0], dtype=paddle.float32)
    grid_w = paddle.arange(grid_size[1], dtype=paddle.float32)
    grid = paddle.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = paddle.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return paddle.unsqueeze(pos_embed, 0)


# class PatchEmbed(nn.Layer):
#     def __init__(
#         self,
#         in_dim: int = 3,
#         patch_size: Tuple[int, int, int] = (1, 16, 16),
#         emb_dim: int = 768,
#         use_norm: bool = False,
#         layer_norm_eps: float = 1e-5,
#     ):
#         super().__init__()
#         self.in_dim = in_dim
#         self.patch_size = patch_size
#         self.emb_dim = emb_dim
#         self.use_norm = use_norm
#         self.layer_norm_eps = layer_norm_eps

#         self.conv = nn.Conv3D(
#             self.in_dim,
#             self.emb_dim,
#             (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
#             (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
#         )
#         if self.use_norm:
#             self.norm = nn.LayerNorm([emb_dim], epsilon=self.layer_norm_eps)

#         self._init_weights()

#     def forward(self, x):
#         b, c, t, h, w = x.shape

#         x = self.conv(x)

#         num_patches = (
#             t // self.patch_size[0],
#             h // self.patch_size[1],
#             w // self.patch_size[2],
#         )

#         x = paddle.reshape(
#             x, (b, self.emb_dim, num_patches[0], num_patches[1] * num_patches[2])
#         )
#         if self.use_norm:
#             x = self.norm(x)

#         return x

#     def _init_weights(self) -> None:
#         initializer.xavier_uniform_(self.conv.weight)
#         initializer.constant_(self.conv.bias, 0)


class MlpBlock(nn.Layer):
    def __init__(self, in_dim: int, dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.dim)
        self.linear2 = nn.Linear(self.dim, self.out_dim)
        self.act = nn.GELU()

        self._init_weights()

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.act(x)
        x = self.linear2(x)
        return x

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.linear1.weight)
        initializer.constant_(self.linear1.bias, 0)
        initializer.xavier_uniform_(self.linear2.weight)
        initializer.constant_(self.linear2.bias, 0)


class SelfAttnBlock(nn.Layer):
    def __init__(
        self, num_heads: int, emb_dim: int, mlp_ratio: int, layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.mlp_ratio = mlp_ratio
        self.layer_norm1 = LayerNorm(emb_dim, -1, layer_norm_eps)
        self.attn_layer = MultiHeadDotProductAttention(
            self.emb_dim,
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
        )
        self.layer_norm2 = LayerNorm(emb_dim, -1, layer_norm_eps)
        self.mlp = MlpBlock(self.emb_dim, self.emb_dim * self.mlp_ratio, self.emb_dim)

    def forward(self, inputs):
        # inputs:  # [B, L/ps, self.emb_dim]
        x = self.layer_norm1(inputs)  # [B, L/ps, self.emb_dim]
        x = self.attn_layer(x, x)  # [B, L/ps, self.emb_dim]
        x = x + inputs  # [B, L/ps, self.emb_dim]

        y = self.layer_norm2(x)  # [B, L/ps, self.emb_dim]
        y = self.mlp(y)  # [B, L/ps, self.emb_dim]

        return x + y  # [B, L/ps, self.emb_dim]


# class CrossAttnBlock(nn.Layer):
#     num_heads: int
#     emb_dim: int
#     mlp_ratio: int
#     layer_norm_eps: float = 1e-5

#     # @nn.compact
#     def __call__(self, q_inputs, kv_inputs):
#         q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
#         kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

#         x = nn.MultiHeadDotProductAttention(
#             num_heads=self.num_heads, qkv_features=self.emb_dim
#         )(q, kv)
#         x = x + q_inputs
#         y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
#         y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
#         return x + y


# class TimeAggregation(nn.Layer):
#     emb_dim: int
#     depth: int
#     num_heads: int = 8
#     num_latents: int = 64
#     mlp_ratio: int = 1
#     layer_norm_eps: float = 1e-5

#     # @nn.compact
#     def __call__(self, x):  # (B, T, S, D) --> (B, T', S, D)
#         latents = self.param(
#             "latents", normal(), (self.num_latents, self.emb_dim)  # (T', D)
#         )

#         latents = repeat(
#             latents, "t d -> b s t d", b=x.shape[0], s=x.shape[2]
#         )  # (B, T', S, D)
#         x = rearrange(x, "b t s d -> b s t d")  # (B, S, T, D)

#         # Transformer
#         for _ in range(self.depth):
#             latents = CrossAttnBlock(
#                 self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
#             )(latents, x)
#         latents = rearrange(latents, "b s t d -> b t s d")  # (B, T', S, D)
#         return latents


class Mlp(nn.Layer):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        out_dim: int,
        kernel_init: Callable = initializer.xavier_uniform_,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_init = kernel_init
        self.layer_norm_eps = layer_norm_eps
        self.linears = nn.LayerList(
            [
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.gelu = nn.GELU(True)
        self.norms = nn.LayerList(
            [
                LayerNorm(self.hidden_dim, axis=-1, epsilon=self.layer_norm_eps)
                for _ in range(num_layers)
            ]
        )

        self.linear_out = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, inputs):
        x = inputs
        for i, (fc, norm) in enumerate(zip(self.linears, self.norms)):
            y = fc(x)
            y = self.gelu(y)
            x = x + y
            x = norm(x)

        x = self.linear_out(x)
        return x


t_emb_init = get_1d_sincos_pos_embed
s_emb_init = get_2d_sincos_pos_embed


# class Encoder(nn.Layer):
#     patch_size: int = (1, 16, 16)
#     emb_dim: int = 256
#     depth: int = 3
#     num_heads: int = 8
#     mlp_ratio: int = 1
#     out_dim: int = 1
#     layer_norm_eps: float = 1e-5

#     # @nn.compact
#     def __call__(self, x):
#         b, t, h, w, c = x.shape

#         x = PatchEmbed(self.patch_size, self.emb_dim)(x)

#         t_emb = self.variable(
#             "pos_emb",
#             "enc_t_emb",
#             t_emb_init,
#             self.emb_dim,
#             t // self.patch_size[0],
#         )

#         s_emb = self.variable(
#             "pos_emb",
#             "enc_s_emb",
#             s_emb_init,
#             self.emb_dim,
#             (h // self.patch_size[1], w // self.patch_size[2]),
#         )

#         x = x + t_emb.value[:, :, paddle.newaxis, :] + s_emb.value[:, paddle.newaxis, :, :]

#         x = TimeAggregation(
#             num_latents=1,
#             emb_dim=self.emb_dim,
#             depth=2,
#             num_heads=self.num_heads,
#             mlp_ratio=self.mlp_ratio,
#             layer_norm_eps=self.layer_norm_eps,
#         )(x)

#         x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
#         x = rearrange(x, "b t s d -> b (t s) d")

#         for _ in range(self.depth):
#             x = SelfAttnBlock(
#                 self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
#             )(x)

#         return x


# class Vit(nn.Layer):
#     patch_size: tuple = (1, 16, 16)
#     emb_dim: int = 256
#     depth: int = 3
#     num_heads: int = 8
#     mlp_ratio: int = 1
#     num_mlp_layers: int = 1
#     out_dim: int = 1
#     layer_norm_eps: float = 1e-5

#     # @nn.compact
#     def __call__(self, x):
#         x = Encoder(
#             self.patch_size,
#             self.emb_dim,
#             self.depth,
#             self.num_heads,
#             self.mlp_ratio,
#             self.layer_norm_eps,
#         )(x)

#         x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

#         x = Mlp(
#             num_layers=self.num_mlp_layers,
#             hidden_dim=self.emb_dim,
#             out_dim=self.patch_size[1] * self.patch_size[2] * self.out_dim,
#             layer_norm_eps=self.layer_norm_eps,
#         )(x)
#         return x


# class FourierEmbs(nn.Layer):
#     embed_scale: float
#     embed_dim: int

#     # @nn.compact
#     def __call__(self, x):
#         kernel = self.param(
#             "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
#         )
#         y = paddle.concat(
#             [paddle.cos(paddle.dot(x, kernel)), paddle.sin(paddle.dot(x, kernel))], axis=-1
#         )
#         return y


# class CVit(nn.Layer):
#     patch_size: tuple = (1, 16, 16)
#     grid_size: tuple = (128, 128)
#     latent_dim: int = 256
#     emb_dim: int = 256
#     depth: int = 3
#     num_heads: int = 8
#     dec_emb_dim: int = 256
#     dec_num_heads: int = 8
#     dec_depth: int = 1
#     num_mlp_layers: int = 1
#     mlp_ratio: int = 1
#     out_dim: int = 1
#     eps: float = 1e5
#     layer_norm_eps: float = 1e-5
#     embedding_type: str = "grid"

#     def setup(self):
#         if self.embedding_type == "grid":
#             # Create grid and latents
#             n_x, n_y = self.grid_size[0], self.grid_size[1]

#             x = paddle.linspace(0, 1, n_x)
#             y = paddle.linspace(0, 1, n_y)
#             xx, yy = paddle.meshgrid(x, y, indexing="ij")

#             self.grid = paddle.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
#             self.latents = self.param("latents", normal(), (n_x * n_y, self.latent_dim))

#     # @nn.compact
#     def __call__(self, x, coords):
#         b, t, h, w, c = x.shape

#         if self.embedding_type == "grid":
#             #
#             d2 = ((coords[:, paddle.newaxis, :] - self.grid[paddle.newaxis, :, :]) ** 2).sum(
#                 axis=2
#             )
#             w = paddle.exp(-self.eps * d2) / paddle.exp(-self.eps * d2).sum(
#                 axis=1, keepdim=True
#             )

#             coords = paddle.einsum("ic,pi->pc", self.latents, w)
#             coords = nn.Dense(self.dec_emb_dim)(coords)
#             coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

#         elif self.embedding_type == "fourier":
#             coords = FourierEmbs(embed_scale=2 * paddle.pi, embed_dim=self.dec_emb_dim)(
#                 coords
#             )

#         elif self.embedding_type == "mlp":
#             coords = MlpBlock(self.dec_emb_dim, self.dec_emb_dim)(coords)
#             coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

#         coords = einops.repeat(coords, "n d -> b n d", b=b)

#         x = Encoder(
#             self.patch_size,
#             self.emb_dim,
#             self.depth,
#             self.num_heads,
#             self.mlp_ratio,
#             self.layer_norm_eps,
#         )(x)

#         x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
#         x = nn.Dense(self.dec_emb_dim)(x)

#         for _ in range(self.dec_depth):
#             x = CrossAttnBlock(
#                 num_heads=self.dec_num_heads,
#                 emb_dim=self.dec_emb_dim,
#                 mlp_ratio=self.mlp_ratio,
#                 layer_norm_eps=self.layer_norm_eps,
#             )(coords, x)

#         x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
#         x = Mlp(
#             num_layers=self.num_mlp_layers,
#             hidden_dim=self.dec_emb_dim,
#             out_dim=self.out_dim,
#             layer_norm_eps=self.layer_norm_eps,
#         )(x)

#         return x


# # Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     assert embed_dim % 2 == 0
#     omega = paddle.arange(embed_dim // 2, dtype=paddle.float32)
#     omega /= embed_dim / 2.0
#     omega = 1.0 / 10000**omega  # (D/2,)

#     pos = pos.reshape([-1])  # (M,)
#     out = paddle.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

#     emb_sin = paddle.sin(out)  # (M, D/2)
#     emb_cos = paddle.cos(out)  # (M, D/2)

#     emb = paddle.concat([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb


# def get_1d_sincos_pos_embed(embed_dim, length):
#     return paddle.unsqueeze(
#         get_1d_sincos_pos_embed_from_grid(
#             embed_dim, paddle.arange(length, dtype=paddle.float32)
#         ),
#         0,
#     )


class LayerNorm(nn.Layer):
    """Custom layer norm which can do normalization along any given axis.

    Args:
        num_features (_type_): Number of features to normalize.
        axis (int): Axis to normalize along.
        epsilon (float): Epsilon for numerical stability.
    """

    def __init__(self, num_features, axis: int, epsilon: float, use_bias: bool = True):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.LayerNorm(num_features, epsilon, bias_attr=use_bias)
        self.axis = axis

    def forward(self, x):
        x_shape = x.shape
        assert x_shape[self.axis] == self.num_features

        if self.axis != -1:
            perm = list(range(x.ndim))
            perm[-1], perm[self.axis] = perm[self.axis], perm[-1]

            perm_inv = np.empty_like(perm)
            perm_inv[perm] = np.arange(len(perm), dtype="int64").tolist()
            x = paddle.transpose(x, perm)

        x = self.norm(x)

        if self.axis != -1:
            x = paddle.transpose(x, perm_inv)

        return x


class PatchEmbed1D(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        patch_size=(4,),
        emb_dim=768,
        use_norm=False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        self.layer_norm_eps = layer_norm_eps
        self.conv = nn.Conv1D(
            in_dim,
            self.emb_dim,
            self.patch_size[0],
            self.patch_size[0],
            data_format="NLC",
        )
        self.norm = (
            LayerNorm(emb_dim, -1, self.layer_norm_eps)
            if self.use_norm
            else nn.Identity()
        )
        self._init_weights()

    def forward(self, x):
        x = self.conv(x)  # [B, L, C] --> [B, L/ps, self.emb_dim]
        if self.use_norm:
            x = self.norm(x)  # [B, L, C] --> [B, L/ps, self.emb_dim]
        return x

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.conv.weight)
        initializer.constant_(self.conv.bias, 0)


# class SelfAttnBlock(nn.Layer):
#     num_heads: int
#     emb_dim: int
#     mlp_ratio: int
#     layer_norm_eps: float = 1e-5

#     # @nn.compact
#     def __call__(self, inputs):
#         x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
#         x = nn.MultiHeadDotProductAttention(
#             num_heads=self.num_heads, qkv_features=self.emb_dim
#         )(x, x)
#         x = x + inputs

#         y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
#         y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

#         return x + y


class CrossAttnBlock(nn.Layer):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        mlp_ratio: int,
        layer_norm_eps: float = 1e-5,
        out_features: int = None,
        qkv_features: int = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.head_dim = self.emb_dim // self.num_heads
        self.layer_norm_q = LayerNorm(self.emb_dim, -1, epsilon=self.layer_norm_eps)
        self.layer_norm_kv = LayerNorm(self.emb_dim, -1, epsilon=self.layer_norm_eps)
        self.layer_norm_y = LayerNorm(self.emb_dim, -1, epsilon=self.layer_norm_eps)
        self.mlp = MlpBlock(self.emb_dim, self.emb_dim * self.mlp_ratio, self.emb_dim)
        self.attn = MultiHeadDotProductAttention(
            self.emb_dim,
            num_heads=num_heads,
            qkv_features=qkv_features,
            out_features=out_features,
        )

    def forward(self, q_inputs, kv_inputs):
        q = self.layer_norm_q(q_inputs)  # [B, L/ps, self.dec_emb_dim]
        kv = self.layer_norm_kv(kv_inputs)  # [B, N_grid, self.dec_emb_dim]
        q_reshape = q  # .reshape(q.shape[:-1] + [self.num_heads, self.head_dim])
        kv_reshape = kv  # .reshape(kv.shape[:-1] + [self.num_heads, self.head_dim])
        x = self.attn(q_reshape, kv_reshape)  # [B, L/ps, self.dec_emb_dim]
        x = x + q_inputs  # [B, L/ps, self.dec_emb_dim]
        y = self.layer_norm_y(x)  # [B, L/ps, self.dec_emb_dim]
        y = self.mlp(y)  # [B, L/ps, self.dec_emb_dim]
        return x + y  # [B, L/ps, self.dec_emb_dim]


class Encoder1D(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        seq_length: int,
        patch_size: int = (4,),
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.patch_embedding = PatchEmbed1D(in_dim, self.patch_size, self.emb_dim)

        self.blocks = nn.LayerList(
            [
                SelfAttnBlock(
                    self.num_heads,
                    self.emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                )
                for _ in range(self.depth)
            ]
        )
        self.register_buffer(
            "pos_emb",
            get_1d_sincos_pos_embed(
                self.emb_dim, self.seq_length // self.patch_size[0]
            ),
        )

    def forward(self, x):
        x = self.patch_embedding(x)  # [B, L/ps, self.emb_dim]
        x = x + self.pos_emb  # [B, L/ps, self.emb_dim]

        for i, block in enumerate(self.blocks):
            x = block(x)  # [B, L/ps, self.emb_dim]

        return x  # [B, L/ps, self.emb_dim]


def dot_product_attention_weights(
    query: paddle.Tensor,
    key: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
      query: queries for calculating attention with shape of `[batch..., q_length,
        num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch..., kv_length,
        num_heads, qk_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.

    Returns:
      Output of shape `[batch..., num_heads, q_length, kv_length]`.
    """
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / (depth**0.5)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = paddle.einsum("...qhd,...khd->...hqk", query, key)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    attn_weights = F.softmax(attn_weights).astype(dtype)

    # apply attention dropout
    return attn_weights


def dot_product_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
      query: queries for calculating attention with shape of `[batch..., q_length,
          num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch..., kv_length,
          num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch..., kv_length,
          num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
          incorporating causal masks, padding masks, proximity bias, etc.

    Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    # dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
    )

    # return weighted sum over values for each query position
    return paddle.einsum("...hqk,...khd->...qhd", attn_weights, value)


class MultiHeadDotProductAttention(nn.Layer):
    """Multi-head dot-product attention.

    Args:
        num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
        dtype: the dtype of the computation (default: infer from inputs and params)
        param_dtype: the dtype passed to parameter initializers (default: float32)
        qkv_features: dimension of the key, query, and value.
        out_features: dimension of the last projection
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rate: dropout rate
        deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer for the kernel of the Dense layers.
        bias_init: initializer for the bias of the Dense layers.
        use_bias: bool: whether pointwise QKVO dense transforms use bias.
        attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
        decode: whether to prepare and use an autoregressive cache.
        normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
    """

    def __init__(
        self,
        in_dim,
        num_heads: int,
        qkv_features: Optional[int] = None,
        out_features: Optional[int] = None,
        use_bias: bool = True,
        attention_fn: Callable[..., paddle.Tensor] = dot_product_attention,
        normalize_qk: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_features = qkv_features or in_dim
        self.out_features = out_features or in_dim
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        # self.decode = decode
        self.normalize_qk = normalize_qk
        assert self.qkv_features % self.num_heads == 0, (
            f"Memory dimension ({self.qkv_features}) must be divisible by number of"
            f" heads ({self.num_heads})."
        )
        self.head_dim = self.qkv_features // self.num_heads

        self.linear_q = nn.Linear(
            in_dim,
            self.qkv_features,
            bias_attr=use_bias,
        )
        self.linear_k = nn.Linear(
            in_dim,
            self.qkv_features,
            bias_attr=use_bias,
        )
        self.linear_v = nn.Linear(
            in_dim,
            self.qkv_features,
            bias_attr=use_bias,
        )
        self.query_ln = (
            LayerNorm(self.qkv_features, use_bias) if normalize_qk else nn.Identity()
        )
        self.key_ln = (
            LayerNorm(self.qkv_features, use_bias) if normalize_qk else nn.Identity()
        )
        self.linear_out = nn.Linear(
            self.qkv_features,
            self.out_features,
            bias_attr=use_bias,
        )

    def forward(
        self,
        inputs_q: paddle.Tensor,
        inputs_k: Optional[paddle.Tensor] = None,
        inputs_v: Optional[paddle.Tensor] = None,
    ):
        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    "`inputs_k` cannot be None if `inputs_v` is not None. "
                    "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                    "value to `inputs_k` and leave `inputs_v` as None."
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k
        elif inputs_v.shape[-1] == inputs_v.shape[-2]:
            logger.warn(
                f"You are passing an paddle.Tensor of shape {inputs_v.shape} "
                "to the `inputs_v` arg, when you may have intended "
                "to pass it to the `mask` arg. As of Flax version "
                "0.7.4, the function signature of "
                "MultiHeadDotProductAttention's `__call__` method "
                "has changed to `__call__(inputs_q, inputs_k=None, "
                "inputs_v=None"
                "deterministic=None)`. Use the kwarg `mask` instead. "
                "See https://github.com/google/flax/discussions/3389 "
                "and read the docstring for more information.",
                DeprecationWarning,
            )

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        q_attn_shape = inputs_q.shape
        q_attn_shape = q_attn_shape[:-1] + [self.num_heads, self.head_dim]
        k_attn_shape = inputs_k.shape
        k_attn_shape = k_attn_shape[:-1] + [self.num_heads, self.head_dim]
        query, key, value = (
            self.linear_q(inputs_q).reshape(q_attn_shape),
            self.linear_k(inputs_k).reshape(k_attn_shape),
            self.linear_v(inputs_v).reshape(k_attn_shape),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)  # type: ignore[call-arg]
            key = self.key_ln(key)  # type: ignore[call-arg]

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        x = x.flatten(-2)
        out = self.linear_out(x)
        # out = out.reshape(out.shape[:-1] + [self.num_heads, -1])
        return out


class CVit1D(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        seq_length: int,
        in_dim: int,
        coords_dim: int,
        patch_size: tuple = (4,),
        grid_size: tuple = (200,),
        latent_dim: int = 256,
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        dec_emb_dim: int = 256,
        dec_num_heads: int = 8,
        dec_depth: int = 1,
        num_mlp_layers: int = 1,
        mlp_ratio: int = 1,
        out_dim: int = 1,
        layer_norm_eps: float = 1e-5,
        embedding_type: str = "grid",
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.seq_length = seq_length
        self.in_dim = in_dim
        self.coords_dim = coords_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dec_emb_dim = dec_emb_dim
        self.dec_num_heads = dec_num_heads
        self.dec_depth = dec_depth
        self.num_mlp_layers = num_mlp_layers
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.layer_norm_eps = layer_norm_eps
        self.embedding_type = embedding_type

        if self.embedding_type == "grid":
            # Create grid and latents
            n_x = self.grid_size[0]
            self.grid = paddle.linspace(0, 1, n_x)
            self.latents = self.create_parameter(
                [n_x, self.latent_dim], default_initializer=nn.initializer.Normal()
            )
            self.fc = nn.Linear(self.latent_dim, self.dec_emb_dim)
            self.norm = LayerNorm(self.dec_emb_dim, -1, self.layer_norm_eps)
        elif self.embedding_type == "mlp":
            self.mlp = MlpBlock(self.latent_dim, self.dec_emb_dim, self.dec_emb_dim)
            self.norm = LayerNorm(self.dec_emb_dim, -1, self.layer_norm_eps)

        self.encoder = Encoder1D(
            self.in_dim,
            self.seq_length,
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.layer_norm_eps,
        )
        self.enc_norm = LayerNorm(self.emb_dim, -1, self.layer_norm_eps)
        self.fc1 = nn.Linear(self.emb_dim, self.dec_emb_dim)
        self.blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    # self.dec_emb_dim,
                    self.dec_num_heads,
                    self.dec_emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                    self.dec_emb_dim,
                    self.dec_emb_dim,
                )
                for _ in range(self.dec_depth)
            ]
        )
        self.block_norm = LayerNorm(self.dec_emb_dim, -1, self.layer_norm_eps)
        self.final_mlp = Mlp(
            self.num_mlp_layers,
            self.dec_emb_dim,
            self.out_dim,
            self.layer_norm_eps,
        )

    def forward_tensor(self, x, coords):
        """
        x: [B, L, C]
        coords: [N_grid, 1]
        """
        b, h, c = x.shape

        if self.embedding_type == "grid":
            d2 = (coords - self.grid[None, :]) ** 2
            w = paddle.exp(-1e5 * d2) / paddle.exp(-1e5 * d2).sum(axis=1, keepdim=True)
            coords = paddle.einsum("ic,pi->pc", self.latents, w)
            coords = self.fc(coords)  # [N_grid, self.dec_emb_dim]
            coords = self.norm(coords)  # [N_grid, self.dec_emb_dim]

        elif self.embedding_type == "mlp":
            coords = self.mlp(coords)  # [N_grid, self.dec_emb_dim]
            coords = self.norm(coords)  # [N_grid, self.dec_emb_dim]

        coords = einops.repeat(
            coords, "n d -> b n d", b=b
        )  # [B, N_grid, self.dec_emb_dim]
        x = self.encoder(x)  # [B, L/ps, self.emb_dim]
        x = self.enc_norm(x)  # [B, L/ps, self.emb_dim]
        x = self.fc1(x)  # [B, L/ps, self.dec_emb_dim]

        for _, block in enumerate(self.blocks):
            x = block(coords, x)  # [B, L/ps, self.dec_emb_dim]

        x = self.block_norm(x)  # [B, L/ps, self.dec_emb_dim]
        x = self.final_mlp(x)

        return x

    def forward(self, x_dict):
        if self._input_transform is not None:
            x = self._input_transform(x_dict)

        x, coords = x_dict[self.input_keys[0]], x_dict[self.input_keys[1]]

        y = self.forward_tensor(x, coords[0])

        y_dict = {self.output_keys[0]: y}
        if self._output_transform is not None:
            y_dict = self._output_transform(x_dict, y_dict)

        return y_dict


if __name__ == "__main__":
    model = CVit1D(
        seq_length=200,
        in_dim=1,
        coords_dim=1,
        patch_size=(4,),
        grid_size=(200,),
        latent_dim=256,
        emb_dim=256,
        depth=6,
        num_heads=16,
        dec_emb_dim=256,
        dec_num_heads=16,
        dec_depth=1,
        num_mlp_layers=1,
        mlp_ratio=1,
        out_dim=1,
        layer_norm_eps=1e-5,
        embedding_type="grid",
    )
    u = paddle.randn([256, 200, 1])  # [N, L, C]
    y = paddle.randn([128, 1])  # [N_grid, 1]
    s = model(u, y)
    print(s.shape)
