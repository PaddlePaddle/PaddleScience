# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

try:
    from einops import rearrange
    from einops import repeat
except ModuleNotFoundError:
    pass
from typing import Dict
from typing import Optional
from typing import Tuple

from ppsci.arch import base
from ppsci.arch.cvit import CrossAttnBlock
from ppsci.arch.cvit import MlpBlock
from ppsci.arch.cvit import MultiHeadDotProductAttention
from ppsci.arch.cvit import SelfAttnBlock
from ppsci.arch.mlp import FourierEmbedding
from ppsci.arch.mlp import PeriodEmbedding
from ppsci.utils import initializer


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


class PatchEmbed(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        patch_size: Tuple[int, ...] = (16, 16),
        emb_dim: int = 768,
        use_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        self.layer_norm_eps = layer_norm_eps
        self.conv = nn.Conv2D(
            in_dim,
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1]),
            (self.patch_size[0], self.patch_size[1]),
            data_format="NHWC",
        )
        if self.use_norm:
            self.norm = nn.LayerNorm(self.emb_dim, self.layer_norm_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.conv.weight)
        initializer.constant_(self.conv.bias, 0)

    def forward(self, x):
        b, h, w, c = x.shape
        x = self.conv(x)  # [B, L, C] --> [B, L/ps, self.emb_dim]
        x = x.reshape([b, -1, self.emb_dim])
        if self.use_norm:
            x = self.norm(x)
        return x


class PerceiverBlock(nn.Layer):
    def __init__(
        self,
        emb_dim: int,
        depth: int,
        num_heads: int = 8,
        num_latents: int = 64,
        mlp_ratio: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.latents = self.create_parameter(
            [self.num_latents, self.emb_dim],
            default_initializer=nn.initializer.Normal(std=1e-2),
        )
        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.emb_dim, self.layer_norm_eps)

    def forward(self, x: paddle.Tensor):
        latents = repeat(self.latents, "l d -> b l d", b=x.shape[0])  # (B, L', D)
        for i in range(self.depth):
            latents = self.cross_attn_blocks[i](latents, x)

        latents = self.norm(latents)
        return latents


class Mlp(nn.Layer):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        out_dim: int,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
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
        self.linear_out = nn.Linear(self.hidden_dim, self.out_dim)

        self._init_weights()

    def forward(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = self.linears[i](x)
            x = self.gelu(x)

        x = self.linear_out(x)
        return x

    def _init_weights(self) -> None:
        for linear in self.linears:
            initializer.xavier_uniform_(linear.weight)
            initializer.constant_(linear.bias, 0)


class Encoder(base.Arch):
    def __init__(
        self,
        in_dim: int,
        patch_size: Tuple[int, ...],
        grid_size: Tuple[int, ...],
        emb_dim: int,
        num_latents: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.emb_dim = emb_dim
        self.num_latents = num_latents
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps

        self.patch_embedding = PatchEmbed(in_dim, patch_size, self.emb_dim)

        pos_emb = get_2d_sincos_pos_embed(
            self.emb_dim,
            (
                self.grid_size[0] // self.patch_size[0],
                self.grid_size[1] // self.patch_size[1],
            ),
        )
        self.pos_emb = self.create_parameter(
            pos_emb.shape, default_initializer=nn.initializer.Assign(pos_emb)
        )
        self.perceive_block = PerceiverBlock(
            emb_dim=self.emb_dim,
            depth=2,
            num_heads=self.num_heads,
            num_latents=self.num_latents,
        )
        self.norm1 = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)
        self.self_attn_blocks = nn.LayerList(
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
        self.norm2 = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        b, h, w, c = x.shape

        # Patch embedding
        x = self.patch_embedding(x)

        # Interpolate positional embeddings to match the input shape
        pos_emb_interp = self.pos_emb.reshape(
            [
                1,
                self.grid_size[0] // self.patch_size[0],
                self.grid_size[1] // self.patch_size[1],
                self.emb_dim,
            ]
        )
        pos_emb_interp = F.interpolate(
            pos_emb_interp,
            [h // self.patch_size[0], w // self.patch_size[1]],
            mode="bilinear",
            data_format="NHWC",
        )
        pos_emb_interp = rearrange(pos_emb_interp, "b h w d -> b (h w) d")
        x = x + pos_emb_interp

        # Embed into tokens of the same length as the latents
        x = self.perceive_block(x)
        x = self.norm1(x)

        # Transformer
        for _, block in enumerate(self.self_attn_blocks):
            x = block(x)
        x = self.norm2(x)
        return x


class Decoder(base.Arch):
    def __init__(
        self,
        in_dim: int,
        fourier_freq: float = 1.0,
        period: Optional[Dict[str, Tuple[float, bool]]] = None,
        dec_depth: int = 2,
        dec_num_heads: int = 8,
        dec_emb_dim: int = 256,
        mlp_ratio: int = 1,
        out_dim: int = 1,
        num_mlp_layers: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.fourier_freq = fourier_freq
        self.period = period
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.dec_emb_dim = dec_emb_dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_mlp_layers = num_mlp_layers
        self.layer_norm_eps = layer_norm_eps

        if self.period is not None:
            self.period_embed = PeriodEmbedding(period)

        self.fourier_embed = FourierEmbedding(
            in_dim if self.period is None else in_dim * 2,
            self.dec_emb_dim,
            self.fourier_freq,
        )

        self.fc = nn.Linear(self.dec_emb_dim, self.dec_emb_dim)

        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
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
        self.block_norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)
        self.final_mlp = Mlp(
            self.num_mlp_layers,
            self.dec_emb_dim,
            self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )

    def forward(self, x, coords):
        b, n, c = x.shape

        # Embed periodic boundary conditions if specified
        if self.period is True:
            # Hardcode the periodicity, assuming the domain is [0, 1]x[0, 1]
            coords = self.period_embed(coords)

        coords = self.fourier_embed(coords)
        if paddle.in_dynamic_mode():
            assert coords.ndim == 3, coords.shape
            assert coords.shape[0] == 1, coords.shape
        coords = paddle.expand(coords, [b, *coords.shape[1:]])
        # coords = repeat(coords, "d -> b n d", n=1, b=b)

        x = self.fc(x)

        for i in range(self.dec_depth):
            coords = self.cross_attn_blocks[i](coords, x)

        x = self.block_norm(coords)
        x = self.final_mlp(x)

        return x


class FAE(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        encoder: Encoder,
        decoder: Decoder,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: Dict[str, paddle.Tensor]):
        coords, x = batch[self.input_keys[0]], batch[self.input_keys[1]]
        # coords: [1, num_query_points, 2]
        # x: [b, h, w, 1]
        # y: [b, num_query_points, 1]
        z = self.encoder(x)  # [b, l, c]

        u_pred = self.decoder(z, coords)
        return {self.output_keys[0]: u_pred}


#####################
# DiT modules below #
#####################


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Layer):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.fc1 = nn.Linear(
            self.emb_dim,
            6 * self.emb_dim,
        )
        initializer.zeros_(self.fc1.weight)
        initializer.zeros_(self.fc1.bias)

        self.ln1 = nn.LayerNorm(self.emb_dim, weight_attr=False, bias_attr=False)
        self.ln2 = nn.LayerNorm(self.emb_dim, weight_attr=False, bias_attr=False)

        self.attn = MultiHeadDotProductAttention(self.emb_dim, self.num_heads)
        for layer in self.attn.sublayers():
            if isinstance(layer, nn.Linear):
                # initializer.xavier_uniform_(layer.weight)
                initializer.lecun_normal_(layer.weight)
                initializer.zeros_(layer.bias)

        self.mlp_block = MlpBlock(
            self.emb_dim, self.emb_dim * self.mlp_ratio, self.emb_dim
        )

    def forward(self, x, c):
        # Calculate adaLn modulation parameters.
        c = F.gelu(c, approximate=True)  # (B, emb_dim)
        c = self.fc1(c)  # (B, 6* emb_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = paddle.split(
            c, 6, axis=-1
        )  # (B, emb_dim)

        # Attention Residual.
        x_norm = self.ln1(x)  # [B, L, C]
        x_modulated = modulate(x_norm, shift_msa, scale_msa)  # [B, L, C]
        attn_x = self.attn(x_modulated, x_modulated)  # [B, L, C]
        x = x + (gate_msa[:, None] * attn_x)  # [B, L, C]

        # MLP Residual.
        x_norm2 = self.ln2(x)  # [B, L, C]
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)  # [B, L, C]
        mlp_x = self.mlp_block(x_modulated2)  # [B, L, C]
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


class TimestepEmbedder(nn.Layer):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        emb_dim: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(self.frequency_embedding_size, self.emb_dim)
        initializer.normal_(self.fc1.weight, std=0.02)
        initializer.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(self.emb_dim, self.emb_dim)
        initializer.normal_(self.fc2.weight, std=0.02)
        initializer.zeros_(self.fc2.bias)

    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = t.astype(paddle.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = paddle.exp(
            -np.log(max_period)
            * paddle.arange(start=0, end=half, dtype=paddle.float32)
            / half
        )  # [half]
        args = t[:, None] * freqs[None]  # [N, 1] * [1, half] => [N, half]
        embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
        return embedding


class DiT(base.Arch):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        seq_len: int,
        out_dim: int,
        with_condition: bool,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.with_condition = with_condition
        pos_emb = get_1d_sincos_pos_embed(
            self.emb_dim,
            self.seq_len,
        )
        self.pos_emb = self.create_parameter(
            pos_emb.shape, default_initializer=nn.initializer.Assign(pos_emb)
        )
        self.fc1 = nn.Linear(self.in_dim, self.emb_dim)
        if with_condition:
            self.fc2 = nn.Linear(self.in_dim, self.emb_dim)

        self.timestep_embedder = TimestepEmbedder(self.emb_dim)

        self.dit_blocks = nn.LayerList(
            [
                DiTBlock(
                    self.emb_dim,
                    self.num_heads,
                    self.mlp_ratio,
                )
                for _ in range(self.depth)
            ]
        )
        self.ln = nn.LayerNorm(self.emb_dim)
        self.fc_out = nn.Linear(self.emb_dim, self.out_dim)

    def forward(self, x, t, c=None):
        # (x = (B, L, C) image, t = (B,) timesteps, c = (B, L, C) conditioning
        assert x.ndim == 3, x.shape
        assert t.ndim == 1, t.shape
        assert t.shape[0] == x.shape[0], f"t.shape: {t.shape}, x.shape: {x.shape}"
        if c is not None:
            assert c.ndim == 3, c.shape
            assert c.shape == x.shape, f"c.shape: {c.shape}, x.shape: {x.shape}"

        x = self.fc1(x)
        x = x + self.pos_emb

        if c is not None:
            c = self.fc2(c)
            x = x + c

        t = self.timestep_embedder(t)  # (B, emb_dim)

        for i in range(self.depth):
            x = self.dit_blocks[i](x, t)

        x = self.ln(x)
        x = self.fc_out(x)

        return x
