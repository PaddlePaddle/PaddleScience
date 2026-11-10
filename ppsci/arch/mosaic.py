# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Mosaic weather forecasting, adapted from maxxxzdn/mosaic (Apache-2.0).

Reference: (Sparse) Attention to the Details: Preserving Spectral Fidelity in
ML-based Weather Forecasting Models, ICML 2026, arXiv:2604.16429.
"""

import math
from dataclasses import dataclass

import numpy as np
import paddle
from einops import rearrange
from einops import repeat
from paddle import nn
from paddle.nn import functional as F

from ppsci.arch import base
from ppsci.arch.mosaic_attention import MosaicBlock
from ppsci.arch.mosaic_attention import RMSNorm


@dataclass
class StageConfig:
    nside: int
    dim: int
    num_heads: int
    block_attn_size: int
    sparse_block_size: int
    sparse_block_count: int
    encoder_depth: int
    decoder_depth: int
    mlp_ratio: float
    gqa_ratio: int


@dataclass
class BottleneckConfig:
    nside: int
    dim: int
    num_heads: int
    block_attn_size: int
    sparse_block_size: int
    sparse_block_count: int
    depth: int
    mlp_ratio: float
    gqa_ratio: int


@dataclass
class ModelConfig:
    dim: int
    num_heads: int
    k_neighbors: int
    qk_norm: bool
    rope: bool
    rope_theta: int
    sparse_every: int
    variables: list
    static_variables: list
    qkv_compress_ratio: int
    cg_stage_cfgs: list
    bottleneck_cfg: BottleneckConfig
    num_history_steps: int = 1
    noise_dim: int = 32
    ortho_init: bool = False
    rmsnorm_elementwise_affine: bool = True
    no_compression: bool = False


def healpix_grid(nside):
    # Optional geometry dependency is needed only when a Mosaic is constructed.
    import healpy as hp

    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True)
    return paddle.to_tensor(
        np.stack([np.rad2deg(phi), 90 - np.rad2deg(theta)], axis=-1).astype("float32")
    )


def rad_to_xyz(lonlat):
    lon, lat = lonlat[..., 0], lonlat[..., 1]
    return paddle.stack(
        [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], axis=-1
    )


class CrossAttentionInterpolate(nn.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.k_neighbors = cfg.k_neighbors
        self.enable_recompute = False
        self.num_heads = cfg.num_heads
        self.scale = (cfg.dim // cfg.num_heads) ** -0.5
        self.kv_norm = RMSNorm(cfg.dim, cfg.rmsnorm_elementwise_affine)
        self.to_q = nn.Linear(3, cfg.dim, bias_attr=False)
        self.to_kv = nn.Linear(cfg.dim, cfg.dim * 2, bias_attr=False)
        self.to_o = nn.Linear(cfg.dim, cfg.dim, bias_attr=False)
        self.register_buffer("neighbors", None, persistable=True)
        self.register_buffer("rel_pos", None, persistable=True)

    @paddle.no_grad()
    def initialize_interpolation_scheme(self, source, target):
        from sklearn.neighbors import BallTree

        tree = BallTree(source.numpy()[:, ::-1], metric="haversine")
        _, indices = tree.query(target.numpy()[:, ::-1], k=self.k_neighbors)
        self.neighbors = paddle.to_tensor(indices, dtype="int64")
        positions = rad_to_xyz(target)[:, None] - rad_to_xyz(source)[self.neighbors]
        self.rel_pos = F.normalize(positions, axis=-1, epsilon=1e-12)

    def forward(self, x):
        if self.neighbors is None:
            raise ValueError("Call initialize_interpolation before forecasting")
        q = rearrange(
            self.to_q(self.rel_pos), "s k (h d) -> s k 1 h d", h=self.num_heads
        )
        kv = rearrange(
            self.to_kv(self.kv_norm(x)),
            "s b (n h d) -> n s b h d",
            n=2,
            h=self.num_heads,
        )
        if self.enable_recompute and self.training and paddle.is_grad_enabled():
            from paddle.distributed.fleet.utils import recompute

            return paddle.concat(
                [
                    recompute(
                        self._interpolate_chunk,
                        q[start : start + 512],
                        kv,
                        self.neighbors[start : start + 512],
                        use_reentrant=False,
                    )
                    for start in range(0, q.shape[0], 512)
                ],
                axis=0,
            )
        return self._interpolate_chunk(q, kv, self.neighbors)

    def _interpolate_chunk(self, q, kv, neighbors):
        k, v = kv[0][neighbors], kv[1][neighbors]
        scores = (q * k).sum(axis=-1, keepdim=True) * self.scale
        weights = F.softmax(scores.astype("float32"), axis=1).astype(k.dtype)
        out = (weights * v).sum(axis=1)
        return self.to_o(rearrange(out, "s b h d -> s b (h d)"))


def relative_positions(fine_nside, coarse_nside):
    fine = rad_to_xyz(paddle.deg2rad(healpix_grid(fine_nside)))
    coarse = rad_to_xyz(paddle.deg2rad(healpix_grid(coarse_nside)))
    factor = (fine_nside // coarse_nside) ** 2
    rel = rearrange(fine, "(n f) d -> n f d", f=factor) - coarse[:, None]
    rel = rearrange(rel, "n f d -> n (f d)")
    return (rel - rel.mean(axis=0, keepdim=True)) / (
        rel.std(axis=0, keepdim=True) + 1e-6
    )


class HEALPixDownsample(nn.Layer):
    def __init__(self, before, after, affine):
        super().__init__()
        self.factor = (before.nside // after.nside) ** 2
        self.proj_x = nn.Linear(self.factor * before.dim, after.dim, bias_attr=False)
        self.proj_pos = nn.Linear(self.factor * 3, after.dim, bias_attr=False)
        self.norm = RMSNorm(after.dim, affine)
        self.register_buffer(
            "rel_pos", relative_positions(before.nside, after.nside), persistable=True
        )

    def forward(self, x):
        x = rearrange(x, "(n f) b c -> n b (f c)", f=self.factor)
        return self.norm(self.proj_x(x) + self.proj_pos(self.rel_pos).unsqueeze(1))


class HEALPixUpsample(nn.Layer):
    def __init__(self, before, after, affine):
        super().__init__()
        self.factor = (after.nside // before.nside) ** 2
        self.proj_x = nn.Linear(before.dim, after.dim * self.factor, bias_attr=False)
        self.proj_pos = nn.Linear(
            self.factor * 3, after.dim * self.factor, bias_attr=False
        )
        self.norm = RMSNorm(after.dim, affine)
        self.register_buffer(
            "rel_pos", relative_positions(after.nside, before.nside), persistable=True
        )

    def forward(self, x, shortcut):
        x = self.proj_x(x) + self.proj_pos(self.rel_pos).unsqueeze(1)
        return self.norm(
            rearrange(x, "n b (f d) -> (n f) b d", f=self.factor) + shortcut
        )


class NoiseGenerator(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_noise = nn.Linear(dim, dim, bias_attr=False)

    def forward(self, batch, dtype, noise=None):
        if noise is None:
            noise = paddle.randn([batch, self.dim], dtype=dtype)
        return self.to_noise(noise)


class UNetStage(nn.Layer):
    def __init__(self, cfg, stage, depth):
        super().__init__()
        self.nside = stage.nside
        self.enable_recompute = False
        self.blocks = nn.LayerList(
            [
                MosaicBlock(
                    cfg, stage, cfg.sparse_every <= 0 or i % cfg.sparse_every != 0
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, z=None):
        for block in self.blocks:
            if self.enable_recompute and self.training and paddle.is_grad_enabled():
                from paddle.distributed.fleet.utils import recompute

                # Frozen input projection in LoRA mode still needs an input
                # requiring gradients for reentrant parameter recomputation.
                x.stop_gradient = False
                # Reentrant outer blocks can contain non-reentrant chunk
                # recomputation; nested saved_tensors_hooks are unsupported.
                x = recompute(block, x, z, use_reentrant=True)
            else:
                x = block(x, z)
        return x


class Mosaic(base.Arch):
    """HEALPix U-Net Transformer with local/compressed/selected attention.

    Input dictionary keys default to ``state`` and ``time``. State has shape
    [B,N,history,lon,lat,channels], time [B,N,2]. Output ``forecast`` has shape
    [B,N*num_noise_samples,lon,lat,channels], preserving reference sample order.
    Optional input ``noise`` supplies standard-normal draws for reproducible
    cross-framework comparisons; framework RNG streams are not assumed equal.
    Static fields and grid geometry must be initialized before forward.

    ``qk_norm`` is kept as reference metadata: the released implementation does
    not apply it. Explicit attention uses gathered KV blocks, not Triton kernels.
    """

    def __init__(
        self,
        config,
        input_keys=("state", "time"),
        output_keys=("forecast",),
        num_noise_samples=1,
    ):
        super().__init__()
        if isinstance(config, dict):
            config = dict(config)
            config["cg_stage_cfgs"] = [
                StageConfig(**s) for s in config["cg_stage_cfgs"]
            ]
            config["bottleneck_cfg"] = BottleneckConfig(**config["bottleneck_cfg"])
            config = ModelConfig(**config)
        self.config = config
        self.input_keys, self.output_keys = input_keys, output_keys
        self.num_noise_samples = num_noise_samples
        self.noise_dim = config.noise_dim
        self.enable_recompute = False
        self._validate_config()
        self.nside = config.cg_stage_cfgs[0].nside
        dim = config.dim
        features = (
            len(config.variables) * config.num_history_steps
            + len(config.static_variables)
            + 7
        )
        affine = config.rmsnorm_elementwise_affine
        if config.noise_dim:
            self.noise_generator = NoiseGenerator(config.noise_dim)
        self.preprocess = nn.Sequential(
            nn.Linear(features, dim, bias_attr=False),
            RMSNorm(dim, affine),
            nn.Silu(),
            nn.Linear(dim, dim, bias_attr=False),
            RMSNorm(dim, affine),
        )
        self.interp_to_hp = CrossAttentionInterpolate(config)
        self.interp_to_ll = CrossAttentionInterpolate(config)
        stages = [*config.cg_stage_cfgs, config.bottleneck_cfg]
        self.encoder_stages = nn.LayerList()
        self.downsample_layers = nn.LayerList()
        for current, following in zip(stages[:-1], stages[1:]):
            self.encoder_stages.append(
                UNetStage(config, current, current.encoder_depth)
            )
            self.downsample_layers.append(HEALPixDownsample(current, following, affine))
        self.bottleneck = UNetStage(
            config, config.bottleneck_cfg, config.bottleneck_cfg.depth
        )
        self.decoder_stages, self.upsample_layers = nn.LayerList(), nn.LayerList()
        for current, following in reversed(list(zip(stages[:-1], stages[1:]))):
            self.upsample_layers.append(HEALPixUpsample(following, current, affine))
            self.decoder_stages.append(
                UNetStage(config, current, current.decoder_depth)
            )
        self.norm_before_interp_ll = RMSNorm(dim, affine)
        self.postprocess = nn.Sequential(
            RMSNorm(dim, affine),
            nn.Linear(dim, dim, bias_attr=False),
            nn.Silu(),
            nn.Linear(dim, len(config.variables), bias_attr=False),
        )
        self.register_buffer("static_vars", None, persistable=True)
        self.apply(self._initialize_weights)
        self._zero_init_residual_layers()
        self.initialize_rope()

    def _validate_config(self):
        cfg = self.config
        if not cfg.cg_stage_cfgs or cfg.dim != cfg.cg_stage_cfgs[0].dim:
            raise ValueError("The first HEALPix stage must have config.dim channels")
        if self.num_noise_samples < 1:
            raise ValueError("num_noise_samples must be positive")
        stages = [*cfg.cg_stage_cfgs, cfg.bottleneck_cfg]
        if cfg.qkv_compress_ratio < 1 or cfg.num_history_steps < 1:
            raise ValueError("Compression ratio and history length must be positive")
        if cfg.k_neighbors < 1 or cfg.noise_dim < 0:
            raise ValueError("k_neighbors must be positive and noise_dim nonnegative")
        for stage in stages:
            if stage.num_heads < 1 or stage.gqa_ratio < 1:
                raise ValueError("Attention head counts and GQA ratio must be positive")
            if stage.block_attn_size < 1 or stage.sparse_block_size < 1:
                raise ValueError("Attention block sizes must be positive")
            if stage.nside < 1 or stage.nside & (stage.nside - 1):
                raise ValueError("Nested HEALPix nside must be a power of two")
            if stage.dim % (stage.num_heads * cfg.qkv_compress_ratio):
                raise ValueError("dim must be divisible by heads * qkv_compress_ratio")
            if stage.num_heads % stage.gqa_ratio:
                raise ValueError("num_heads must be divisible by gqa_ratio")
            tokens = 12 * stage.nside**2
            if tokens % stage.block_attn_size or tokens % stage.sparse_block_size:
                raise ValueError(
                    "Attention block sizes must divide the HEALPix token count"
                )
            if not 0 <= stage.sparse_block_count <= tokens // stage.sparse_block_size:
                raise ValueError("sparse_block_count exceeds the available blocks")
            if (
                cfg.no_compression
                and cfg.sparse_every > 0
                and stage.sparse_block_count == 0
            ):
                raise ValueError("no_compression requires at least one selected block")
        for before, after in zip(stages[:-1], stages[1:]):
            if before.nside < after.nside or before.nside % after.nside:
                raise ValueError("Encoder nside must decrease by integer factors")

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            fan_in, fan_out = layer.weight.shape
            std = 1 / math.sqrt(fan_in) * min(1, math.sqrt(fan_out / fan_in))
            if self.config.ortho_init:
                nn.initializer.Orthogonal(gain=std)(layer.weight)
            else:
                nn.initializer.Normal(std=std)(layer.weight)

    def _zero_init_residual_layers(self):
        init = (
            nn.initializer.Orthogonal(gain=0.01)
            if self.config.ortho_init
            else nn.initializer.Normal(std=0.01)
        )
        for stage in [*self.encoder_stages, self.bottleneck, *self.decoder_stages]:
            for block in stage.blocks:
                init(block.attention.to_o.weight)
                init(block.ffn.w2.weight)
                if self.noise_dim:
                    nn.initializer.Normal(std=0.01)(block.ffn.noise_bias.weight)
        for up in self.upsample_layers:
            init(up.proj_x.weight)
            init(up.proj_pos.weight)
        if self.noise_dim:
            nn.initializer.Normal(std=0.01)(self.noise_generator.to_noise.weight)

    def set_recompute(self, enabled=True):
        """Opt in to block/interpolation and selected-chunk activation recomputation.

        Only active in training with gradients enabled. Does not change checkpoint
        keys or inference arithmetic. Explicit noise is sampled before recomputation.
        """
        self.enable_recompute = enabled
        self.interp_to_hp.enable_recompute = enabled
        self.interp_to_ll.enable_recompute = enabled
        for stage in [*self.encoder_stages, self.bottleneck, *self.decoder_stages]:
            stage.enable_recompute = enabled
            for block in stage.blocks:
                block.attention.enable_recompute = enabled

    def _interpolate(self, layer, x):
        if self.enable_recompute and self.training and paddle.is_grad_enabled():
            from paddle.distributed.fleet.utils import recompute

            return recompute(layer, x, use_reentrant=True)
        return layer(x)

    def initialize_rope(self):
        if self.config.rope:
            for stage in [*self.encoder_stages, self.bottleneck, *self.decoder_stages]:
                grid = healpix_grid(stage.nside)
                for block in stage.blocks:
                    block.attention.q_rope.initialize_rope(grid)
                    block.attention.k_rope.initialize_rope(grid)

    @paddle.no_grad()
    def initialize_interpolation(self, longitude, latitude):
        ll = paddle.deg2rad(
            paddle.stack(paddle.meshgrid(longitude, latitude), axis=-1).reshape([-1, 2])
        )
        hp = paddle.deg2rad(healpix_grid(self.nside))
        self.interp_to_hp.initialize_interpolation_scheme(ll, hp)
        self.interp_to_ll.initialize_interpolation_scheme(hp, ll)

    @paddle.no_grad()
    def initialize_static_vars(self, static_vars, longitude, latitude):
        ll = paddle.deg2rad(paddle.stack(paddle.meshgrid(longitude, latitude), axis=-1))
        fields = paddle.concat([static_vars, rad_to_xyz(ll)], axis=-1)
        # Large FP32 reductions lose precision on the full 240 x 121 grid.
        # Geometry is initialized once, so accumulate statistics in FP64.
        statistics = fields.astype("float64")
        normalized = (
            (statistics - statistics.mean(axis=(0, 1), keepdim=True))
            / (statistics.std(axis=(0, 1), keepdim=True) + 1e-6)
        ).astype(fields.dtype)
        self.static_vars = rearrange(normalized, "lon lat c -> (lon lat) 1 c")

    @staticmethod
    def time_embedding(time):
        day, year = time[:, :1], time[:, 1:2]
        return paddle.concat(
            [
                (2 * math.pi * day).sin(),
                (2 * math.pi * day).cos(),
                (2 * math.pi * year).sin(),
                (2 * math.pi * year).cos(),
            ],
            axis=-1,
        )

    def forward(self, inputs):
        if self._input_transform is not None:
            inputs = self._input_transform(inputs)
        x, time = [inputs[key] for key in self.input_keys]
        if self.static_vars is None:
            raise ValueError("Call initialize_static_vars before forecasting")
        b, n, _, lon, lat, _ = x.shape
        samples = self.num_noise_samples
        batch = b * samples * n
        z = (
            self.noise_generator(batch, x.dtype, inputs.get("noise"))
            if self.noise_dim
            else None
        )
        x = repeat(x, "b n t lon lat c -> (lon lat) (b s n) (t c)", s=samples)
        time = repeat(time, "b n d -> (b s n) d", s=samples)
        x = paddle.concat(
            [
                x,
                self.static_vars.expand([-1, batch, -1]),
                self.time_embedding(time)[None].expand([x.shape[0], -1, -1]),
            ],
            axis=-1,
        )
        x = self._interpolate(self.interp_to_hp, self.preprocess(x))
        skips = []
        for encoder, down in zip(self.encoder_stages, self.downsample_layers):
            x = encoder(x, z)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x, z)
        for decoder, up, skip in zip(
            self.decoder_stages, self.upsample_layers, reversed(skips)
        ):
            x = decoder(up(x, skip), z)
        x = self.postprocess(
            self._interpolate(self.interp_to_ll, self.norm_before_interp_ll(x))
        )
        x = rearrange(
            x,
            "(lon lat) (b n s) c -> b (n s) lon lat c",
            lon=lon,
            lat=lat,
            b=b,
            s=samples,
        )
        output = {self.output_keys[0]: x}
        if self._output_transform is not None:
            output = self._output_transform(inputs, output)
        return output
