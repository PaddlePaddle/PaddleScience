"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

from datetime import timezone
from typing import Optional

import einops
import numpy as np
import paddle

from ppsci.arch.aurora.batch import Batch
from ppsci.arch.aurora.model.fourier import absolute_time_expansion
from ppsci.arch.aurora.model.fourier import lead_time_expansion
from ppsci.arch.aurora.model.fourier import lead_time_expansion_v3
from ppsci.arch.aurora.model.fourier import levels_expansion
from ppsci.arch.aurora.model.fourier import pos_expansion
from ppsci.arch.aurora.model.fourier import scale_expansion
from ppsci.arch.aurora.model.levelcond import LevelConditioned
from ppsci.arch.aurora.model.patchembed import LevelPatchEmbed
from ppsci.arch.aurora.model.perceiver import MLP
from ppsci.arch.aurora.model.perceiver import PerceiverResampler
from ppsci.arch.aurora.model.posencoding import pos_scale_enc
from ppsci.arch.aurora.model.util import check_lat_lon_dtype
from ppsci.arch.aurora.model.util import init_weights

__all__ = ["Perceiver3DEncoder"]


class Perceiver3DEncoder(paddle.nn.Layer):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: (tuple[str, ...] | None),
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.0,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-05,
        stabilise_level_agg: bool = False,
        level_condition: Optional[tuple[int | float, ...]] = None,
        dynamic_vars: bool = False,
        atmos_static_vars: bool = False,
        simulate_indexing_bug: bool = False,
        use_updated_lead_time_embedding: bool = False,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.0`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to `1e-5`.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            level_condition (tuple[int | float, ...], optional): Make the patch embeddings dependent
                on pressure level. If you want to enable this feature, provide a tuple of all
                possible pressure levels.
            dynamic_vars (bool, optional): Use dynamically generated static variables, like time
                of day. Defaults to `False`.
            atmos_static_vars (bool, optional): Also concatenate the static variables to the
                atmospheric variables. Defaults to `False`.
            simulate_indexing_bug (bool, optional): Simulate an indexing bug that's present for the
                air pollution version of Aurora. This is necessary to obtain numerical equivalence
                to the original implementation. Defaults to `False`.
            use_updated_lead_time_embedding (bool, optional): Whether to use the updated lead time
                embedding with a minimum wavelength of 6 hours. Defaults to `False`.
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.level_condition = level_condition
        self.dynamic_vars = dynamic_vars
        self.atmos_static_vars = atmos_static_vars
        self.simulate_indexing_bug = simulate_indexing_bug
        if self.dynamic_vars:
            if static_vars is None:
                static_vars = ()
            static_vars += (
                "tod_cos",
                "tod_sin",
                "dow_cos",
                "dow_sin",
                "doy_cos",
                "doy_sin",
            )
        if static_vars:
            surf_vars += static_vars
            if self.atmos_static_vars:
                atmos_vars += tuple(f"static_{v}" for v in static_vars)
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        self.atmos_latents = paddle.nn.Parameter(
            paddle.randn(latent_levels - 1, embed_dim)
        )
        self.surf_level_encoding = paddle.nn.Parameter(paddle.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = paddle.nn.LayerNorm(embed_dim)
        self.pos_embed = paddle.nn.Linear(embed_dim, embed_dim)
        self.scale_embed = paddle.nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = paddle.nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = paddle.nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = paddle.nn.Linear(embed_dim, embed_dim)
        self.use_updated_lead_time_embedding = use_updated_lead_time_embedding
        assert max_history_size > 0, "At least one history step is required."
        self.surf_token_embeds = LevelPatchEmbed(
            surf_vars, patch_size, embed_dim, max_history_size
        )
        if not self.level_condition:
            self.atmos_token_embeds = LevelPatchEmbed(
                atmos_vars, patch_size, embed_dim, max_history_size
            )
        else:
            self.atmos_token_embeds = LevelConditioned(
                lambda: LevelPatchEmbed(
                    atmos_vars, patch_size, embed_dim, max_history_size
                ),
                levels=self.level_condition,
                levels_dim=-5,
            )
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
        )
        self.pos_drop = paddle.nn.Dropout(p=drop_rate)
        self.apply(init_weights)
        paddle.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        paddle.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: paddle.Tensor) -> paddle.Tensor:
        """Aggregate pressure level information.

        Args:
            x (paddle.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            paddle.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)
        x = paddle.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)
        latents = paddle.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)
        x = self.level_agg(latents, x)
        x = x.unflatten(dim=0, sizes=(B, L))
        x = paddle.einsum("blcd->bcld", x)
        return x

    def forward(self, batch: Batch, lead_times: paddle.Tensor) -> paddle.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`aurora.Batch`): Batch to encode.
            lead_times (:class:`paddle.Tensor`): Lead times of shape `(batch,)` in hours.

        Returns:
            paddle.Tensor: Encoding of shape `(B, L, D)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels
        x_surf = paddle.stack(tuple(batch.surf_vars.values()), axis=2)
        x_static = paddle.stack(tuple(batch.static_vars.values()), axis=2)
        x_atmos = paddle.stack(tuple(batch.atmos_vars.values()), axis=2)
        B, T, _, C, H, W = x_atmos.size()
        assert x_surf.shape[:2] == (
            B,
            T,
        ), f"Expected shape {B, T}, got {x_surf.shape[:2]}."
        if static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
            if self.dynamic_vars:
                ones = paddle.ones(
                    (1, T, 1, H, W), device=x_static.device, dtype=x_static.dtype
                )
                time = batch.metadata.time
                x_dynamic = paddle.cat(
                    [
                        paddle.cat(
                            (
                                ones * np.cos(2 * np.pi * time[b].hour / 24),
                                ones * np.sin(2 * np.pi * time[b].hour / 24),
                                ones * np.cos(2 * np.pi * time[b].weekday() / 7),
                                ones * np.sin(2 * np.pi * time[b].weekday() / 7),
                                ones * np.cos(2 * np.pi * time[b].day / 365.25),
                                ones * np.sin(2 * np.pi * time[b].day / 365.25),
                            ),
                            axis=-3,
                        )
                        for b in range(B)
                    ],
                    axis=0,
                )
                dynamic_vars = (
                    "tod_cos",
                    "tod_sin",
                    "dow_cos",
                    "dow_sin",
                    "doy_cos",
                    "doy_sin",
                )
                x_surf = paddle.cat((x_surf, x_static, x_dynamic), axis=2)
                surf_vars = surf_vars + static_vars + dynamic_vars
                if self.atmos_static_vars:
                    atmos_vars += tuple(
                        f"static_{v}" for v in static_vars + dynamic_vars
                    )
                    inds = -1, -1, -1, len(atmos_levels), -1, -1
                    x_atmos = paddle.cat(
                        (
                            x_atmos,
                            x_static[..., None, :, :].expand(*inds),
                            x_dynamic[..., None, :, :].expand(*inds),
                        ),
                        axis=2,
                    )
            else:
                x_surf = paddle.cat((x_surf, x_static), axis=2)
                surf_vars = surf_vars + static_vars
                if self.atmos_static_vars:
                    atmos_vars = atmos_vars + static_vars
                    x_atmos = paddle.cat(
                        (
                            x_atmos,
                            x_static[..., None, :, :].expand(
                                -1, -1, -1, len(atmos_levels), -1, -1
                            ),
                        ),
                        axis=2,
                    )
        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=paddle.float32), lon.to(dtype=paddle.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W
        x_surf = einops.rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_token_embeds(x_surf, surf_vars)
        dtype = x_surf.dtype
        if self.simulate_indexing_bug and "z" in atmos_vars:
            i_z = atmos_vars.index("z")
            i_static_z = atmos_vars.index("static_z")
            x_atmos = paddle.cat(
                (
                    x_atmos[:, :, :i_static_z],
                    x_atmos[:, :, i_z : i_z + 1],
                    x_atmos[:, :, i_static_z + 1 :],
                ),
                axis=2,
            )
        if not self.level_condition:
            x_atmos = einops.rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
            x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
            x_atmos = einops.rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)
        else:
            x_atmos = einops.rearrange(x_atmos, "b t v c h w -> b c v t h w")
            x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars, levels=atmos_levels)
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))
        atmos_levels_tensor = paddle.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(
            dtype=dtype
        )
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[
            None, :, None, :
        ]
        x_atmos = x_atmos + atmos_levels_embed
        x_atmos = self.aggregate_levels(x_atmos)
        x = paddle.cat((x_surf.unsqueeze(1), x_atmos), axis=1)
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode
        x = x.reshape(B, -1, self.embed_dim)
        expansion = (
            lead_time_expansion_v3
            if self.use_updated_lead_time_embedding
            else lead_time_expansion
        )
        lead_time_encode = expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        x = x + lead_time_emb.unsqueeze(1)
        # Meteorological times without tzinfo represent UTC. datetime.timestamp()
        # otherwise silently applies the host timezone (and historical DST).
        absolute_times_list = [
            (t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)).timestamp()
            / 3600
            for t in batch.metadata.time
        ]
        absolute_times = paddle.tensor(
            absolute_times_list, dtype=paddle.float32, device=x.device
        )
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(
            absolute_time_encode.to(dtype=dtype)
        )
        x = x + absolute_time_embed.unsqueeze(1)
        x = self.pos_drop(x)
        return x
