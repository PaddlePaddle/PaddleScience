"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import einops
import paddle

from ppsci.arch.aurora.batch import Batch
from ppsci.arch.aurora.batch import Metadata
from ppsci.arch.aurora.model.fourier import levels_expansion
from ppsci.arch.aurora.model.levelcond import LevelConditioned
from ppsci.arch.aurora.model.perceiver import PerceiverResampler
from ppsci.arch.aurora.model.util import check_lat_lon_dtype
from ppsci.arch.aurora.model.util import init_weights
from ppsci.arch.aurora.model.util import unpatchify

__all__ = ["Perceiver3DDecoder"]


class LinearPatchReconstruction(paddle.nn.Linear):
    """Linear layer for patch reconstruction."""


class Perceiver3DDecoder(paddle.nn.Layer):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-05,
        level_condition: Optional[tuple[int | float, ...]] = None,
        separate_perceiver: tuple[str, ...] = (),
        modulation_heads: tuple[str, ...] = (),
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            embed_dim (int, optional): Embedding dim.. Defaults to `1024`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `1`.
            head_dim (int, optional): Dimension of the attention heads used in the aggregation
                blocks. Defaults to `64`.
            num_heads (int, optional): Number of attention heads used in the aggregation blocks.
                Defaults to `8`.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimensionality.
                Defaults to `4.0`.
            drop_rate (float, optional): Drop-out rate for input patches. Defaults to `0.0`.
            perceiver_ln_eps (float, optional): Layer norm. epsilon for the Perceiver blocks.
                Defaults to `1e-5`.
            level_condition (tuple[int | float, ...], optional): Make the patch embeddings dependent
                on pressure level. If you want to enable this feature, provide a tuple of all
                possible pressure levels.
            separate_perceiver (tuple[str, ...], optional): In the decoder, use a separate Perceiver
                for specific atmospheric variables. This can be helpful at fine-tuning time to deal
                with variables that have a significantly different behaviour. If you want to enable
                this features, set this to the collection of variables that should be run on a
                separate Perceiver.
            modulation_heads (tuple[str, ...], optional): Names of every variable for which to
                enable an additional head, the so-called modulation head, that can be used to
                predict the difference.
        """
        super().__init__()
        surf_vars += tuple(
            f"{name}_mod" for name in surf_vars if name in modulation_heads
        )
        atmos_vars += tuple(
            f"{name}_mod" for name in atmos_vars if name in modulation_heads
        )
        if modulation_heads:
            separate_perceiver += tuple(f"{name}_mod" for name in separate_perceiver)
        self.patch_size = patch_size
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.embed_dim = embed_dim
        self.level_condition = level_condition
        self.separate_perceiver = separate_perceiver
        self.modulation_heads = modulation_heads
        self.level_decoder = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            residual_latent=True,
            ln_eps=perceiver_ln_eps,
        )
        if self.separate_perceiver:
            self.level_decoder_alternate = PerceiverResampler(
                latent_dim=embed_dim,
                context_dim=embed_dim,
                depth=depth,
                head_dim=head_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                residual_latent=True,
                ln_eps=perceiver_ln_eps,
            )
        self.surf_heads = paddle.nn.LayerDict(
            {
                name: LinearPatchReconstruction(embed_dim, patch_size**2)
                for name in surf_vars
            }
        )
        if not self.level_condition:
            self.atmos_heads = paddle.nn.LayerDict(
                {
                    name: LinearPatchReconstruction(embed_dim, patch_size**2)
                    for name in atmos_vars
                }
            )
        else:
            self.atmos_heads = paddle.nn.LayerDict(
                {
                    name: LevelConditioned(
                        lambda: LinearPatchReconstruction(embed_dim, patch_size**2),
                        levels=self.level_condition,
                        levels_dim=-2,
                    )
                    for name in atmos_vars
                }
            )
        self.atmos_levels_embed = paddle.nn.Linear(embed_dim, embed_dim)
        self.apply(init_weights)

    def deaggregate_levels(
        self,
        level_embed: paddle.Tensor,
        x: paddle.Tensor,
        level_decoder: paddle.nn.Layer,
        max_batch_size: int = 2048,
    ) -> paddle.Tensor:
        """Deaggregate pressure level information.

        Args:
            level_embed (paddle.Tensor): Level embedding of shape `(B, L, C, D)`.
            x (paddle.Tensor): Aggregated input of shape `(B, L, C', D)`.
            level_decoder (nn.Module): Pressure level decoder.

        Returns:
            paddle.Tensor: Deaggregate output of shape `(B, L, C, D)`.
        """
        B, L, C, D = level_embed.shape
        level_embed = level_embed.flatten(0, 1)
        x = x.flatten(0, 1)
        _msg = f"Batch size mismatch. Found {level_embed.size(0)} and {x.size(0)}."
        assert level_embed.size(0) == x.size(0), _msg
        assert (
            len(level_embed.shape) == 3
        ), f"Expected 3 dims, found {level_embed.dims()}."
        assert x.dim() == 3, f"Expected 3 dims, found {x.dim()}."
        # Grid locations are independent in this Perceiver. Bound the entire
        # evaluation batch including its MLP/normalisation activations, keeping
        # all pressure levels and context latents at every location.
        if (
            not level_decoder.training
            and not paddle.is_grad_enabled()
            and x.shape[0] > max_batch_size
        ):
            x = paddle.concat(
                [
                    level_decoder(
                        level_embed[start : start + max_batch_size],
                        x[start : start + max_batch_size],
                    )
                    for start in range(0, x.shape[0], max_batch_size)
                ],
                axis=0,
            )
        else:
            x = level_decoder(level_embed, x)
        x = x.reshape(B, L, C, D)
        return x

    def forward(
        self,
        x: paddle.Tensor,
        batch: Batch,
        patch_res: tuple[int, int, int],
        lead_times: paddle.Tensor,
    ) -> Batch:
        """Forward pass.

        Args:
            x (paddle.Tensor): Backbone output of shape `(B, L, D)`.
            batch (:class:`aurora.Batch`): Batch to make predictions for.
            patch_res (tuple[int, int, int]): Patch resolution
            lead_times (paddle.Tensor): Lead times of shape `(batch,)` in hours.

        Returns:
            :class:`aurora.batch.Batch`: Prediction for `batch`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels
        surf_vars += tuple(
            f"{name}_mod" for name in surf_vars if name in self.modulation_heads
        )
        atmos_vars += tuple(
            f"{name}_mod" for name in atmos_vars if name in self.modulation_heads
        )
        B, L, D = x.shape
        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=paddle.float32), lon.to(dtype=paddle.float32)
        H, W = lat.shape[0], lon.shape[-1]
        x = einops.rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0],
            H=patch_res[1],
            W=patch_res[2],
        )
        x_surf = paddle.stack(
            [self.surf_heads[name](x[..., :1, :]) for name in surf_vars], axis=-1
        )
        x_surf = x_surf.reshape(*x_surf.shape[:3], -1)
        surf_preds = unpatchify(x_surf, len(surf_vars), H, W, self.patch_size)
        surf_preds = surf_preds.squeeze(2)
        atmos_levels_encode = levels_expansion(
            paddle.tensor(atmos_levels, device=x.device), self.embed_dim
        ).to(dtype=x.dtype)
        levels_embed = self.atmos_levels_embed(atmos_levels_encode)
        levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
        x_atmos = self.deaggregate_levels(
            levels_embed, x[..., 1:, :], self.level_decoder
        )
        if self.separate_perceiver:
            x_atmos_alternate = self.deaggregate_levels(
                levels_embed, x[..., 1:, :], self.level_decoder_alternate
            )
        else:
            x_atmos_alternate = x_atmos
        head_inputs = [
            (x_atmos if name not in self.separate_perceiver else x_atmos_alternate)
            for name in atmos_vars
        ]
        if not self.level_condition:
            x_atmos = paddle.stack(
                [self.atmos_heads[name](x) for name, x in zip(atmos_vars, head_inputs)],
                axis=-1,
            )
        else:
            x_atmos = paddle.stack(
                [
                    self.atmos_heads[name](x, levels=atmos_levels)
                    for name, x in zip(atmos_vars, head_inputs)
                ],
                axis=-1,
            )
        x_atmos = x_atmos.reshape(*x_atmos.shape[:3], -1)
        atmos_preds = unpatchify(x_atmos, len(atmos_vars), H, W, self.patch_size)
        pred_time = tuple(
            t + timedelta(hours=float(lead_times[i]))
            for i, t in enumerate(batch.metadata.time)
        )
        return Batch(
            {v: surf_preds[:, i] for i, v in enumerate(surf_vars)},
            batch.static_vars,
            {v: atmos_preds[:, i] for i, v in enumerate(atmos_vars)},
            Metadata(
                lat=lat,
                lon=lon,
                time=pred_time,
                atmos_levels=atmos_levels,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )
