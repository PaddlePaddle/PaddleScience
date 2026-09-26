# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache
from typing import Tuple

import numpy as np
import paddle

from ppsci.arch import base
from ppsci.arch.climax.parallelpatchembed import ParallelVarPatchEmbed
from ppsci.arch.climax.pos_embed import get_1d_sincos_pos_embed_from_grid
from ppsci.arch.climax.pos_embed import get_2d_sincos_pos_embed
from ppsci.arch.paddle_timm import Block
from ppsci.arch.paddle_timm import PatchEmbed
from ppsci.arch.paddle_timm import trunc_normal_


class ClimaX(base.Arch):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        input_keys (Tuple[str, ...]): Keys of the input dict. The first key holds
            the input variables tensor `[B, Vi, H, W]`, the second key holds the
            lead times tensor `[B]` or `[B, 1]`.
        output_keys (Tuple[str, ...]): Keys of the output dict. The first key
            holds the predicted variables tensor `[B, Vo, H, W]`.
        default_vars (list): List of input variables the embeddings are built for.
        out_vars (list | str, optional): Output variables selected from the
            predicted channels; must be a subset of `default_vars` (the head
            predicts one channel per default variable). Defaults to all of
            `default_vars`.
        img_size (list): Image size of the input data.
        patch_size (int): Patch size of the input data.
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer layers.
        decoder_depth (int): Number of decoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension.
        drop_path (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        parallel_patch_embed (bool): Whether to use parallel patch embedding.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.ClimaX(
        ...     input_keys=("x", "lead_times"),
        ...     output_keys=("CO2",),
        ...     default_vars=["CO2", "SO2", "CH4", "BC"],
        ...     out_vars=("CO2",),
        ...     img_size=[32, 64],
        ...     embed_dim=32,
        ...     depth=1,
        ...     num_heads=4,
        ... )
        >>> out = model({"x": paddle.rand([1, 4, 32, 64]), "lead_times": paddle.rand([1, 1])})
        >>> print(list(out["CO2"].shape))
        [1, 1, 32, 64]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        out_vars=None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        if out_vars is None:
            out_vars = default_vars
        if isinstance(out_vars, str):
            out_vars = [out_vars]
        self.out_vars = tuple(out_vars)
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(
                len(default_vars), img_size, patch_size, embed_dim
            )
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = paddle.nn.ModuleList(
                [
                    PatchEmbed(img_size, patch_size, 1, embed_dim)
                    for i in range(len(default_vars))
                ]
            )
            self.num_patches = self.token_embeds[0].num_patches
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.var_query = paddle.nn.Parameter(
            paddle.zeros(1, 1, embed_dim), requires_grad=True
        )
        self.var_agg = paddle.compat.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.pos_embed = paddle.nn.Parameter(
            paddle.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = paddle.compat.nn.Linear(1, embed_dim)
        self.pos_drop = paddle.nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path, depth)]
        self.blocks = paddle.nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=paddle.nn.LayerNorm,
                    proj_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = paddle.nn.LayerNorm(embed_dim)
        self.head = paddle.nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(paddle.compat.nn.Linear(embed_dim, embed_dim))
            self.head.append(paddle.nn.GELU())
        self.head.append(
            paddle.compat.nn.Linear(embed_dim, len(self.default_vars) * patch_size**2)
        )
        self.head = paddle.nn.Sequential(*self.head)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(paddle.from_numpy(pos_embed).float().unsqueeze(0))
        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(paddle.from_numpy(var_embed).float().unsqueeze(0))
        if self.parallel_patch_embed:
            for i in range(len(self.token_embeds.proj_weights)):
                w = self.token_embeds.proj_weights[i].data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        else:
            for i in range(len(self.token_embeds)):
                w = self.token_embeds[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.compat.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                paddle.nn.init.constant_(m.bias, 0)
        elif isinstance(m, paddle.nn.LayerNorm):
            paddle.nn.init.constant_(m.bias, 0)
            paddle.nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = paddle.nn.Parameter(
            paddle.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return paddle.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: paddle.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = paddle.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: paddle.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = paddle.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)
        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)
        x = x.squeeze()
        x = x.unflatten(dim=0, sizes=(b, l))
        return x

    def forward_encoder(self, x: paddle.Tensor, lead_times: paddle.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                # paddle LayerList indexing requires a Python int
                # (var_ids[i] is a 0-dim tensor)
                embeds.append(self.token_embeds[int(id)](x[:, i : i + 1]))
            x = paddle.stack(embeds, dim=1)
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = self.aggregate_variables(x)
        x = x + self.pos_embed
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, input_dict):
        """Forward pass through the model.

        Args:
            input_dict (Dict[str, paddle.Tensor]): The first input key holds the
                input weather/climate variables `[B, Vi, H, W]`, the second key
                holds the lead times `[B]` or `[B, 1]`.

        Returns:
            Dict[str, paddle.Tensor]: Predicted weather/climate variables
                `[B, Vo, H, W]` keyed by the first output key.
        """
        if self._input_transform is not None:
            input_dict = self._input_transform(input_dict)

        x = input_dict[self.input_keys[0]]
        lead_times = input_dict[self.input_keys[1]]
        # The lead-time embedding consumes a flat [B] tensor; [B, 1] input
        # (batched 1-element lead times) is flattened here.
        lead_times = lead_times.reshape([-1])
        out_transformers = self.forward_encoder(x, lead_times, self.default_vars)
        preds = self.head(out_transformers)
        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(self.out_vars, preds.device)
        preds = preds[:, out_var_ids]
        output_dict = {self.output_keys[0]: preds}

        if self._output_transform is not None:
            output_dict = self._output_transform(input_dict, output_dict)
        return output_dict
