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

from typing import Tuple

import numpy as np
import paddle

from ppsci.arch.climax.arch import ClimaX
from ppsci.arch.climax.pos_embed import get_1d_sincos_pos_embed_from_grid


class ClimaXClimateBench(ClimaX):
    """ClimaX variant for the ClimateBench climate projection task.

    The encoder consumes `[B, T, Vi, H, W]` inputs, aggregates over the time
    dimension with cross-attention, and the head regresses one `[H, W]` map
    (Vo is fixed to 1) from a single global token.

    Args:
        input_keys (Tuple[str, ...]): Keys of the input dict. The first key holds
            the input variables tensor `[B, T, Vi, H, W]`, the second key holds
            the lead times tensor `[B]` or `[B, 1]`.
        output_keys (Tuple[str, ...]): Keys of the output dict. The first key
            holds the prediction tensor `[B, 1, H, W]`.
        default_vars (list): List of input variables the embeddings are built for.
        out_vars (list | str): Output variable name(s), kept as metadata.
        img_size (list): Image size of the input data.
        time_history (int): Number of input time steps.
        patch_size (int): Patch size of the input data.
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer layers.
        decoder_depth (int): Number of decoder layers (base head, replaced here).
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension.
        drop_path (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        parallel_patch_embed (bool): Whether to use parallel patch embedding.
        freeze_encoder (bool): Whether to freeze the transformer blocks except
            their normalization parameters.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.ClimaXClimateBench(
        ...     input_keys=("x", "lead_times"),
        ...     output_keys=("tas",),
        ...     default_vars=["CO2", "SO2", "CH4", "BC"],
        ...     out_vars="tas",
        ...     img_size=[32, 64],
        ...     time_history=10,
        ...     embed_dim=32,
        ...     depth=1,
        ...     num_heads=4,
        ... )
        >>> out = model({"x": paddle.rand([1, 10, 4, 32, 64]), "lead_times": paddle.rand([1, 1])})
        >>> print(list(out["tas"].shape))
        [1, 1, 32, 64]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        default_vars,
        out_vars,
        img_size=[32, 64],
        time_history=1,
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        freeze_encoder=False,
    ):
        assert out_vars is not None
        super().__init__(
            input_keys,
            output_keys,
            default_vars,
            img_size,
            patch_size,
            embed_dim,
            depth,
            decoder_depth,
            num_heads,
            mlp_ratio,
            drop_path,
            drop_rate,
            parallel_patch_embed,
            out_vars=out_vars,
        )
        self.time_history = time_history
        self.freeze_encoder = freeze_encoder
        self.time_pos_embed = paddle.nn.Parameter(
            paddle.zeros(1, time_history, embed_dim), requires_grad=True
        )
        self.time_agg = paddle.compat.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.time_query = paddle.nn.Parameter(
            paddle.zeros(1, 1, embed_dim), requires_grad=True
        )
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.time_pos_embed.shape[-1], np.arange(self.time_history)
        )
        self.time_pos_embed.data.copy_(
            paddle.from_numpy(time_pos_embed).float().unsqueeze(0)
        )
        self.head = paddle.compat.nn.Linear(embed_dim, img_size[0] * img_size[1])
        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                if "norm" in name:
                    continue
                else:
                    p.requires_grad_(False)

    def forward_encoder(self, x: paddle.Tensor, lead_times: paddle.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)
        else:
            # Index the ModuleList with Python ints looked up from var_map so
            # the to-static tracer can resolve the layer selection; values and
            # loop order are identical to var_ids.
            id_list = [int(self.var_map[var]) for var in variables]
            for i in range(len(id_list)):
                embeds.append(self.token_embeds[id_list[i]](x[:, i : i + 1]))
            x = paddle.stack(embeds, dim=1)
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = self.aggregate_variables(x)
        x = x + self.pos_embed
        x = x.unflatten(0, sizes=(b, t))
        x = x + self.time_pos_embed.unsqueeze(2)
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2)
        x = x + lead_time_emb
        x = x.flatten(0, 1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.unflatten(0, sizes=(b, t))
        x = x.mean(-2)
        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.time_agg(time_query, x, x)
        return x

    def forward(self, input_dict):
        """Forward pass through the model.

        Args:
            input_dict (Dict[str, paddle.Tensor]): The first input key holds the
                input variables `[B, T, Vi, H, W]`, the second key holds the
                lead times `[B]` or `[B, 1]`.

        Returns:
            Dict[str, paddle.Tensor]: Prediction `[B, 1, H, W]` keyed by the
                first output key.
        """
        if self._input_transform is not None:
            input_dict = self._input_transform(input_dict)

        x = input_dict[self.input_keys[0]]
        lead_times = input_dict[self.input_keys[1]]
        # The lead-time embedding consumes a flat [B] tensor; [B, 1] input
        # (batched 1-element lead times) is flattened here.
        lead_times = lead_times.reshape([-1])
        x = self.forward_encoder(x, lead_times, self.default_vars)
        preds = self.head(x)
        preds = preds.reshape(-1, 1, self.img_size[0], self.img_size[1])
        output_dict = {self.output_keys[0]: preds}

        if self._output_transform is not None:
            output_dict = self._output_transform(input_dict, output_dict)
        return output_dict
