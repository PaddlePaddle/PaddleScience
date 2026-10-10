# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Transformer Block

Hacked together by / Copyright 2020 Ross Wightman
# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24).

Attention is constructed directly (no attn-layer indirection) and
init_values must be None (LayerScale is Identity).
"""

from __future__ import annotations

from typing import Optional
from typing import Type

import paddle

from ppsci.arch.paddle_timm.attention import Attention
from ppsci.arch.paddle_timm.drop import DropPath
from ppsci.arch.paddle_timm.mlp import Mlp


class Block(paddle.nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[paddle.nn.Module] = paddle.nn.GELU,
        norm_layer: Type[paddle.nn.Module] = paddle.nn.LayerNorm,
    ) -> None:
        """Initialize Block.

        Args:
            embed_dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale (unsupported here, must be None).
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        assert init_values is None, "init_values must be None"
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = paddle.nn.Identity()
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )
        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = paddle.nn.Identity()
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )

    def forward(self, x, attn_mask: Optional = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
