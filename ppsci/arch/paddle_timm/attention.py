# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention

Hacked together by / Copyright 2020 Ross Wightman
# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24).

Linear is native paddle.nn.Linear ([in, out] weight layout; torch Linear
weights must be transposed when transferred). Attention is computed with
the manual q*scale @ k^T -> softmax -> @ v branch.
"""

from __future__ import annotations

from typing import Optional
from typing import Type

import paddle


def maybe_add_mask(scores, attn_mask: Optional = None):
    return scores if attn_mask is None else scores + attn_mask


class Attention(paddle.nn.Layer):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[paddle.nn.Layer]] = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            attn_head_dim: Dimension of each attention head. If None, computed as dim // num_heads.
            dim_out: Output dimension. If None, same as dim.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization to attention output before projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor for QK normalization if enabled.
        """
        super().__init__()
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert (
                norm_layer is not None
            ), "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim**-0.5
        self.qkv = paddle.nn.Linear(
            dim, self.attn_dim * 3, bias_attr=None if qkv_bias else False
        )
        self.q_norm = norm_layer(head_dim) if qk_norm else paddle.nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else paddle.nn.Identity()
        self.attn_drop = paddle.nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim) if scale_norm else paddle.nn.Identity()
        self.proj = paddle.nn.Linear(
            self.attn_dim, dim_out, bias_attr=None if proj_bias else False
        )
        self.proj_drop = paddle.nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional = None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
