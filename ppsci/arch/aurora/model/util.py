"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import math
from typing import TypeVar

import einops
import paddle

from ppsci.arch.aurora._layers import trunc_normal_

__all__ = [
    "unpatchify",
    "check_lat_lon_dtype",
    "maybe_adjust_windows",
    "init_weights",
    "fp16_safe_scaled_dot_product_attention",
    "scaled_dot_product_attention",
]


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    max_logit_elements=64 * 1024 * 1024,
):
    """Attention in Aurora's [batch, heads, tokens, dim] layout.

    Keep the explicit FP32/FP64 path for CPU and numerical comparisons. Paddle's
    fused low-precision GPU kernel expects [batch, tokens, heads, dim]. Large
    FP32/FP64 window batches are evaluated in independent chunks to bound the
    temporary score matrix; token interactions within each window are unchanged.
    """
    if query.place.is_gpu_place() and query.dtype in (paddle.float16, paddle.bfloat16):
        out = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            training=dropout_p > 0,
        )
        return out.transpose([0, 2, 1, 3])
    elements_per_batch = query.shape[1] * query.shape[2] * key.shape[2]
    chunk_size = max(1, max_logit_elements // elements_per_batch)
    if query.shape[0] > chunk_size:
        outputs = []
        for start in range(0, query.shape[0], chunk_size):
            stop = start + chunk_size
            mask = attn_mask
            if mask is not None and mask.ndim == 4 and mask.shape[0] == query.shape[0]:
                mask = mask[start:stop]
            outputs.append(
                fp16_safe_scaled_dot_product_attention(
                    query[start:stop],
                    key if key.shape[0] == 1 else key[start:stop],
                    value if value.shape[0] == 1 else value[start:stop],
                    attn_mask=mask,
                    dropout_p=dropout_p,
                )
            )
        return paddle.concat(outputs, axis=0)
    return fp16_safe_scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
    )


def unpatchify(x: paddle.Tensor, V: int, H: int, W: int, P: int) -> paddle.Tensor:
    """Unpatchify hidden representation.

    Args:
        x (paddle.Tensor): Patchified input of shape `(B, L, C, V * P^2)` where `P` is the
            patch size.
        V (int): Number of variables.
        H (int): Number of latitudes.
        W (int): Number of longitudes.
        P (int): Patch size.

    Returns:
        paddle.Tensor: Unpatchified representation of shape `(B, V, C, H, W)`.
    """
    assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D."
    B, C = x.size(0), x.size(2)
    H = H // P
    W = W // P
    assert x.size(1) == H * W
    assert x.size(-1) == V * P**2
    x = x.reshape(shape=(B, H, W, C, P, P, V))
    x = einops.rearrange(x, "B H W C P1 P2 V -> B V C H P1 W P2")
    x = x.reshape(shape=(B, V, C, H * P, W * P))
    return x


def check_lat_lon_dtype(lat: paddle.Tensor, lon: paddle.Tensor) -> None:
    """Assert that `lat` and `lon` are at least `float32`s."""
    assert lat.dtype in [
        paddle.float32,
        paddle.float64,
    ], f"Latitude num. unstable: {lat.dtype}."
    assert lon.dtype in [
        paddle.float32,
        paddle.float64,
    ], f"Longitude num. unstable: {lon.dtype}."


T = TypeVar("T", tuple[int, int], tuple[int, int, int])


def maybe_adjust_windows(window_size: T, shift_size: T, res: T) -> tuple[T, T]:
    """Adjust the window size and shift size if the input resolution is smaller than the window
    size."""
    err_msg = f"Expected same length, found {len(window_size)}, {len(shift_size)} and {len(res)}."
    assert len(window_size) == len(shift_size) == len(res), err_msg
    mut_shift_size, mut_window_size = list(shift_size), list(window_size)
    for i in range(len(res)):
        if res[i] <= window_size[i]:
            mut_shift_size[i] = 0
            mut_window_size[i] = res[i]
    new_window_size: T = tuple(mut_window_size)
    new_shift_size: T = tuple(mut_shift_size)
    assert (
        min(new_window_size) > 0
    ), f"Window size must be positive. Found {new_window_size}."
    assert (
        min(new_shift_size) >= 0
    ), f"Shift size must be non-negative. Found {new_shift_size}."
    return new_window_size, new_shift_size


def fp16_safe_scaled_dot_product_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attn_mask: (paddle.Tensor | None) = None,
    dropout_p: float = 0.0,
    scale: (float | None) = None,
) -> paddle.Tensor:
    """Scaled dot-product attention with float16 overflow protection.

    Equivalent to :func:`paddle.nn.functional.scaled_dot_product_attention`, but clamps intermediate
    attention weights when running in float16 to prevent overflow or inf values that can appear with
    large sequence lengths.
    """
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ (key.transpose(-2, -1) * scale_factor)
    if attn_weight.dtype == paddle.float16:
        max_val = paddle.finfo(attn_weight.dtype).max
        clamp_value = paddle.where(
            paddle.isinf(attn_weight).any(), max_val - 1000, max_val
        )
        attn_weight = paddle.clamp(attn_weight, min=-clamp_value, max=clamp_value)
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    attn_weight = paddle.nn.functional.softmax(attn_weight, axis=-1)
    if dropout_p > 0.0:
        attn_weight = paddle.nn.functional.dropout(
            x=attn_weight, p=dropout_p, training=True
        )
    return attn_weight @ value


def init_weights(m: paddle.nn.Layer):
    """Initialise weights of a module with a truncated normal distribution.

    `nn.LayerNorm` is initialised with a `weight` of 1 and a `bias` of 0.

    Args:
        m (paddle.nn.Layer): Module.
    """
    if isinstance(
        m,
        (
            paddle.nn.Linear,
            paddle.nn.Conv2D,
            paddle.nn.Conv3D,
            paddle.nn.Conv2DTranspose,
            paddle.nn.Conv3DTranspose,
        ),
    ):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0)
    elif isinstance(m, paddle.nn.LayerNorm):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)
