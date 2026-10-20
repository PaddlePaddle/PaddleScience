"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

`AdaptiveLayerNorm` was inspired by the following file:

    https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101
"""

from __future__ import annotations

import paddle

__all__ = ["AdaptiveLayerNorm"]


class AdaptiveLayerNorm(paddle.nn.Layer):
    """Adaptive layer normalisation with scale and shift modulation."""

    def __init__(self, dim: int, context_dim: int, scale_bias: float = 0) -> None:
        super().__init__()
        self.ln = paddle.nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_modulation = paddle.nn.Sequential(
            paddle.nn.SiLU(), paddle.nn.Linear(context_dim, dim * 2)
        )
        self.scale_bias = scale_bias
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights."""
        paddle.nn.init.zeros_(self.ln_modulation[-1].weight)
        paddle.nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, x: paddle.Tensor, c: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            x (paddle.Tensor): Input tensor of shape `(B, L, D)`.
            c (paddle.Tensor): Conditioning tensor of shape `(B, D)` or `(B, L, D)`.

        Returns:
            paddle.Tensor: Output tensor of shape `(B, L, D)`.
        """
        c = self.ln_modulation(c)
        if c.ndim == 2:
            c = c.unsqueeze(1)
        shift, scale = c.chunk(2, dim=-1)
        return self.ln(x) * (self.scale_bias + scale) + shift
