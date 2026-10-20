"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import math
from typing import Optional

import paddle

from ppsci.arch.aurora._layers import to_2tuple

__all__ = ["LevelPatchEmbed"]


class LevelPatchEmbed(paddle.nn.Layer):
    """At either the surface or at a single pressure level, maps all variables into a single
    embedding."""

    def __init__(
        self,
        var_names: tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        history_size: int = 1,
        norm_layer: Optional[paddle.nn.Layer] = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.var_names = var_names
        self.kernel_size = (history_size,) + to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.weights = paddle.nn.ParameterDict(
            {
                name: paddle.nn.Parameter(paddle.empty(embed_dim, 1, *self.kernel_size))
                for name in var_names
            }
        )
        self.bias = paddle.nn.Parameter(paddle.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else paddle.nn.Identity()
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        for weight in self.weights.values():
            paddle.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = paddle.nn.init._calculate_fan_in_and_fan_out(
            next(iter(self.weights.values()))
        )
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            paddle.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: paddle.Tensor, var_names: tuple[str, ...]) -> paddle.Tensor:
        """Run the embedding.

        Args:
            x (:class:`paddle.Tensor`): Tensor to embed of a shape of `(B, V, T, H, W)`.
            var_names (tuple[str, ...]): Names of the variables in `x`. The length should be equal
                to `V`.

        Returns:
            :class:`paddle.Tensor`: Embedded tensor a shape of `(B, L, D]) if flattened,
                where `L = H * W / P^2`. Otherwise, the shape is `(B, D, H', W')`.

        """
        B, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[0]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[1]} != 0."
        assert len(set(var_names)) == len(
            var_names
        ), f"{var_names} contains duplicates."
        weight = paddle.cat(
            [self.weights[name][:, :, :T, ...] for name in var_names], axis=1
        )
        stride = (T,) + self.kernel_size[1:]
        proj = paddle.nn.functional.conv3d(x, weight, self.bias, stride=stride)
        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)
            proj = proj.transpose(1, 2)
        x = self.norm(proj)
        return x
