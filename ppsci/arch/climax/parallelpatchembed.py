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

from __future__ import annotations

import math

import paddle

from ppsci.arch.paddle_timm import to_2tuple


def _get_conv2d_weights(in_channels, out_channels, kernel_size):
    weight = paddle.empty(out_channels, in_channels, *kernel_size)
    return weight


def _get_conv2d_biases(out_channels):
    bias = paddle.empty(out_channels)
    return bias


class ParallelVarPatchEmbed(paddle.nn.Layer):
    """Variable to Patch Embedding with multiple variables in a single kernel.
    Key idea is to use Grouped Convolutions.

    Args:
        max_vars (int): Maximum number of variables
        img_size (int): Image size
        patch_size (int): Patch size
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        flatten (bool, optional): Flatten the output. Defaults to True.
    """

    def __init__(
        self,
        max_vars: int,
        img_size,
        patch_size,
        embed_dim,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.max_vars = max_vars
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        grouped_weights = paddle.stack(
            [
                _get_conv2d_weights(1, embed_dim, self.patch_size)
                for _ in range(max_vars)
            ],
            dim=0,
        )
        self.proj_weights = paddle.nn.Parameter(grouped_weights)
        grouped_biases = paddle.stack(
            [_get_conv2d_biases(embed_dim) for _ in range(max_vars)], dim=0
        )
        self.proj_biases = paddle.nn.Parameter(grouped_biases)
        self.norm = norm_layer(embed_dim) if norm_layer else paddle.nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_vars):
            paddle.nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = paddle.nn.init._calculate_fan_in_and_fan_out(
                self.proj_weights[idx]
            )
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                paddle.nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, vars=None):
        B, C, H, W = x.shape
        if vars is None:
            vars = range(self.max_vars)
        weights = self.proj_weights[vars].flatten(0, 1)
        biases = self.proj_biases[vars].flatten(0, 1)
        groups = len(vars)
        proj = paddle.nn.functional.conv2d(
            x, weights, biases, groups=groups, stride=self.patch_size
        )
        if self.flatten:
            proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
            proj = proj.flatten(3).transpose(2, 3)
        proj = self.norm(proj)
        return proj
