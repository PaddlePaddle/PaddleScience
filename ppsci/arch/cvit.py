# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

try:
    import einops
except ModuleNotFoundError:
    pass
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple

import paddle
from paddle import nn
from paddle.nn import functional as F

from ppsci.arch import base
from ppsci.utils import initializer


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: paddle.Tensor):
    if embed_dim % 2 != 0:
        raise ValueError(f"embedding dimension({embed_dim}) must be divisible by 2")

    omega = paddle.arange(embed_dim // 2, dtype=paddle.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape([-1])  # (M,)
    out = paddle.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = paddle.sin(out)  # (M, D/2)
    emb_cos = paddle.cos(out)  # (M, D/2)

    emb = paddle.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, length: int):
    return paddle.unsqueeze(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, paddle.arange(length, dtype=paddle.float32)
        ),
        0,
    )


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int]):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        if embed_dim % 2 != 0:
            raise ValueError(f"embedding dimension({embed_dim}) must be divisible by 2")

        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = paddle.concat([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = paddle.arange(grid_size[0], dtype=paddle.float32)
    grid_w = paddle.arange(grid_size[1], dtype=paddle.float32)
    grid = paddle.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = paddle.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return paddle.unsqueeze(pos_embed, 0)


class MlpBlock(nn.Layer):
    def __init__(self, in_dim: int, dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.dim)
        self.act = nn.GELU(True)
        self.linear2 = nn.Linear(self.dim, self.out_dim)

        self._init_weights()

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.act(x)
        x = self.linear2(x)
        return x

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.linear1.weight)
        initializer.constant_(self.linear1.bias, 0)
        initializer.xavier_uniform_(self.linear2.weight)
        initializer.constant_(self.linear2.bias, 0)


class SelfAttnBlock(nn.Layer):
    def __init__(
        self, num_heads: int, emb_dim: int, mlp_ratio: int, layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.mlp_ratio = mlp_ratio
        self.layer_norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.attn_layer = MultiHeadDotProductAttention(
            self.emb_dim,
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
        )
        self.layer_norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.mlp = MlpBlock(self.emb_dim, self.emb_dim * self.mlp_ratio, self.emb_dim)

    def forward(self, inputs):
        # inputs:  # [B, L/ps, self.emb_dim]
        x = self.layer_norm1(inputs)
        x = self.attn_layer(x, x)
        x = x + inputs
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class Mlp(nn.Layer):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        out_dim: int,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_norm_eps = layer_norm_eps
        self.linears = nn.LayerList(
            [
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.gelu = nn.GELU(True)
        self.norms = nn.LayerList(
            [
                nn.LayerNorm(self.hidden_dim, self.layer_norm_eps)
                for _ in range(self.num_layers)
            ]
        )

        self.linear_out = nn.Linear(self.hidden_dim, self.out_dim)

        self._init_weights()

    def forward(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            y = self.linears[i](x)
            y = self.gelu(y)
            x = x + y
            x = self.norms[i](x)

        x = self.linear_out(x)
        return x

    def _init_weights(self) -> None:
        for linear in self.linears:
            initializer.xavier_uniform_(linear.weight)
            initializer.constant_(linear.bias, 0)


class PatchEmbed1D(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        patch_size: Sequence[int] = (4,),
        emb_dim: int = 768,
        use_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        self.layer_norm_eps = layer_norm_eps
        self.conv = nn.Conv1D(
            in_dim,
            self.emb_dim,
            self.patch_size[0],
            self.patch_size[0],
            data_format="NLC",
        )
        self.norm = (
            nn.LayerNorm(self.emb_dim, self.layer_norm_eps)
            if self.use_norm
            else nn.Identity()
        )
        self._init_weights()

    def forward(self, x):
        x = self.conv(x)  # [B, L, C] --> [B, L/ps, self.emb_dim]
        if self.use_norm:
            x = self.norm(x)
        return x

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.conv.weight)
        initializer.constant_(self.conv.bias, 0)


class PatchEmbed(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        spatial_dims: Sequence[int],
        patch_size: Tuple[int, ...] = (1, 16, 16),
        emb_dim: int = 768,
        use_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        self.layer_norm_eps = layer_norm_eps
        self.conv = nn.Conv3D(
            in_dim,
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            data_format="NDHWC",
        )
        self.norm = (
            nn.LayerNorm(self.emb_dim, self.layer_norm_eps)
            if self.use_norm
            else nn.Identity()
        )
        t, h, w = spatial_dims
        self.num_patches = [
            t // self.patch_size[0],
            h // self.patch_size[1],
            w // self.patch_size[2],
        ]
        self._init_weights()

    def forward(self, x):
        b, t, h, w, c = x.shape

        x = self.conv(x)  # [B, L, C] --> [B, L/ps, self.emb_dim]
        x = x.reshape(
            [
                b,
                self.num_patches[0],
                self.num_patches[1] * self.num_patches[2],
                self.emb_dim,
            ]
        )
        if self.use_norm:
            x = self.norm(x)
        return x

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.conv.weight)
        initializer.constant_(self.conv.bias, 0)


class CrossAttnBlock(nn.Layer):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        mlp_ratio: int,
        layer_norm_eps: float = 1e-5,
        out_features: int = None,
        qkv_features: int = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.head_dim = self.emb_dim // self.num_heads

        self.layer_norm_q = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)
        self.layer_norm_kv = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)

        self.attn_layer = MultiHeadDotProductAttention(
            self.emb_dim,
            num_heads=num_heads,
            qkv_features=qkv_features,
            out_features=out_features,
        )
        self.layer_norm_y = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)
        self.mlp = MlpBlock(self.emb_dim, self.emb_dim * self.mlp_ratio, self.emb_dim)

    def forward(self, q_inputs, kv_inputs):
        # [B, L/ps, self.dec_emb_dim]
        q = self.layer_norm_q(q_inputs)
        kv = self.layer_norm_kv(kv_inputs)
        x = self.attn_layer(q, kv)
        x = x + q_inputs
        y = self.layer_norm_y(x)
        y = self.mlp(y)
        return x + y


class Encoder1D(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        spatial_dims: int,
        patch_size: int = (4,),
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.patch_embedding = PatchEmbed1D(in_dim, self.patch_size, self.emb_dim)

        self.self_attn_blocks = nn.LayerList(
            [
                SelfAttnBlock(
                    self.num_heads,
                    self.emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                )
                for _ in range(self.depth)
            ]
        )
        pos_emb = get_1d_sincos_pos_embed(
            self.emb_dim, self.spatial_dims // self.patch_size[0]
        )
        self.pos_emb = self.create_parameter(
            pos_emb.shape, default_initializer=nn.initializer.Assign(pos_emb)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.pos_emb

        for _, block in enumerate(self.self_attn_blocks):
            x = block(x)

        return x


class TimeAggregation(nn.Layer):
    def __init__(
        self,
        emb_dim: int,
        depth: int,
        num_heads: int = 8,
        num_latents: int = 64,
        mlp_ratio: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.latents = self.create_parameter(
            [self.num_latents, self.emb_dim],
            default_initializer=nn.initializer.Normal(std=1e-2),
        )
        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x):  # (B, T, S, D) --> (B, T', S, D)
        latents = einops.repeat(
            self.latents, "t d -> b s t d", b=x.shape[0], s=x.shape[2]
        )  # (B, T', S, D)
        x = einops.rearrange(x, "b t s d -> b s t d")  # (B, S, T, D)

        # Transformer
        for i, block in enumerate(self.cross_attn_blocks):
            latents = block(latents, x)

        latents = einops.rearrange(latents, "b s t d -> b t s d")  # (B, T', S, D)
        return latents


class Encoder(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        spatial_dims: Sequence[int],
        patch_size: int = (1, 16, 16),
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.patch_embedding = PatchEmbed(
            in_dim, spatial_dims, self.patch_size, self.emb_dim
        )

        self.time_aggreator = TimeAggregation(
            self.emb_dim,
            2,
            self.num_heads,
            1,
            self.mlp_ratio,
            self.layer_norm_eps,
        )
        self.norm = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)

        self.self_attn_blocks = nn.LayerList(
            [
                SelfAttnBlock(
                    self.num_heads,
                    self.emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                )
                for _ in range(self.depth)
            ]
        )
        t, h, w = spatial_dims

        time_emb = get_1d_sincos_pos_embed(self.emb_dim, t // self.patch_size[0])
        self.time_emb = self.create_parameter(
            time_emb.shape, default_initializer=nn.initializer.Assign(time_emb)
        )

        pos_emb = get_2d_sincos_pos_embed(
            self.emb_dim, (h // self.patch_size[1], w // self.patch_size[2])
        )
        self.pos_emb = self.create_parameter(
            pos_emb.shape, default_initializer=nn.initializer.Assign(pos_emb)
        )

    def forward(self, x):
        # patchify
        x = self.patch_embedding(x)

        # add positional embedding
        x = x + self.time_emb.unsqueeze(2) + self.pos_emb.unsqueeze(1)

        # aggregate along time dimension
        x = self.time_aggreator(x)
        x = self.norm(x)
        x = einops.rearrange(x, "b t s d -> b (t s) d")

        for _, block in enumerate(self.self_attn_blocks):
            x = block(x)

        return x


def dot_product_attention_weights(
    query: paddle.Tensor,
    key: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
        query: queries for calculating attention with shape of [batch..., q_length,
            num_heads, qk_depth_per_head].
        key: keys for calculating attention with shape of [batch..., kv_length,
            num_heads, qk_depth_per_head].
        bias: bias for the attention weights. This should be broadcastable to the
            shape [batch..., num_heads, q_length, kv_length]. This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.

    Returns:
        Output of shape [batch..., num_heads, q_length, kv_length].
    """
    dtype = query.dtype

    if paddle.in_dynamic_mode():
        assert query.ndim == key.ndim, "q, k must have same rank."
        assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
        assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
        assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / (depth**0.5)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = paddle.einsum("...qhd,...khd->...hqk", query, key)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    attn_weights = F.softmax(attn_weights).astype(dtype)

    # apply attention dropout
    return attn_weights


def dot_product_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
        query: queries for calculating attention with shape of [batch..., q_length,
            num_heads, qk_depth_per_head].
        key: keys for calculating attention with shape of [batch..., kv_length,
            num_heads, qk_depth_per_head].
        value: values to be used in attention with shape of [batch..., kv_length,
            num_heads, v_depth_per_head].
        bias: bias for the attention weights. This should be broadcastable to the
            shape [batch..., num_heads, q_length, kv_length]. This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.

    Returns:
        paddle.Tensor: Output of shape [batch..., q_length, num_heads, v_depth_per_head].
    """
    if paddle.in_dynamic_mode():
        assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
        assert (
            query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
        ), "q, k, v batch dims must match."
        assert (
            query.shape[-2] == key.shape[-2] == value.shape[-2]
        ), "q, k, v num_heads must match."
        assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
    )

    # return weighted sum over values for each query position
    return paddle.einsum("...hqk,...khd->...qhd", attn_weights, value)


class MultiHeadDotProductAttention(nn.Layer):
    """Multi-head dot-product attention.

    Args:
        in_dim: Number of input dimensions.
        num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
            should be divisible by the number of heads.
        qkv_features: dimension of the key, query, and value.
        out_features: dimension of the last projection
        use_bias: bool: whether pointwise QKVO dense transforms use bias.
        attention_fn: dot_product_attention or compatible function. Accepts query,
            key, value, and returns output of shape [bs, dim1, dim2, ..., dimN,,
            num_heads, value_channels]`
        normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
    """

    def __init__(
        self,
        in_dim,
        num_heads: int,
        qkv_features: Optional[int] = None,
        out_features: Optional[int] = None,
        use_bias: bool = True,
        attention_fn: Callable[..., paddle.Tensor] = dot_product_attention,
        normalize_qk: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_features = qkv_features or in_dim
        self.out_features = out_features or in_dim
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.normalize_qk = normalize_qk
        assert self.qkv_features % self.num_heads == 0, (
            f"Memory dimension ({self.qkv_features}) must be divisible by number of"
            f" heads ({self.num_heads})."
        )
        self.head_dim = self.qkv_features // self.num_heads

        self.linear_q = nn.Linear(
            in_dim,
            self.qkv_features,
            bias_attr=use_bias,
        )
        self.linear_k = nn.Linear(
            in_dim,
            self.qkv_features,
            bias_attr=use_bias,
        )
        self.linear_v = nn.Linear(
            in_dim,
            self.qkv_features,
            bias_attr=use_bias,
        )
        self.query_ln = (
            nn.LayerNorm(self.qkv_features) if normalize_qk else nn.Identity()
        )
        self.key_ln = nn.LayerNorm(self.qkv_features) if normalize_qk else nn.Identity()
        self.linear_out = nn.Linear(
            self.qkv_features,
            self.out_features,
            bias_attr=use_bias,
        )

    def forward(
        self,
        inputs_q: paddle.Tensor,
        inputs_kv: Optional[paddle.Tensor] = None,
    ):
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        q_attn_shape = inputs_q.shape
        q_attn_shape = q_attn_shape[:-1] + [self.num_heads, self.head_dim]

        kv_attn_shape = inputs_kv.shape
        kv_attn_shape = kv_attn_shape[:-1] + [self.num_heads, self.head_dim]
        query, key, value = (
            self.linear_q(inputs_q).reshape(q_attn_shape),
            self.linear_k(inputs_kv).reshape(kv_attn_shape),
            self.linear_v(inputs_kv).reshape(kv_attn_shape),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)
            key = self.key_ln(key)

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
        )
        # back to the original inputs dimensions
        x = x.reshape(x.shape[:-2] + [x.shape[-2] * x.shape[-1]])
        out = self.linear_out(x)
        return out


class CVit1D(base.Arch):
    """
    1D Convolutional Vision Transformer (CVit1D) class.

    [Bridging Operator Learning and Conditioned Neural Fields: A Unifying Perspective](https://arxiv.org/abs/2405.13998)

    Args:
        input_keys (Sequence[str]): Keys identifying the input tensors.
        output_keys (Sequence[str]): Keys identifying the output tensors.
        spatial_dims (int): The spatial dimensions of the input data.
        in_dim (int): The dimensionality of the input data.
        coords_dim (int): The dimensionality of the positional encoding.
        patch_size (Sequence[int], optional): Size of the patches. Defaults to (4,).
        grid_size (Sequence[int], optional): Size of the grid. Defaults to (200,).
        latent_dim (int, optional): Dimensionality of the latent space. Defaults to 256.
        emb_dim (int, optional): Dimensionality of the embedding space. Defaults to 256.
        depth (int, optional): Number of transformer encoder layers. Defaults to 3.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dec_emb_dim (int, optional): Dimensionality of the decoder embedding space. Defaults to 256.
        dec_num_heads (int, optional): Number of decoder attention heads. Defaults to 8.
        dec_depth (int, optional): Number of decoder transformer layers. Defaults to 1.
        num_mlp_layers (int, optional): Number of layers in the MLP. Defaults to 1.
        mlp_ratio (int, optional): Ratio for determining the size of the MLP's hidden layer. Defaults to 1.
        out_dim (int, optional): Dimensionality of the output data. Defaults to 1.
        layer_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
        embedding_type (str, optional): Type of embedding to use ("grid" or other options). Defaults to "grid".

    Examples:
        >>> import ppsci
        >>> b, l, c = 2, 32, 1
        >>> l_query = 42
        >>> c_in = 1
        >>> c_out = 1
        >>> model = ppsci.arch.CVit1D(
        ...     input_keys=["u", "y"],
        ...     output_keys=["s"],
        ...     in_dim=c_in,
        ...     coords_dim=1,
        ...     spatial_dims=l,
        ...     patch_size=[4],
        ...     grid_size=[l],
        ...     latent_dim=32,
        ...     emb_dim=32,
        ...     depth=3,
        ...     num_heads=8,
        ...     dec_emb_dim=32,
        ...     dec_num_heads=8,
        ...     dec_depth=1,
        ...     num_mlp_layers=1,
        ...     mlp_ratio=1,
        ...     out_dim=c_out,
        ...     layer_norm_eps=1e-5,
        ...     embedding_type="grid",
        ... )
        >>> x = paddle.randn([b, l, c_in])
        >>> coords = paddle.randn([l_query, 1])
        >>> out = model({"u": x, "y": coords})["s"]
        >>> print(out.shape) # output shape should be [b, l_query, c_out]
        [2, 42, 1]
    """

    def __init__(
        self,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        spatial_dims: int,
        in_dim: int,
        coords_dim: int,
        patch_size: Sequence[int] = (4,),
        grid_size: Sequence[int] = (200,),
        latent_dim: int = 256,
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        dec_emb_dim: int = 256,
        dec_num_heads: int = 8,
        dec_depth: int = 1,
        num_mlp_layers: int = 1,
        mlp_ratio: int = 1,
        out_dim: int = 1,
        layer_norm_eps: float = 1e-5,
        embedding_type: str = "grid",
    ):
        if not importlib.util.find_spec("einops"):
            raise ModuleNotFoundError(
                "Please install `einops` by running 'pip install einops'."
            )
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.spatial_dims = spatial_dims
        self.in_dim = in_dim
        self.coords_dim = coords_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dec_emb_dim = dec_emb_dim
        self.dec_num_heads = dec_num_heads
        self.dec_depth = dec_depth
        self.num_mlp_layers = num_mlp_layers
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.layer_norm_eps = layer_norm_eps
        self.embedding_type = embedding_type

        if self.embedding_type == "grid":
            # Create grid and latents
            n_x = self.grid_size[0]
            self.grid = paddle.linspace(0, 1, n_x)
            self.latents = self.create_parameter(
                [n_x, self.latent_dim],
                default_initializer=nn.initializer.Normal(std=1e-2),
            )
            self.fc = nn.Linear(self.latent_dim, self.dec_emb_dim)
            self.norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)
        elif self.embedding_type == "mlp":
            self.mlp = MlpBlock(self.latent_dim, self.dec_emb_dim, self.dec_emb_dim)
            self.norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)

        self.encoder = Encoder1D(
            self.in_dim,
            self.spatial_dims,
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.layer_norm_eps,
        )
        self.enc_norm = nn.LayerNorm(self.emb_dim, self.layer_norm_eps)
        self.fc1 = nn.Linear(self.emb_dim, self.dec_emb_dim)
        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    self.dec_num_heads,
                    self.dec_emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                    self.dec_emb_dim,
                    self.dec_emb_dim,
                )
                for _ in range(self.dec_depth)
            ]
        )
        self.block_norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)
        self.final_mlp = Mlp(
            self.num_mlp_layers,
            self.dec_emb_dim,
            self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )

    def forward_tensor(self, x, coords):
        b, h, c = x.shape

        # process query coordinates
        if self.embedding_type == "grid":
            d2 = (coords - self.grid.unsqueeze(0)) ** 2
            w = paddle.exp(-1e5 * d2) / paddle.exp(-1e5 * d2).sum(axis=1, keepdim=True)
            coords = paddle.einsum("ic,pi->pc", self.latents, w)
            coords = self.fc(coords)
            coords = self.norm(coords)
        elif self.embedding_type == "mlp":
            coords = self.mlp(coords)
            coords = self.norm(coords)

        coords = einops.repeat(coords, "n d -> b n d", b=b)

        # process input function(encoder)
        x = self.encoder(x)
        x = self.enc_norm(x)
        x = self.fc1(x)

        # decoder
        for i, block in enumerate(self.cross_attn_blocks):
            x = block(coords, x)

        # mlp
        x = self.block_norm(x)
        x = self.final_mlp(x)

        return x

    def forward(self, x_dict):
        if self._input_transform is not None:
            x = self._input_transform(x_dict)

        x, coords = x_dict[self.input_keys[0]], x_dict[self.input_keys[1]]
        if coords.ndim >= 3:
            coords = coords[0]  # [b, n, c] -> [n, c]

        y = self.forward_tensor(x, coords)

        y_dict = {self.output_keys[0]: y}
        if self._output_transform is not None:
            y_dict = self._output_transform(x_dict, y_dict)

        return y_dict


class CVit(base.Arch):
    """
    CVit architecture.

    [Bridging Operator Learning and Conditioned Neural Fields: A Unifying Perspective](https://arxiv.org/abs/2405.13998)

    Args:
        input_keys (Sequence[str]): Input keys.
        output_keys (Sequence[str]): Output keys.
        in_dim (int): Dimensionality of the input data.
        coords_dim (int): Dimensionality of the coordinates.
        spatial_dims (Sequence[int]): Spatial dimensions.
        patch_size (Sequence[int], optional): Size of the patches. Defaults to (1, 16, 16).
        grid_size (Sequence[int], optional): Size of the grid. Defaults to (128, 128).
        latent_dim (int, optional): Dimensionality of the latent space. Defaults to 256.
        emb_dim (int, optional): Dimensionality of the embedding space. Defaults to 256.
        depth (int, optional): Number of transformer encoder layers. Defaults to 3.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dec_emb_dim (int, optional): Dimensionality of the decoder embedding space. Defaults to 256.
        dec_num_heads (int, optional): Number of decoder attention heads. Defaults to 8.
        dec_depth (int, optional): Number of decoder transformer layers. Defaults to 1.
        num_mlp_layers (int, optional): Number of MLP layers. Defaults to 1.
        mlp_ratio (int, optional): Ratio of hidden units. Defaults to 1.
        out_dim (int, optional): Dimensionality of the output. Defaults to 1.
        layer_norm_eps (float, optional): Epsilon value for layer normalization. Defaults to 1e-5.
        embedding_type (str, optional): Type of embedding. Defaults to "grid".

    Examples:
        >>> import ppsci
        >>> b, t, h, w, c_in = 2, 4, 8, 8, 3
        >>> c_out = 3
        >>> h_query, w_query = 32, 32
        >>> model = ppsci.arch.CVit(
        ...     input_keys=["u", "y"],
        ...     output_keys=["s"],
        ...     in_dim=c_in,
        ...     coords_dim=2,
        ...     spatial_dims=[t, h, w],
        ...     patch_size=(1, 4, 4),
        ...     grid_size=(h, w),
        ...     latent_dim=32,
        ...     emb_dim=32,
        ...     depth=3,
        ...     num_heads=8,
        ...     dec_emb_dim=32,
        ...     dec_num_heads=8,
        ...     dec_depth=1,
        ...     num_mlp_layers=1,
        ...     mlp_ratio=1,
        ...     out_dim=c_out,
        ...     layer_norm_eps=1e-5,
        ...     embedding_type="grid",
        ... )
        >>> x = paddle.randn([b, t, h, w, c_in])
        >>> coords = paddle.randn([h_query * w_query, 2])
        >>> out = model({"u": x, "y": coords})["s"]
        >>> print(out.shape) # output shape should be [b, h_query * w_query, c_out]
        [2, 1024, 3]
    """

    def __init__(
        self,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        in_dim: int,
        coords_dim: int,
        spatial_dims: Sequence[int],
        patch_size: Sequence[int] = (1, 16, 16),
        grid_size: Sequence[int] = (128, 128),
        latent_dim: int = 256,
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        dec_emb_dim: int = 256,
        dec_num_heads: int = 8,
        dec_depth: int = 1,
        num_mlp_layers: int = 1,
        mlp_ratio: int = 1,
        out_dim: int = 1,
        layer_norm_eps: float = 1e-5,
        embedding_type: str = "grid",
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.spatial_dims = spatial_dims
        self.in_dim = in_dim
        self.coords_dim = coords_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dec_emb_dim = dec_emb_dim
        self.dec_num_heads = dec_num_heads
        self.dec_depth = dec_depth
        self.num_mlp_layers = num_mlp_layers
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.layer_norm_eps = layer_norm_eps
        self.embedding_type = embedding_type

        if self.embedding_type == "grid":
            # Create grid and latents
            n_x, n_y = self.grid_size[0], self.grid_size[1]

            x = paddle.linspace(0, 1, n_x)
            y = paddle.linspace(0, 1, n_y)
            xx, yy = paddle.meshgrid(x, y, indexing="ij")

            self.grid = paddle.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
            self.latents = self.create_parameter(
                [n_x * n_y, self.latent_dim],
                default_initializer=nn.initializer.Normal(std=1e-2),
            )
            self.fc = nn.Linear(self.latent_dim, self.dec_emb_dim)
            self.norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)
        elif self.embedding_type == "mlp":
            self.mlp = MlpBlock(self.latent_dim, self.dec_emb_dim, self.dec_emb_dim)
            self.norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)

        self.encoder = Encoder(
            self.in_dim,
            self.spatial_dims,
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.layer_norm_eps,
        )
        self.enc_norm = nn.LayerNorm(self.emb_dim, self.layer_norm_eps)
        self.fc1 = nn.Linear(self.emb_dim, self.dec_emb_dim)
        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    self.dec_num_heads,
                    self.dec_emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                    self.dec_emb_dim,
                    self.dec_emb_dim,
                )
                for _ in range(self.dec_depth)
            ]
        )
        self.block_norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)
        self.final_mlp = Mlp(
            self.num_mlp_layers,
            self.dec_emb_dim,
            self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )

    def forward_tensor(self, x, coords):
        b, t, h, w, c = x.shape

        # process query coordinates
        if self.embedding_type == "grid":
            d2 = ((coords.unsqueeze(1) - self.grid.unsqueeze(0)) ** 2).sum(axis=2)
            w = paddle.exp(-1e5 * d2) / paddle.exp(-1e5 * d2).sum(axis=1, keepdim=True)
            coords = paddle.einsum("ic,pi->pc", self.latents, w)
            coords = self.fc(coords)
            coords = self.norm(coords)
        elif self.embedding_type == "mlp":
            coords = self.mlp(coords)
            coords = self.norm(coords)

        coords = einops.repeat(coords, "n d -> b n d", b=b)

        # process input function(encoder)
        x = self.encoder(x)
        x = self.enc_norm(x)
        x = self.fc1(x)

        # decoder
        for i, block in enumerate(self.cross_attn_blocks):
            x = block(coords, x)

        # mlp
        x = self.block_norm(x)
        x = self.final_mlp(x)

        return x

    def forward(self, x_dict):
        if self._input_transform is not None:
            x = self._input_transform(x_dict)

        x, coords = x_dict[self.input_keys[0]], x_dict[self.input_keys[1]]
        if coords.ndim >= 3:
            coords = coords[0]  # [b, n, c] -> [n, c]

        y = self.forward_tensor(x, coords)

        y_dict = {self.output_keys[0]: y}
        if self._output_transform is not None:
            y_dict = self._output_transform(x_dict, y_dict)

        return y_dict
