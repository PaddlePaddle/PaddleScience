# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paddle attention primitives for maxxxzdn/mosaic (Apache-2.0).

The selected-block path gathers only selected KV blocks; it does not construct
a token-level global attention mask. Top-k routing intentionally has no gradient,
matching the reference. Arithmetic here is ordinary FP32/BF16 Paddle arithmetic;
the original Triton kernel additionally quantizes intermediates to BF16.
"""

import paddle
from einops import rearrange
from einops import reduce
from paddle import nn
from paddle.nn import functional as F


class RMSNorm(nn.Layer):
    """Match torch.nn.RMSNorm's dtype-dependent default epsilon."""

    def __init__(self, dim, elementwise_affine=True):
        super().__init__()
        self.weight = (
            self.create_parameter([dim], default_initializer=nn.initializer.Constant(1))
            if elementwise_affine
            else None
        )

    def forward(self, x):
        eps = paddle.finfo(x.dtype).eps
        out = x * paddle.rsqrt(paddle.mean(x.square(), axis=-1, keepdim=True) + eps)
        return out if self.weight is None else out * self.weight


def attention(q, k, v):
    """Scaled dot-product attention with contiguous GQA head expansion."""
    groups = q.shape[-2] // k.shape[-2]
    if groups > 1:
        k = paddle.repeat_interleave(k, groups, axis=-2)
        v = paddle.repeat_interleave(v, groups, axis=-2)
    q, k, v = [x.transpose([0, 2, 1, 3]) for x in (q, k, v)]
    scores = paddle.matmul(q, k, transpose_y=True) * (q.shape[-1] ** -0.5)
    probs = F.softmax(scores.astype("float32"), axis=-1).astype(v.dtype)
    return paddle.matmul(probs, v).transpose([0, 2, 1, 3])


def block_attention(q, k, v, block_size, recompute_chunks=False):
    batch = q.shape[0]
    q, k, v = [
        rearrange(x, "b (n s) h d -> (b n) s h d", s=block_size) for x in (q, k, v)
    ]
    if recompute_chunks:
        from paddle.distributed.fleet.utils import recompute

        outputs = [
            recompute(
                attention,
                q[start : start + 2],
                k[start : start + 2],
                v[start : start + 2],
                use_reentrant=False,
            )
            for start in range(0, q.shape[0], 2)
        ]
        out = paddle.concat(outputs, axis=0)
    else:
        out = attention(q, k, v)
    return rearrange(out, "(b n) s h d -> b (n s) h d", b=batch)


@paddle.no_grad()
def attn_topk(q, k, count):
    groups = q.shape[2] // k.shape[2]
    if groups > 1:
        k = paddle.repeat_interleave(k, groups, axis=2)
    scores = paddle.matmul(
        q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), transpose_y=True
    )
    if groups > 1:
        # This seemingly unusual grouping is present in the official reference.
        scores = reduce(scores, "b (g h) t k -> b h t k", "mean", g=groups)
    return paddle.topk(scores.transpose([0, 2, 1, 3]), count, axis=-1)[1]


def _selected_chunk(queries, key_bank, value_bank, indices, count, block_size):
    batch, heads, chunks = indices.shape[:3]
    dim = key_bank.shape[-1]
    keys, values = [
        paddle.gather(bank, indices.flatten(), axis=0).reshape(
            [batch, heads, chunks, count * block_size, dim]
        )
        for bank in (key_bank, value_bank)
    ]
    scores = paddle.matmul(queries, keys.unsqueeze(3), transpose_y=True) * (dim**-0.5)
    weights = F.softmax(scores.astype("float32"), axis=-1).astype(values.dtype)
    return paddle.matmul(weights, values.unsqueeze(3))


def sparse_attention(
    q, k, v, indices, block_size, query_chunk_size=8, recompute_chunks=False
):
    """Gather selected keys per query block and KV head, then attend jointly.

    q: [B,N,Hq,D], k/v: [B,N,Hkv,D], indices: [B,N/S,Hkv,K].
    Memory scales with selected blocks rather than the global N squared grid.
    """
    batch, length, kv_heads, dim = k.shape
    blocks = length // block_size
    count = indices.shape[-1]
    # Flatten [batch, head, block] to gather without expanding a global mask.
    offsets = (
        paddle.arange(batch * kv_heads, dtype="int64").reshape([batch, kv_heads, 1, 1])
        * blocks
    )
    flat_indices = indices.transpose([0, 2, 1, 3]) + offsets
    banks = [
        rearrange(tensor, "b (n s) h d -> (b h n) s d", s=block_size)
        for tensor in (k, v)
    ]
    queries = rearrange(q, "b (n s) (h g) d -> b h n g s d", s=block_size, h=kv_heads)
    outputs = []
    for start in range(0, blocks, query_chunk_size):
        stop = min(start + query_chunk_size, blocks)
        args = (
            queries[:, :, start:stop],
            *banks,
            flat_indices[:, :, start:stop],
            count,
            block_size,
        )
        if recompute_chunks:
            from paddle.distributed.fleet.utils import recompute

            outputs.append(recompute(_selected_chunk, *args, use_reentrant=False))
        else:
            outputs.append(_selected_chunk(*args))
    out = paddle.concat(outputs, axis=2)
    return rearrange(out, "b h n g s d -> b (n s) (h g) d")


def mosaic_attention(
    q,
    k,
    v,
    weights,
    local_size,
    sparse_size,
    count,
    local_only,
    no_compression=False,
    recompute_chunks=False,
):
    local = block_attention(q, k, v, local_size, recompute_chunks)
    if local_only:
        return local
    qc, kc = [
        reduce(x, "b (n s) h d -> b n h d", "mean", s=sparse_size) for x in (q, k)
    ]
    if no_compression:
        selected = sparse_attention(
            q,
            k,
            v,
            attn_topk(qc, kc, count),
            sparse_size,
            recompute_chunks=recompute_chunks,
        )
        total = weights[0] + weights[2] + 1e-8
        return (local * weights[0] + selected * weights[2]) / total
    vc = reduce(v, "b (n s) h d -> b n h d", "mean", s=sparse_size)
    compressed = paddle.repeat_interleave(attention(qc, kc, vc), sparse_size, axis=1)
    if count == 0:
        total = weights[0] + weights[1] + 1e-8
        return (local * weights[0] + compressed * weights[1]) / total
    selected = sparse_attention(
        q,
        k,
        v,
        attn_topk(qc, kc, count),
        sparse_size,
        recompute_chunks=recompute_chunks,
    )
    return local * weights[0] + compressed * weights[1] + selected * weights[2]


class RoPE(nn.Layer):
    def __init__(self, dim, theta=10000):
        super().__init__()
        if dim % 4:
            raise ValueError(
                "Mosaic's longitude/latitude RoPE requires head_dim % 4 == 0"
            )
        self.dim, self.theta = dim, theta

    def initialize_rope(self, positions):
        freq = 1.0 / (
            self.theta
            ** (paddle.arange(0, self.dim // 2, 2, dtype="float32") / (self.dim // 2))
        )
        angles = paddle.deg2rad(positions)
        phases = paddle.concat(
            [angles[:, :1] * freq[None], angles[:, 1:2] * freq[None]], axis=-1
        )
        self.register_buffer("cos_freqs", phases.cos(), persistable=True)
        self.register_buffer("sin_freqs", phases.sin(), persistable=True)

    def forward(self, x):
        pairs = x.astype("float32").reshape([*x.shape[:-1], -1, 2])
        rotated = paddle.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).flatten(-2)
        cos = paddle.repeat_interleave(self.cos_freqs, 2, axis=-1)[None, :, None]
        sin = paddle.repeat_interleave(self.sin_freqs, 2, axis=-1)[None, :, None]
        return (x.astype("float32") * cos + rotated * sin).astype(x.dtype)


class MosaicAttention(nn.Layer):
    def __init__(self, cfg, stage, local_only):
        super().__init__()
        self.local_only = local_only
        self.enable_recompute = False
        self.no_compression = cfg.no_compression
        self.local_size = stage.block_attn_size
        self.sparse_size = stage.sparse_block_size
        self.count = stage.sparse_block_count
        self.q_heads = stage.num_heads
        self.kv_heads = stage.num_heads // stage.gqa_ratio
        head_dim = stage.dim // self.q_heads // cfg.qkv_compress_ratio
        self.to_q = nn.Linear(stage.dim, self.q_heads * head_dim, bias_attr=False)
        self.to_k = nn.Linear(stage.dim, self.kv_heads * head_dim, bias_attr=False)
        self.to_v = nn.Linear(stage.dim, self.kv_heads * head_dim, bias_attr=False)
        self.to_o = nn.Linear(self.q_heads * head_dim, stage.dim, bias_attr=False)
        self.q_rope = RoPE(head_dim, cfg.rope_theta) if cfg.rope else None
        self.k_rope = RoPE(head_dim, cfg.rope_theta) if cfg.rope else None
        self.to_strategy_combine_mlp = (
            None
            if local_only
            else nn.Linear(stage.dim, 3 * self.q_heads, bias_attr=False)
        )

    def forward(self, x):
        weights = None
        if not self.local_only:
            logits = rearrange(
                self.to_strategy_combine_mlp(x), "t b (h s) -> s b t h", h=self.q_heads
            )
            weights = (
                F.softmax(logits.astype("float32"), axis=0)
                .astype(x.dtype)
                .unsqueeze(-1)
            )
        q = rearrange(self.to_q(x), "s b (h d) -> b s h d", h=self.q_heads)
        k = rearrange(self.to_k(x), "s b (h d) -> b s h d", h=self.kv_heads)
        v = rearrange(self.to_v(x), "s b (h d) -> b s h d", h=self.kv_heads)
        if self.q_rope is not None:
            q, k = self.q_rope(q), self.k_rope(k)
        out = mosaic_attention(
            q,
            k,
            v,
            weights,
            self.local_size,
            self.sparse_size,
            self.count,
            self.local_only,
            self.no_compression,
            self.enable_recompute and self.training and paddle.is_grad_enabled(),
        )
        return self.to_o(rearrange(out, "b s h d -> s b (h d)"))


class ConditionalSwiGLU(nn.Layer):
    def __init__(self, dim, hidden_dim, noise_dim):
        super().__init__()
        self.w13 = nn.Linear(dim, hidden_dim * 2, bias_attr=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias_attr=False)
        if noise_dim:
            self.noise_bias = nn.Linear(noise_dim, hidden_dim, bias_attr=False)

    def forward(self, x, z=None):
        x1, x3 = self.w13(x).chunk(2, axis=-1)
        noise = 0 if z is None else self.noise_bias(z).unsqueeze(0)
        return self.w2(F.silu(x1 + noise) * x3)


class MosaicBlock(nn.Layer):
    def __init__(self, cfg, stage, local_only):
        super().__init__()
        self.attention = MosaicAttention(cfg, stage, local_only)
        self.norm1 = RMSNorm(stage.dim, cfg.rmsnorm_elementwise_affine)
        self.norm2 = RMSNorm(stage.dim, cfg.rmsnorm_elementwise_affine)
        self.ffn = ConditionalSwiGLU(
            stage.dim, int(stage.dim * stage.mlp_ratio), cfg.noise_dim
        )

    def forward(self, x, z=None):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x), z)
