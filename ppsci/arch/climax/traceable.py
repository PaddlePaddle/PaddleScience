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

"""Traceable drop-in replacements for ``paddle.compat.nn`` layers.

The ``paddle.compat.nn`` modules delegate their forward pass to
``paddle.compat.nn.functional``, whose bodies branch on runtime flags
(``get_flags`` / ``is_compiled_with_cuda`` / ``in_dynamic_or_pir_mode``).
Under SOT symbolic execution every such branch breaks the traced graph, so
the CINN compiler never receives the surrounding subgraphs and the wrapped
model falls back to dynamic execution. The subclasses below reimplement
forward with plain operators; parameter names, shapes and layouts are
unchanged, so existing checkpoints load without conversion.
"""

from __future__ import annotations

import math

import paddle


class TraceableLinear(paddle.compat.nn.Linear):
    """``paddle.compat.nn.Linear`` with a branch-free forward (matmul + bias)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # plain bool fixed at construction: an `is not None` test on the
        # parameter inside forward would break tracing again
        self._use_bias = self.bias is not None

    def forward(self, input):
        out = paddle.matmul(input, self.weight, transpose_y=True)
        if self._use_bias:
            out = out + self.bias
        return out


class TraceableMultiheadAttention(paddle.compat.nn.MultiheadAttention):
    """``paddle.compat.nn.MultiheadAttention`` with a branch-free forward.

    Only the pattern used by this model is supported: ``batch_first=True``,
    equal query/key/value embed dims, no masks, no dropout and no
    ``add_bias_kv``; attention weights are returned as ``None`` (the model
    discards them).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dropout != 0:
            raise ValueError("TraceableMultiheadAttention only supports dropout=0")
        if not self.batch_first:
            raise ValueError(
                "TraceableMultiheadAttention only supports batch_first=True"
            )
        if not self._qkv_same_embed_dim:
            raise ValueError(
                "TraceableMultiheadAttention only supports kdim=vdim=embed_dim"
            )
        if getattr(self, "bias_k", None) is not None:
            raise ValueError("TraceableMultiheadAttention does not support add_bias_kv")
        self._use_bias = self.in_proj_bias is not None

    def forward(self, query, key, value):
        e = self.embed_dim
        h = self.num_heads
        d = self.head_dim
        # in-projection: rows of the packed [3E, E] weight are [q; k; v]
        w = self.in_proj_weight
        q = paddle.matmul(query, w[:e].T)
        k = paddle.matmul(key, w[e : 2 * e].T)
        v = paddle.matmul(value, w[2 * e :].T)
        if self._use_bias:
            b = self.in_proj_bias
            q = q + b[:e]
            k = k + b[e : 2 * e]
            v = v + b[2 * e :]
        bsz, qlen = q.shape[0], q.shape[1]
        klen = k.shape[1]
        q = q.reshape([bsz, qlen, h, d]).transpose([0, 2, 1, 3])
        k = k.reshape([bsz, klen, h, d]).transpose([0, 2, 1, 3])
        v = v.reshape([bsz, klen, h, d]).transpose([0, 2, 1, 3])
        attn = paddle.nn.functional.softmax(
            paddle.matmul(q, k.transpose([0, 1, 3, 2])) / math.sqrt(d), axis=-1
        )
        out = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([bsz, qlen, e])
        # out_proj keeps the compat Linear [out_features, in_features] layout
        out = paddle.matmul(out, self.out_proj.weight, transpose_y=True)
        if self._use_bias:
            out = out + self.out_proj.bias
        return out, None
