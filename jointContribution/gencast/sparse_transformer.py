# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer with either dense or sparse attention.

The sparse attention implemented here is for nodes to attend only to themselves
and their neighbours on the graph). It assumes that the adjacency matrix has a
banded structure, and is implemented with dense operations computing with only
the diagonal, super diagonal, and subdiagonal blocks of the tri-block-diagonal
attention matrix.

The basic model structure of the transformer and some functions were adapted
from xlm's transformer_simple.py.
"""

import math
import warnings
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from graphcast import graphtype
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)


class FeedForwardBlock(nn.Layer):
    """Feed-forward block."""

    def __init__(self, cfg):
        super(FeedForwardBlock, self).__init__()
        self.cfg = cfg
        stddev = math.sqrt(cfg.ffw_winit_mult / (cfg.num_layers * cfg.d_model))
        stddev_final = math.sqrt(
            cfg.ffw_winit_final_mult / (cfg.num_layers * cfg.ffw_hidden)
        )
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=stddev)
        )
        weight_attr_final = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=stddev_final
            )
        )
        self.ffw_up = nn.Linear(cfg.d_model, cfg.ffw_hidden, weight_attr=weight_attr)
        self.ffw_down = nn.Linear(
            cfg.ffw_hidden, cfg.d_model, weight_attr=weight_attr_final
        )

    def forward(self, x):
        x = self.ffw_up(x)
        x = getattr(F, self.cfg.activation)(x)
        return self.ffw_down(x)


class MultiheadLinear(nn.Layer):
    def __init__(self, qkv, cfg):
        super(MultiheadLinear, self).__init__()
        self.cfg = cfg
        self.qkv = qkv
        head_size = self.cfg.value_size if qkv == "v" else self.cfg.key_size

        stddev = math.sqrt(
            cfg.attn_winit_mult / (cfg.num_layers * cfg.num_heads * head_size)
        )
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=stddev)
        )

        self.linear = nn.Linear(
            cfg.num_heads * head_size,
            cfg.num_heads * head_size,
            weight_attr=weight_attr,
            bias_attr=False,  # with_bias=False
        )

    def forward(self, x):
        out = self.linear(x)

        shape = list(out.shape[:-1]) + [
            self.cfg.num_heads,
            self.cfg.value_size if self.qkv == "v" else self.cfg.key_size,
        ]
        return paddle.reshape(out, shape)


def get_mask_block_size(mask: sparse.csr_matrix) -> int:
    """Get blocksize of the adjacency matrix (attn mask) for the permuted mesh."""
    # sub-diagonal bandwidth
    lbandwidth = (np.arange(mask.shape[0]) - (mask != 0).argmax(axis=0) + 1).max()
    # super-diagonal bandwidth
    ubandwidth = (
        (mask.shape[0] - 1)
        - np.argmax(mask[::-1, :] != 0, axis=0)
        - np.arange(mask.shape[0])
        + 1
    ).max()
    block_size = np.maximum(lbandwidth, ubandwidth)
    return block_size


def triblockdiag_softmax(
    logits: Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Softmax given the diag, upper diag, and lower diag logit blocks."""

    logits_d, logits_u, logits_l = logits

    m = paddle.max(
        paddle.stack(
            [
                paddle.max(logits_d, axis=-1, keepdim=True),
                paddle.max(logits_u, axis=-1, keepdim=True),
                paddle.max(logits_l, axis=-1, keepdim=True),
            ]
        ),
        axis=0,
    )

    unnormalized_d = paddle.exp(logits_d - m)
    unnormalized_u = paddle.exp(logits_u - m)
    unnormalized_l = paddle.exp(logits_l - m)

    denom = (
        paddle.sum(unnormalized_d, axis=-1, keepdim=True)
        + paddle.sum(unnormalized_u, axis=-1, keepdim=True)
        + paddle.sum(unnormalized_l, axis=-1, keepdim=True)
    )

    logits_d = unnormalized_d / denom
    logits_u = unnormalized_u / denom
    logits_l = unnormalized_l / denom

    return (logits_d, logits_u, logits_l)


def layernorm(
    x: paddle.Tensor, create_scale: bool, create_offset: bool
) -> paddle.Tensor:
    """Layer normalization using PaddlePaddle."""
    layer_norm = paddle.nn.LayerNorm(
        normalized_shape=x.shape[-1],  # Normalize across the last dimension
        weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(1.0) if create_scale else None
        ),
        bias_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0) if create_offset else None
        ),
        epsilon=1e-5,
    )
    return layer_norm(x)


def mask_block_diags(
    mask: sparse.csr_matrix, num_padding_nodes: int, block_size: int
) -> paddle.Tensor:
    """Pad and reshape mask diag, super-diag and sub-diag blocks."""
    # Add zero padding to the mask
    mask_padding_rows = sparse.csr_matrix(
        (num_padding_nodes, mask.shape[1]), dtype=np.int32
    )
    mask = sparse.vstack([mask, mask_padding_rows])
    mask_padding_cols = sparse.csr_matrix(
        (mask.shape[0], num_padding_nodes), dtype=np.int32
    )
    mask = sparse.hstack([mask, mask_padding_cols])

    assert (mask.shape[-1] % block_size) == 0

    # Convert the sparse mask to a dense format for manipulation
    mask_dense = mask.toarray()

    mask_diag_blocks = paddle.stack(
        [
            paddle.to_tensor(
                mask_dense[
                    i * block_size : (i + 1) * block_size,
                    i * block_size : (i + 1) * block_size,
                ]
            )
            for i in range(mask_dense.shape[0] // block_size)
        ]
    )

    mask_upper_diag_blocks = paddle.stack(
        [
            paddle.to_tensor(
                mask_dense[
                    i * block_size : (i + 1) * block_size,
                    (i + 1) * block_size : (i + 2) * block_size,
                ]
            )
            for i in range(mask_dense.shape[0] // block_size - 1)
        ]
        + [paddle.zeros((block_size, block_size), dtype=mask_diag_blocks.dtype)]
    )

    mask_lower_diag_blocks = paddle.stack(
        [paddle.zeros((block_size, block_size), dtype=mask_diag_blocks.dtype)]
        + [
            paddle.to_tensor(
                mask_dense[
                    (i + 1) * block_size : (i + 2) * block_size,
                    i * block_size : (i + 1) * block_size,
                ]
            )
            for i in range(mask_dense.shape[0] // block_size - 1)
        ]
    )

    mask = paddle.stack(
        [mask_diag_blocks, mask_upper_diag_blocks, mask_lower_diag_blocks]
    )
    mask = paddle.unsqueeze(mask, axis=(0, 3))
    return mask


def _get_adj_matrix_for_edge_set(
    graph: graphtype.GraphGridMesh,
    add_self_edges: bool,
):
    """Returns the adjacency matrix for the given graph and edge set."""

    sender_n_node = graph.mesh_node_feat.shape[0]
    receiver_n_node = graph.mesh_node_feat.shape[0]

    # Build adjacency matrix.
    adj_mat = sparse.csr_matrix((sender_n_node, receiver_n_node), dtype=np.bool_)

    s = graph.mesh2mesh_src_index.numpy().copy()
    r = graph.mesh2mesh_dst_index.numpy().copy()

    adj_mat[s, r] = True

    if add_self_edges:
        # Should only do this if we are certain the adjacency matrix is square.
        adj_mat[np.arange(sender_n_node), np.arange(receiver_n_node)] = True
    return adj_mat


class LinearNormConditioning(nn.Layer):
    """Module for norm conditioning.

    Conditions the normalization of "inputs" by applying a linear layer to the
    "norm_conditioning" which produces the scale and variance which are applied to
    each channel (across the last dim) of "inputs".
    """

    def __init__(self, in_features, out_features):
        super(LinearNormConditioning, self).__init__()
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=1e-8)
        )
        self.linear = nn.Linear(
            in_features,
            out_features * 2,
            weight_attr=weight_attr,
        )

    def forward(self, inputs: paddle.Tensor, norm_conditioning: paddle.Tensor):

        conditional_scale_offset = self.linear(norm_conditioning)
        scale_minus_one, offset = paddle.split(conditional_scale_offset, 2, axis=-1)
        scale = scale_minus_one + 1.0
        return inputs * scale + offset


class Block(nn.Layer):
    """Transformer block (mha and ffw)."""

    def __init__(self, config):
        super(Block, self).__init__()
        self._cfg = config

        self.norm_conditioning = LinearNormConditioning(
            config.norm_conditioning_feat, config.mesh_node_emb_dim
        )
        self.norm_conditioning_1 = LinearNormConditioning(
            config.norm_conditioning_feat, config.mesh_node_emb_dim
        )
        self.ffw = FeedForwardBlock(config)

        stddev = math.sqrt(
            config.attn_winit_final_mult
            / (config.num_layers * config.num_heads * config.value_size)
        )
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=stddev)
        )
        self.mha_final_layer = nn.Linear(
            config.num_heads * config.value_size,
            config.d_model,
            weight_attr=weight_attr,
        )
        self.mha_proj_q = MultiheadLinear("q", config)
        self.mha_proj_k = MultiheadLinear("k", config)
        self.mha_proj_v = MultiheadLinear("v", config)

    def forward(self, graph):
        # x shape is (batch, num_nodes, feature_dim)

        mask = graph.adj_mat**self._cfg.attention_k_hop
        self.mask_block_size = get_mask_block_size(mask)

        self.num_padding_nodes = int(
            np.ceil(mask.shape[0] / self.mask_block_size) * self.mask_block_size
            - mask.shape[0]
        )
        self.mask = mask_block_diags(mask, self.num_padding_nodes, self.mask_block_size)

        x = graph.mesh_node_feat
        x = paddle.transpose(x, perm=[1, 0, 2])
        num_nodes = x.shape[1]

        def attn(x):
            if self._cfg.attention_type == "triblockdiag_mha":
                # We pad -> reshape -> compute attn -> reshape -> select at each block
                # so as to avoid complications involved in making the norm layers and
                # ffw blocks account for the padding. However, this might be decreasing
                # efficiency.

                # Add padding so that number of nodes is divisible into blocks
                x = F.pad(x, [0, 0, 0, self.num_padding_nodes, 0, 0])
                x = x.reshape(
                    [
                        x.shape[0],
                        x.shape[1] // self.mask_block_size,
                        self.mask_block_size,
                        x.shape[-1],
                    ]
                )
                x = self.triblockdiag_mha(x, x, mask=self.mask, cfg=self._cfg)
                x = x.reshape(
                    [x.shape[0], num_nodes + self.num_padding_nodes, x.shape[-1]]
                )
                return x[:, :num_nodes, :]
            else:
                raise NotImplementedError()

        norm_conditioning_layer = self.norm_conditioning(
            layernorm(x, create_scale=False, create_offset=False),
            norm_conditioning=paddle.unsqueeze(graph.global_norm_conditioning, axis=1),
        )

        x = x + attn(norm_conditioning_layer)

        norm_conditioning_layer = self.norm_conditioning_1(
            layernorm(x, create_scale=False, create_offset=False),
            norm_conditioning=paddle.unsqueeze(graph.global_norm_conditioning, axis=1),
        )
        x = x + self.ffw(norm_conditioning_layer)
        x = paddle.transpose(x, perm=[1, 0, 2])
        graph.mesh_node_feat = x
        return graph

    def triblockdiag_mha(
        self, q_input: paddle.Tensor, kv_input: paddle.Tensor, mask: paddle.Tensor, cfg
    ) -> paddle.Tensor:
        """Triblockdiag multihead attention."""

        # q_inputs, kv_input: (batch, num_blocks, block_size, num_heads, d_model)
        q = self.mha_proj_q(q_input)
        k = self.mha_proj_k(kv_input)
        v = self.mha_proj_v(kv_input)

        k = F.pad(k, [0, 0, 0, 0, 1, 1], data_format="NDHWC")
        v = F.pad(v, [0, 0, 0, 0, 1, 1], data_format="NDHWC")

        def qk_prod(queries, keys):
            return paddle.einsum("bnqhd,bnkhd->bnhqk", queries, keys)

        # q shape is (batch, num_blocks, block_size, num_heads, qk_dim)
        # k shape is (batch, num_blocks + 2, block_size, num_heads, qk_dim)
        logits_d = qk_prod(q, k[:, 1:-1, ...]) * (cfg.key_size**-0.5)
        logits_u = qk_prod(q, k[:, 2:, ...]) * (cfg.key_size**-0.5)
        logits_l = qk_prod(q, k[:, :-2, ...]) * (cfg.key_size**-0.5)

        # apply mask
        logits_d = paddle.where(
            paddle.cast(mask[:, 0, ...], dtype="bool"),
            logits_d,
            paddle.to_tensor(-1e30),
        )
        logits_u = paddle.where(
            paddle.cast(mask[:, 1, ...], dtype="bool"),
            logits_u,
            paddle.to_tensor(-1e30),
        )
        logits_l = paddle.where(
            paddle.cast(mask[:, 2, ...], dtype="bool"),
            logits_l,
            paddle.to_tensor(-1e30),
        )

        logits_d, logits_u, logits_l = triblockdiag_softmax(
            (logits_d, logits_u, logits_l)
        )

        def av_prod(attn_weights, values):
            return paddle.einsum("bnhqk,bnkhd->bnqhd", attn_weights, values)

        out_d = av_prod(logits_d, v[:, 1:-1, ...])
        out_u = av_prod(logits_u, v[:, 2:, ...])
        out_l = av_prod(logits_l, v[:, :-2, ...])
        # x shape is (batch, num_blocks, block_size, num_heads, d_model)
        x = out_d + out_u + out_l

        x = paddle.reshape(x, x.shape[:-2] + [cfg.num_heads * cfg.value_size])
        x = self.mha_final_layer(x)
        return x


class Transformer(nn.Layer):
    """Main transformer module that processes embeddings.

    All but the very first and very last layer of a 'classic' Transformer:
    Receives already embedded inputs instead of discrete tokens.
    Outputs an embedding for each 'node'/'position' rather than logits.
    """

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.block = nn.Sequential()
        for idx in range(config.num_layers):
            self.block.add_sublayer(
                f"{idx}",
                Block(config),
            )

        self.final_norm_conditioning = LinearNormConditioning(
            config.norm_conditioning_feat, config.mesh_node_emb_dim
        )

    def forward(self, x: graphtype.GraphGridMesh):
        # node_features expected to have shape (batch, num_nodes, d)
        adj_mat = _get_adj_matrix_for_edge_set(
            graph=x,
            add_self_edges=True,
        )
        x.adj_mat = adj_mat

        x = self.block(x)

        node_features = x.mesh_node_feat
        node_features = paddle.transpose(node_features, perm=[1, 0, 2])
        node_features = self.final_norm_conditioning(
            layernorm(node_features, create_scale=False, create_offset=False),
            norm_conditioning=paddle.unsqueeze(x.global_norm_conditioning, axis=1),
        )
        node_features = paddle.transpose(node_features, perm=[1, 0, 2])
        x.mesh_node_feat = node_features

        return x
