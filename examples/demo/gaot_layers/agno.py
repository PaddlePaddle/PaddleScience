# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
AGNO模块 - 注意力图神经算子实现（核心消息传递机制）
输入: 坐标、特征、邻居信息 | 输出: 聚合后特征 | 地位: 核心组件，被MAGNO使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .mlp import LinearChannelMLP
from .utils.scatter import segment_csr


class AGNO(nn.Layer):
    """
    Attentional Graph Neural Operator.

    Computes attentionally-weighted integral transforms:
        ∫_{A(x)} α(x,y) * k(x, y, [f(y)]) * [f(y)] dy

    Where:
    - α(x,y) is the attention weight between query x and neighbor y
    - A(x) is the neighborhood of x
    - k is a learnable kernel (MLP)
    - f is the input function

    Parameters
    ----------
    channel_mlp_layers : list
        Layer sizes for kernel MLP [input_dim, hidden, ..., output_dim]
    transform_type : str
        Type of transform:
        - 'linear': k(x, y)
        - 'nonlinear': k(x, y, f(y))
        - 'linear_kernelonly': k(x, y) without multiplying f(y)
        - 'nonlinear_kernelonly': k(x, y, f(y)) without multiplying f(y)
    use_attn : bool, default False
        Whether to use attention mechanism
    attention_type : str, default 'cosine'
        Attention type: 'cosine' or 'dot_product'
    coord_dim : int, optional
        Coordinate dimension (required if use_attn=True)
    use_torch_scatter : bool, default True
        (Placeholder for compatibility, always uses paddle implementation)
    """

    def __init__(
        self,
        channel_mlp_layers=None,
        channel_mlp=None,
        transform_type="linear",
        use_attn=False,
        attention_type="cosine",
        coord_dim=None,
        use_torch_scatter=True,
    ):
        super().__init__()

        # Store configuration
        self.transform_type = transform_type
        self.use_attn = use_attn
        self.attention_type = attention_type

        # Validate parameters
        if channel_mlp is None and channel_mlp_layers is None:
            raise ValueError(
                "Either channel_mlp or channel_mlp_layers must be provided."
            )
        if self.transform_type not in [
            "linear_kernelonly",
            "linear",
            "nonlinear_kernelonly",
            "nonlinear",
        ]:
            raise ValueError(f"Invalid transform_type: {transform_type}")
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")
            self.coord_dim = coord_dim
            if self.attention_type not in ["cosine", "dot_product"]:
                raise ValueError(f"Invalid attention_type: {self.attention_type}")

        # Initialize kernel MLP
        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers, non_linearity=F.gelu
            )
            # Store output dimension
            self.out_channels = channel_mlp_layers[-1]
        else:
            self.channel_mlp = channel_mlp
            # Try to get output dimension from MLP
            if hasattr(channel_mlp, "layers") and len(channel_mlp.layers) > 0:
                last_layer = channel_mlp.layers[-1]
                if hasattr(last_layer, "weight"):
                    self.out_channels = last_layer.weight.shape[0]
                else:
                    self.out_channels = None
            else:
                self.out_channels = None

        # Initialize attention projection if needed
        if self.use_attn and self.attention_type == "dot_product":
            attention_dim = 64
            self.query_proj = nn.Linear(self.coord_dim, attention_dim)
            self.key_proj = nn.Linear(self.coord_dim, attention_dim)
            self.scaling_factor = 1.0 / (attention_dim**0.5)

    def _segment_softmax(self, attention_scores, indptr):
        """
        Apply segment-wise softmax for attention weight normalization.

        Parameters
        ----------
        attention_scores : paddle.Tensor [num_neighbors]
            Raw attention scores
        indptr : paddle.Tensor [n_queries + 1]
            CSR index pointers

        Returns
        -------
        paddle.Tensor [num_neighbors]
            Normalized attention weights
        """
        # Compute max per segment for numerical stability
        max_values = segment_csr(
            attention_scores.unsqueeze(-1), indptr, reduce="max"
        ).squeeze(-1)

        # Expand max values
        max_values_expanded = paddle.repeat_interleave(
            max_values, indptr[1:] - indptr[:-1], axis=0
        )

        # Stable exp
        attention_scores = attention_scores - max_values_expanded
        exp_scores = paddle.exp(attention_scores)

        # Sum exp scores per segment
        sum_exp = segment_csr(exp_scores.unsqueeze(-1), indptr, reduce="sum").squeeze(
            -1
        )

        # Expand sum
        sum_exp_expanded = paddle.repeat_interleave(
            sum_exp, indptr[1:] - indptr[:-1], axis=0
        )

        # Normalize
        attention_weights = exp_scores / (sum_exp_expanded + 1e-8)

        return attention_weights

    def forward(
        self,
        y: paddle.Tensor,
        neighbors: Dict[str, paddle.Tensor],
        x: Optional[paddle.Tensor] = None,
        f_y: Optional[paddle.Tensor] = None,
        weights: Optional[paddle.Tensor] = None,
        # Legacy parameters for backward compatibility
        query_coord: Optional[paddle.Tensor] = None,
        key_coord: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Forward pass of AGNO - 接口与PyTorch完全一致

        Parameters
        ----------
        y : paddle.Tensor [n, coord_dim]
            物理点坐标（与PyTorch一致）
        neighbors : Dict
            邻居信息字典，包含:
            - 'edge_index': [2, E] 边 [query_idx, key_idx]
            - 'indptr': [N+1] CSR索引指针 (可选)
        x : paddle.Tensor [m, coord_dim], optional
            查询点坐标，如果为None则x=y（自查询）
        f_y : paddle.Tensor [batch, n, in_channels], optional
            输入特征，如果为None则只做坐标编码
        weights : paddle.Tensor, optional
            可选的边权重
        query_coord : paddle.Tensor, optional
            遗留参数，为了向后兼容
        key_coord : paddle.Tensor, optional
            遗留参数，为了向后兼容

        Returns
        -------
        paddle.Tensor [batch, m, out_channels] or [m, out_channels]
            输出特征
        """
        # Handle legacy parameters
        if query_coord is not None:
            x = query_coord
        if key_coord is not None:
            y = key_coord

        # If x is None, self-query (x=y)
        if x is None:
            x = y
        edge_index = neighbors["edge_index"]  # [2, E]
        query_indices = edge_index[0]  # [E]
        key_indices = edge_index[1]  # [E]

        num_queries = x.shape[0]  # m
        num_keys = y.shape[0]  # n
        edge_index.shape[1]

        # If f_y is None, create dummy features
        if f_y is None:
            f_y = paddle.ones(
                [1, num_keys, self.in_channels if hasattr(self, "in_channels") else 1],
                dtype=y.dtype,
            )

        # Ensure f_y is 3D: [batch, n, in_channels]
        if f_y.ndim == 2:
            f_y = f_y.unsqueeze(0)

        batch_size = f_y.shape[0]

        # Get indptr for CSR format
        if "indptr" in neighbors:
            indptr = neighbors["indptr"]
        else:
            indptr = self._compute_indptr(query_indices, num_queries)

        # Build edge features: [query_coord, key_coord, f_y_edge]
        query_coords_edge = x[query_indices]  # [E, coord_dim]
        key_coords_edge = y[key_indices]  # [E, coord_dim]

        # Build edge coordinate features
        if self.transform_type in ["linear", "linear_kernelonly"]:
            edge_coord_features = paddle.concat(
                [query_coords_edge, key_coords_edge], axis=-1
            )
        else:
            edge_coord_features = paddle.concat(
                [query_coords_edge, key_coords_edge], axis=-1
            )

        # Process each batch
        outputs = []
        for b in range(batch_size):
            f_y_b = f_y[b]  # [n, in_channels]
            f_y_edge = f_y_b[key_indices]  # [E, in_channels]

            # Build kernel input
            if self.transform_type in ["nonlinear", "nonlinear_kernelonly"]:
                kernel_input = paddle.concat([edge_coord_features, f_y_edge], axis=-1)
            else:
                kernel_input = edge_coord_features

            # Apply kernel MLP
            kernel_output = self.channel_mlp(kernel_input)  # [E, out_channels]

            # Decide whether to multiply by f_y based on transform_type
            if self.transform_type.endswith("_kernelonly"):
                rep_features = kernel_output
            else:
                rep_features = kernel_output * f_y_edge
            # Apply attention weights if enabled
            if self.use_attn:
                if self.attention_type == "cosine":
                    # Cosine similarity attention
                    # Normalize coordinates
                    query_norm = F.normalize(query_coords_edge, axis=-1)
                    key_norm = F.normalize(key_coords_edge, axis=-1)

                    # Cosine similarity
                    attention_scores = (query_norm * key_norm).sum(axis=-1)  # [E]

                elif self.attention_type == "dot_product":
                    # Scaled dot-product attention
                    q = self.query_proj(query_coords_edge)  # [E, attention_dim]
                    k = self.key_proj(key_coords_edge)  # [E, attention_dim]

                    attention_scores = (q * k).sum(axis=-1) * self.scaling_factor  # [E]

                # Segment-wise softmax
                attention_weights = self._segment_softmax(attention_scores, indptr)

                # Weight representations
                rep_features = rep_features * attention_weights.unsqueeze(-1)

            # Aggregate using segment_csr
            out_features = segment_csr(
                rep_features, indptr, reduce="sum"
            )  # [N', out_channels]

            # Ensure output has correct number of queries
            if out_features.shape[0] < num_queries:
                full_output = paddle.zeros(
                    [num_queries, self.out_channels], dtype=out_features.dtype
                )
                full_output[: out_features.shape[0]] = out_features
                out_features = full_output

            outputs.append(out_features)

        # Return stacked outputs or squeeze if batch_size=1
        result = paddle.stack(outputs, axis=0)  # [batch, m, out_channels]
        if batch_size == 1:
            result = result.squeeze(0)  # [m, out_channels]

        return result

    def _compute_indptr(
        self, indices: paddle.Tensor, num_queries: int
    ) -> paddle.Tensor:
        """
        Compute CSR indptr from edge indices.

        Parameters
        ----------
        indices : paddle.Tensor [E]
            Query indices
        num_queries : int
            Number of query points

        Returns
        -------
        paddle.Tensor [num_queries + 1]
            CSR index pointers
        """
        # Count occurrences of each index
        counts = paddle.zeros([num_queries], dtype="int64")
        for i in range(num_queries):
            counts[i] = (indices == i).sum()

        # Cumulative sum to get indptr
        indptr = paddle.concat(
            [paddle.zeros([1], dtype="int64"), paddle.cumsum(counts, axis=0)]
        )

        return indptr
