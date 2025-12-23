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
几何嵌入模块 - GeometricEmbedding实现（statistical/pointnet方法）
输入: 坐标点和邻居信息 | 输出: 几何特征 | 地位: 核心组件，被MAGNO使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from typing import Dict
from typing import Literal

import paddle
import paddle.nn as nn

from .mlp import ChannelMLP
from .utils.scatter import scatter_max
from .utils.scatter import scatter_mean
from .utils.scatter import scatter_sum


def node_pos_encode(pos: paddle.Tensor) -> paddle.Tensor:
    """
    Positional encoding for node coordinates.

    Parameters
    ----------
    pos : paddle.Tensor [N, D]
        Node positions

    Returns
    -------
    paddle.Tensor [N, D*8]
        Encoded positions using sin/cos at multiple frequencies
    """
    pos.shape[-1]

    # Multiple frequency scales
    freq_bands = 2 ** paddle.arange(0, 4, dtype=pos.dtype)  # [1, 2, 4, 8]

    encoded = []
    for freq in freq_bands:
        encoded.append(paddle.sin(freq * pos))
        encoded.append(paddle.cos(freq * pos))

    return paddle.concat(encoded, axis=-1)  # [N, D*8]


class GeometricEmbedding(nn.Layer):
    """
    Geometric embedding layer.

    Encodes local geometric properties of neighborhoods into features.
    Supports two methods:
    - 'statistical': Statistical features (mean, std, covariance, etc.)
    - 'pointnet': PointNet-style feature aggregation

    Parameters
    ----------
    input_dim : int
        Input coordinate dimension (2 for 2D, 3 for 3D)
    output_dim : int
        Output feature dimension
    method : str, default 'statistical'
        Embedding method: 'statistical' or 'pointnet'
    pooling : str, default 'max'
        Pooling method for pointnet: 'max', 'mean', or 'sum'
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        method: Literal["statistical", "pointnet"] = "statistical",
        pooling: Literal["max", "mean", "sum"] = "max",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.pooling = pooling

        if method == "statistical":
            # Statistical features MLP
            # Features: D_avg, D_std, Delta (D dimensions), Cov (D*D dimensions)
            stat_feat_dim = 2 + input_dim + input_dim * input_dim
            self.mlp = ChannelMLP(
                in_channels=stat_feat_dim,
                hidden_channels=output_dim * 2,
                out_channels=output_dim,
                n_layers=2,
            )
        elif method == "pointnet":
            # PointNet-style encoder
            self.encoder = ChannelMLP(
                in_channels=input_dim,
                hidden_channels=output_dim * 2,
                out_channels=output_dim,
                n_layers=2,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(
        self,
        input_geom: paddle.Tensor,
        latent_queries: paddle.Tensor,
        spatial_nbrs: Dict[str, paddle.Tensor],
    ) -> paddle.Tensor:
        """
        几何嵌入 - 接口与PyTorch完全一致

        Parameters
        ----------
        input_geom : paddle.Tensor [n, coord_dim]
            输入点的坐标（从中提取邻居位置）
        latent_queries : paddle.Tensor [m, coord_dim]
            查询点的坐标
        spatial_nbrs : Dict
            邻居信息字典，包含:
            - 'edge_index': [2, E] 边 [query_idx, key_idx]

        Returns
        -------
        paddle.Tensor [m, output_dim]
            几何嵌入特征
        """
        if self.method == "statistical":
            return self._statistical_embedding(input_geom, latent_queries, spatial_nbrs)
        elif self.method == "pointnet":
            return self._pointnet_embedding(input_geom, latent_queries, spatial_nbrs)

    def _statistical_embedding(
        self,
        input_geom: paddle.Tensor,
        latent_queries: paddle.Tensor,
        spatial_nbrs: Dict[str, paddle.Tensor],
    ) -> paddle.Tensor:
        """
        Statistical geometric embedding.

        Computes statistical properties of neighborhoods:
        - Average distance to neighbors
        - Standard deviation of distances
        - Displacement from centroid
        - Local covariance matrix
        """
        edge_index = spatial_nbrs["edge_index"]  # [2, E]
        query_indices = edge_index[0]  # [E]
        key_indices = edge_index[1]  # [E]
        num_queries = latent_queries.shape[0]

        # 从input_geom提取邻居位置
        nbr_pos = input_geom[key_indices]  # [E, coord_dim]

        # 1. Compute distances
        query_pos_expanded = latent_queries[query_indices]  # [E, D]
        distances = paddle.norm(nbr_pos - query_pos_expanded, axis=-1)  # [E]

        # 2. Average distance
        D_avg = scatter_mean(
            distances.unsqueeze(-1), query_indices, dim=0, dim_size=num_queries
        ).squeeze(
            -1
        )  # [num_queries]

        # 3. Standard deviation
        distances_sq = distances**2
        E_X2 = scatter_mean(
            distances_sq.unsqueeze(-1), query_indices, dim=0, dim_size=num_queries
        ).squeeze(
            -1
        )  # [num_queries]

        D_std = paddle.sqrt(
            paddle.maximum(E_X2 - D_avg**2, paddle.zeros_like(E_X2))
        )  # [num_queries]

        # 4. Centroid displacement
        nbr_centroid = scatter_mean(
            nbr_pos, query_indices, dim=0, dim_size=num_queries
        )  # [num_queries, D]

        Delta = nbr_centroid - latent_queries  # [num_queries, D]

        # 5. Local covariance
        nbr_centered = nbr_pos - nbr_centroid[query_indices]  # [E, D]

        # Compute outer product: [E, D, D]
        cov_components = nbr_centered.unsqueeze(2) * nbr_centered.unsqueeze(1)

        # Sum over neighbors
        cov_sum = scatter_sum(
            cov_components.reshape([-1, self.input_dim * self.input_dim]),
            query_indices,
            dim=0,
            dim_size=num_queries,
        )  # [num_queries, D*D]

        # Count neighbors per query
        ones = paddle.ones([len(query_indices), 1], dtype=nbr_pos.dtype)
        N_i = scatter_sum(ones, query_indices, dim=0, dim_size=num_queries).squeeze(-1)
        N_i_clamped = paddle.maximum(N_i, paddle.ones_like(N_i))

        # Normalize covariance
        cov = cov_sum / N_i_clamped.unsqueeze(-1)  # [num_queries, D*D]

        # Concatenate all statistical features
        stat_features = paddle.concat(
            [
                D_avg.unsqueeze(-1),  # [num_queries, 1]
                D_std.unsqueeze(-1),  # [num_queries, 1]
                Delta,  # [num_queries, D]
                cov,  # [num_queries, D*D]
            ],
            axis=-1,
        )  # [num_queries, 2 + D + D*D]

        # Pass through MLP
        embedding = self.mlp(stat_features)  # [num_queries, output_dim]

        return embedding

    def _pointnet_embedding(
        self,
        input_geom: paddle.Tensor,
        latent_queries: paddle.Tensor,
        spatial_nbrs: Dict[str, paddle.Tensor],
    ) -> paddle.Tensor:
        """
        PointNet-style geometric embedding.

        Applies MLP to relative positions and aggregates with pooling.
        """
        edge_index = spatial_nbrs["edge_index"]  # [2, E]
        query_indices = edge_index[0]  # [E]
        key_indices = edge_index[1]  # [E]
        num_queries = latent_queries.shape[0]

        # 从input_geom提取邻居位置
        nbr_pos = input_geom[key_indices]  # [E, coord_dim]

        # Center neighbors around query points
        query_pos_expanded = latent_queries[query_indices]  # [E, D]
        nbr_centered = nbr_pos - query_pos_expanded  # [E, D]

        # Encode neighbor features
        nbr_features = self.encoder(nbr_centered)  # [E, output_dim]

        # Pool features per query
        if self.pooling == "max":
            pooled_features, _ = scatter_max(
                nbr_features, query_indices, dim=0, dim_size=num_queries
            )
        elif self.pooling == "mean":
            pooled_features = scatter_mean(
                nbr_features, query_indices, dim=0, dim_size=num_queries
            )
        elif self.pooling == "sum":
            pooled_features = scatter_sum(
                nbr_features, query_indices, dim=0, dim_size=num_queries
            )
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled_features  # [num_queries, output_dim]
