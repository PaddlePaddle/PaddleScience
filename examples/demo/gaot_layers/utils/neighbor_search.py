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
邻居搜索模块 - radius/knn搜索 + 缓存机制（替代torch_cluster）
输入: 查询点和候选点 | 输出: 邻居边索引 | 地位: 基础工具，被MAGNO使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from scipy.spatial import cKDTree


class NeighborSearch:
    """
    Neighbor search for graph construction.

    Supports:
    - Radius search
    - KNN search
    - Multiple search methods
    - Caching for efficiency
    """

    def __init__(self, method: str = "scipy"):
        """
        Parameters
        ----------
        method : str
            Search method: 'scipy', 'gpu', 'chunked'
        """
        self.method = method
        self.cache = {}

    def radius_search(
        self,
        queries: paddle.Tensor,
        keys: paddle.Tensor,
        radius: float,
        max_neighbors: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Radius-based neighbor search.

        Parameters
        ----------
        queries : paddle.Tensor [N, D]
            Query points
        keys : paddle.Tensor [M, D]
            Key points (candidate neighbors)
        radius : float
            Search radius
        max_neighbors : int, optional
            Maximum neighbors per query
        cache_key : str, optional
            Cache key for reusing results

        Returns
        -------
        edge_index : paddle.Tensor [2, E]
            Edge indices [query_idx, key_idx]
        """
        # Check cache
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        # Convert to numpy for scipy
        queries_np = queries.numpy()
        keys_np = keys.numpy()

        # Build KDTree
        tree = cKDTree(keys_np)

        # Query all neighbors within radius
        neighbors_list = tree.query_ball_point(queries_np, r=radius)

        # Convert to edge index format
        query_indices = []
        key_indices = []

        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) > 0:
                # Limit neighbors if specified
                if max_neighbors and len(neighbors) > max_neighbors:
                    neighbors = neighbors[:max_neighbors]

                query_indices.extend([i] * len(neighbors))
                key_indices.extend(neighbors)

        # Create edge index tensor
        if len(query_indices) > 0:
            edge_index = paddle.to_tensor(
                np.array([query_indices, key_indices], dtype=np.int64)
            )
        else:
            # Empty graph
            edge_index = paddle.zeros([2, 0], dtype="int64")

        # Cache result
        if cache_key:
            self.cache[cache_key] = edge_index

        return edge_index

    def knn_search(
        self,
        queries: paddle.Tensor,
        keys: paddle.Tensor,
        k: int,
        cache_key: Optional[str] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        K-nearest neighbor search.

        Parameters
        ----------
        queries : paddle.Tensor [N, D]
            Query points
        keys : paddle.Tensor [M, D]
            Key points
        k : int
            Number of nearest neighbors
        cache_key : str, optional
            Cache key

        Returns
        -------
        edge_index : paddle.Tensor [2, E]
            Edge indices
        """
        # Check cache
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        # Convert to numpy
        queries_np = queries.numpy()
        keys_np = keys.numpy()

        # Build KDTree
        tree = cKDTree(keys_np)

        # Query k nearest neighbors
        distances, indices = tree.query(queries_np, k=k)

        # Build edge index
        n_queries = len(queries_np)
        query_indices = np.repeat(np.arange(n_queries), k)
        key_indices = indices.flatten()

        edge_index = paddle.to_tensor(
            np.array([query_indices, key_indices], dtype=np.int64)
        )

        # Cache result
        if cache_key:
            self.cache[cache_key] = edge_index

        return edge_index

    def clear_cache(self):
        """Clear neighbor cache."""
        self.cache.clear()

    def __call__(
        self,
        queries: paddle.Tensor,
        keys: paddle.Tensor,
        radius: Optional[float] = None,
        k: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Flexible neighbor search.

        Uses radius search if radius is provided, else KNN.
        """
        if radius is not None:
            return self.radius_search(queries, keys, radius, **kwargs)
        elif k is not None:
            return self.knn_search(queries, keys, k, **kwargs)
        else:
            raise ValueError("Must provide either radius or k")
