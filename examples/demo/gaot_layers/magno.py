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
MAGNO模块 - 多尺度图注意力编解码器（MAGNOEncoder/Decoder）
输入: 点云坐标和特征 | 输出: 编码/解码后特征 | 地位: 核心组件，被完整GAOT模型使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import paddle
import paddle.nn as nn

from .agno import AGNO
from .gemb import GeometricEmbedding
from .gemb import node_pos_encode
from .mlp import ChannelMLP
from .utils.neighbor_search import NeighborSearch


@dataclass
class MAGNOConfig:
    """MAGNO Configuration."""

    # Core Parameters
    coord_dim: int = 2
    radius: float = 0.033
    hidden_size: int = 64
    mlp_layers: int = 3
    lifting_channels: int = 32

    # Multi-scale
    scales: List[float] = field(default_factory=lambda: [1.0])
    use_scale_weights: bool = False

    # Attention and Embedding
    use_attention: bool = True
    attention_type: str = "cosine"
    use_geoembed: bool = True
    embedding_method: str = "statistical"
    pooling: str = "max"

    # Transform and Sampling
    transform_type: str = "linear"
    sampling_strategy: Optional[str] = None
    max_neighbors: Optional[int] = None
    sample_ratio: Optional[float] = None

    # Advanced
    node_embedding: bool = False
    neighbor_search_method: str = "scipy"
    use_torch_scatter: bool = True
    neighbor_strategy: str = "radius"
    precompute_edges: bool = False


class MAGNOEncoder(nn.Layer):
    """
    MAGNO Encoder: Physical points → Latent grid.

    Supports:
    - 2D and 3D coordinates
    - Fixed coordinates (fx) and variable coordinates (vx)
    - Multi-scale feature extraction
    - Geometric embedding

    Parameters
    ----------
    in_channels : int
        Input feature dimension
    out_channels : int
        Output feature dimension
    config : MAGNOConfig
        Configuration object
    """

    def __init__(self, in_channels: int, out_channels: int, config: MAGNOConfig):
        super().__init__()

        self.config = config
        self.coord_dim = config.coord_dim
        self.scales = config.scales
        self.use_scale_weights = config.use_scale_weights
        self.precompute_edges = config.precompute_edges
        self.use_geoembed = config.use_geoembed
        self.node_embedding = config.node_embedding

        # Neighbor search
        self.nb_search = NeighborSearch(method=config.neighbor_search_method)
        self.neighbor_cache = {}

        # Edge sampling parameters
        self.sampling_strategy = config.sampling_strategy
        self.max_neighbors = config.max_neighbors
        self.sample_ratio = config.sample_ratio

        # Kernel input dimension
        kernel_coord_dim = self._compute_kernel_coord_dim()
        kernel_input_dim = kernel_coord_dim * 2

        if config.transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            kernel_input_dim += in_channels

        # MLP layer sizes
        mlp_sizes = (
            [kernel_input_dim]
            + [config.hidden_size] * config.mlp_layers
            + [out_channels]
        )

        # Core modules
        self.agno = AGNO(
            channel_mlp_layers=mlp_sizes,
            transform_type=config.transform_type,
            use_attn=config.use_attention,
            attention_type=config.attention_type,
            coord_dim=kernel_coord_dim,
            use_torch_scatter=config.use_torch_scatter,
        )

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_size,
            out_channels=out_channels,
            n_layers=1,
        )

        # Geometric embedding
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=out_channels,
                method=config.embedding_method,
                pooling=config.pooling,
            )
            # Recovery layer to merge AGNO output and geometric embedding
            # Input: concatenated features [batch, channels+geoembed_channels, nodes]
            # Output: [batch, out_channels, nodes]
            self.recovery = nn.Sequential(
                nn.Linear(2 * out_channels, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, out_channels),
            )

        # Scale weighting
        if self.use_scale_weights:
            self.scale_weighting = nn.Sequential(
                nn.Linear(kernel_coord_dim, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, len(self.scales)),
            )
            self.scale_weight_activation = nn.Softmax(axis=-1)

    def _compute_kernel_coord_dim(self) -> int:
        """Compute effective coordinate dimension for kernel."""
        coord_dim = self.coord_dim
        if self.node_embedding:
            coord_dim = self.coord_dim * 4 * 2
        return coord_dim

    def _detect_coordinate_mode(self, x_coord: paddle.Tensor) -> Literal["fx", "vx"]:
        """Auto-detect coordinate mode."""
        if x_coord.ndim == 2:
            return "fx"
        elif x_coord.ndim == 3:
            return "vx"
        else:
            raise ValueError(f"x_coord must be 2D or 3D, got shape {x_coord.shape}")

    def _compute_neighbors(
        self,
        x_coord: paddle.Tensor,
        latent_coord: paddle.Tensor,
        mode: Literal["fx", "vx"],
    ) -> List:
        """Compute neighbor lists with caching."""
        cache_key = (
            f"enc_{mode}_{x_coord.shape}_{latent_coord.shape}_{tuple(self.scales)}"
        )

        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]

        neighbors_per_scale = []

        if mode == "fx":
            # Fixed coordinates
            for scale in self.scales:
                scaled_radius = self.config.radius * scale
                edge_index = self.nb_search.radius_search(
                    queries=latent_coord,
                    keys=x_coord,
                    radius=scaled_radius,
                    max_neighbors=self.max_neighbors,
                )
                # Add neighbor positions for geometric embedding
                key_indices = edge_index[1]  # [E]
                nbr_pos = x_coord[key_indices]  # [E, coord_dim]
                neighbors = {"edge_index": edge_index, "pos": nbr_pos}
                neighbors_per_scale.append(neighbors)
        else:
            # Variable coordinates
            batch_size = x_coord.shape[0]
            neighbors_per_batch = []

            for b in range(batch_size):
                neighbors_per_scale_batch = []
                for scale in self.scales:
                    scaled_radius = self.config.radius * scale
                    edge_index = self.nb_search.radius_search(
                        queries=latent_coord,
                        keys=x_coord[b],
                        radius=scaled_radius,
                        max_neighbors=self.max_neighbors,
                    )
                    # Add neighbor positions for geometric embedding
                    key_indices = edge_index[1]  # [E]
                    nbr_pos = x_coord[b][key_indices]  # [E, coord_dim]
                    neighbors = {"edge_index": edge_index, "pos": nbr_pos}
                    neighbors_per_scale_batch.append(neighbors)
                neighbors_per_batch.append(neighbors_per_scale_batch)
            neighbors_per_scale = neighbors_per_batch

        self.neighbor_cache[cache_key] = neighbors_per_scale
        return neighbors_per_scale

    def forward(
        self,
        x_coord: paddle.Tensor,
        pndata: paddle.Tensor,
        latent_tokens_coord: paddle.Tensor,
        encoder_nbrs: Optional[Union[List, List[List]]] = None,
    ) -> paddle.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_coord : paddle.Tensor
            Physical coordinates
            - fx mode: [num_nodes, coord_dim]
            - vx mode: [batch_size, num_nodes, coord_dim]
        pndata : paddle.Tensor [batch_size, num_nodes, in_channels]
            Physical node features
        latent_tokens_coord : paddle.Tensor [num_latent, coord_dim]
            Target latent grid coordinates
        encoder_nbrs : Optional
            Precomputed neighbors

        Returns
        -------
        paddle.Tensor [batch_size, num_latent, out_channels]
            Encoded features on latent grid
        """
        # Detect coordinate mode
        coord_mode = self._detect_coordinate_mode(x_coord)
        pndata.shape[0]

        # Validate inputs
        if coord_mode == "fx":
            x_coord.shape[0]
        else:
            x_coord.shape[1]

        # Compute or use precomputed neighbors
        if self.precompute_edges:
            if encoder_nbrs is None:
                raise ValueError("encoder_nbrs required when precompute_edges=True")
            neighbors_per_scale = encoder_nbrs
        else:
            neighbors_per_scale = self._compute_neighbors(
                x_coord, latent_tokens_coord, coord_mode
            )

        # Lift input features
        # pndata input shape: [batch, nodes, in_channels]
        pndata = self.lifting(pndata)  # [batch, nodes, lifting_channels]
        pndata = pndata.transpose([0, 2, 1])  # [batch, lifting_channels, nodes]

        # Prepare scale weights
        if self.use_scale_weights:
            scale_weights = self.scale_weighting(latent_tokens_coord)
            scale_weights = self.scale_weight_activation(scale_weights)

        # Process each scale
        if coord_mode == "fx":
            encoded_scales = self._forward_fx_mode(
                x_coord, pndata, latent_tokens_coord, neighbors_per_scale
            )
        else:
            encoded_scales = self._forward_vx_mode(
                x_coord, pndata, latent_tokens_coord, neighbors_per_scale
            )

        # Combine scales
        if len(encoded_scales) == 1:
            encoded = encoded_scales[0]
        else:
            if self.use_scale_weights:
                encoded = paddle.zeros_like(encoded_scales[0])
                for i, enc in enumerate(encoded_scales):
                    weights = scale_weights[:, i : i + 1].unsqueeze(0)
                    encoded += weights * enc
            else:
                encoded = paddle.stack(encoded_scales, axis=0).mean(axis=0)

        return encoded

    def _forward_fx_mode(self, x_coord, pndata, latent_coord, neighbors_per_scale):
        """Forward for fixed coordinates."""
        batch_size = pndata.shape[0]
        encoded_scales = []

        for neighbors in neighbors_per_scale:
            # Prepare coordinates
            if self.node_embedding:
                phys_coord = node_pos_encode(x_coord)
                latent_coord_proc = node_pos_encode(latent_coord)
            else:
                phys_coord = x_coord
                latent_coord_proc = latent_coord

            # Process each batch - use new AGNO interface
            # pndata: [batch, channels, nodes] -> transpose for AGNO: [batch, nodes, channels]
            pndata_for_agno = pndata.transpose([0, 2, 1])  # [batch, nodes, channels]

            # Call AGNO with new interface (y=coordinates, f_y=features)
            encoded = self.agno(
                y=phys_coord,  # 物理点坐标 [n, coord_dim]
                neighbors=neighbors,
                x=latent_coord_proc,  # 查询点坐标 [m, coord_dim]
                f_y=pndata_for_agno,  # 输入特征 [batch, n, channels]
            )  # Returns: [batch, m, out_channels]

            # Apply geometric embedding
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    input_geom=phys_coord,  # 物理点坐标
                    latent_queries=latent_coord_proc,  # 查询点坐标
                    spatial_nbrs=neighbors,  # 邻居信息
                )  # Returns: [m, geoembed_channels]

                # Expand for batch
                geoembedding = geoembedding.unsqueeze(0).expand(
                    [batch_size, -1, -1]
                )  # [batch, m, geoembed_channels]

                # Concatenate and recover
                encoded = paddle.concat(
                    [encoded, geoembedding], axis=-1
                )  # [batch, m, channels+geoembed_channels]
                # Recovery expects [batch, m, 2*channels]
                encoded = self.recovery(encoded)  # [batch, m, channels]

            encoded_scales.append(encoded)

        return encoded_scales

    def _forward_vx_mode(self, x_coord, pndata, latent_coord, neighbors_per_scale):
        """Forward for variable coordinates."""
        batch_size = x_coord.shape[0]
        encoded_scales = []

        for scale_idx, neighbors_batch in enumerate(neighbors_per_scale):
            encoded_batch = []

            for b in range(batch_size):
                neighbors = neighbors_batch[b]

                # Prepare coordinates
                if self.node_embedding:
                    phys_coord = node_pos_encode(x_coord[b])
                    latent_coord_proc = node_pos_encode(latent_coord)
                else:
                    phys_coord = x_coord[b]
                    latent_coord_proc = latent_coord

                # pndata: [batch, channels, nodes] -> get batch b and transpose
                pndata_b = pndata[b].transpose([1, 0])  # [nodes, channels]
                pndata_b_for_agno = pndata_b.unsqueeze(0)  # [1, nodes, channels]

                # Call AGNO with new interface
                encoded_b = self.agno(
                    y=phys_coord,  # 物理点坐标 [n, coord_dim]
                    neighbors=neighbors,
                    x=latent_coord_proc,  # 查询点坐标 [m, coord_dim]
                    f_y=pndata_b_for_agno,  # 输入特征 [1, n, channels]
                )  # Returns: [m, out_channels] (batch=1 so squeezed)

                # Geometric embedding
                if self.use_geoembed:
                    geoembedding = self.geoembed(
                        input_geom=phys_coord,  # 物理点坐标
                        latent_queries=latent_coord_proc,  # 查询点坐标
                        spatial_nbrs=neighbors,  # 邻居信息
                    )  # [m, geoembed_channels]

                    encoded_b = paddle.concat(
                        [encoded_b, geoembedding], axis=-1
                    )  # [m, 2*channels]
                    # Recovery expects [m, 2*channels]
                    encoded_b = self.recovery(encoded_b)  # [m, channels]

                encoded_batch.append(encoded_b.unsqueeze(0))  # [1, m, channels]

            encoded_scale = paddle.concat(encoded_batch, axis=0)
            encoded_scales.append(encoded_scale)

        return encoded_scales


class MAGNODecoder(nn.Layer):
    """
    MAGNO Decoder: Latent grid → Physical points.

    Parameters
    ----------
    in_channels : int
        Input feature dimension
    out_channels : int
        Output feature dimension
    config : MAGNOConfig
        Configuration object
    """

    def __init__(self, in_channels: int, out_channels: int, config: MAGNOConfig):
        super().__init__()

        self.config = config
        self.coord_dim = config.coord_dim
        self.scales = config.scales
        self.use_scale_weights = config.use_scale_weights
        self.precompute_edges = config.precompute_edges
        self.use_geoembed = config.use_geoembed
        self.node_embedding = config.node_embedding

        # Neighbor search
        self.nb_search = NeighborSearch(method=config.neighbor_search_method)
        self.neighbor_cache = {}

        # Edge sampling
        self.sampling_strategy = config.sampling_strategy
        self.max_neighbors = config.max_neighbors
        self.sample_ratio = config.sample_ratio

        # Kernel input dimension
        kernel_coord_dim = self._compute_kernel_coord_dim()
        kernel_input_dim = kernel_coord_dim * 2

        if config.transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            kernel_input_dim += in_channels

        # MLP sizes
        mlp_sizes = (
            [kernel_input_dim]
            + [config.hidden_size] * config.mlp_layers
            + [in_channels]
        )

        # Core modules
        self.agno = AGNO(
            channel_mlp_layers=mlp_sizes,
            transform_type=config.transform_type,
            use_attn=config.use_attention,
            attention_type=config.attention_type,
            coord_dim=kernel_coord_dim,
            use_torch_scatter=config.use_torch_scatter,
        )

        self.projection = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_size,
            out_channels=out_channels,
            n_layers=1,
        )

        # Geometric embedding
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=in_channels,
                method=config.embedding_method,
                pooling=config.pooling,
            )
            # Recovery layer to merge AGNO output and geometric embedding
            # Input: concatenated features [batch, channels+geoembed_channels, nodes]
            # Output: [batch, in_channels, nodes]
            self.recovery = nn.Sequential(
                nn.Linear(2 * in_channels, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, in_channels),
            )

        # Scale weighting
        if self.use_scale_weights:
            self.scale_weighting = nn.Sequential(
                nn.Linear(kernel_coord_dim, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, len(self.scales)),
            )
            self.scale_weight_activation = nn.Softmax(axis=-1)

    def _compute_kernel_coord_dim(self) -> int:
        coord_dim = self.coord_dim
        if self.node_embedding:
            coord_dim = self.coord_dim * 4 * 2
        return coord_dim

    def _detect_coordinate_mode(
        self, query_coord: paddle.Tensor
    ) -> Literal["fx", "vx"]:
        if query_coord.ndim == 2:
            return "fx"
        elif query_coord.ndim == 3:
            return "vx"
        else:
            raise ValueError(f"query_coord must be 2D or 3D, got {query_coord.shape}")

    def _compute_neighbors(
        self,
        latent_coord: paddle.Tensor,
        query_coord: paddle.Tensor,
        mode: Literal["fx", "vx"],
    ) -> List:
        """Compute neighbors with caching."""
        cache_key = (
            f"dec_{mode}_{latent_coord.shape}_{query_coord.shape}_{tuple(self.scales)}"
        )

        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]

        neighbors_per_scale = []

        if mode == "fx":
            for scale in self.scales:
                scaled_radius = self.config.radius * scale
                edge_index = self.nb_search.radius_search(
                    queries=query_coord,
                    keys=latent_coord,
                    radius=scaled_radius,
                    max_neighbors=self.max_neighbors,
                )
                # Add neighbor positions for geometric embedding
                key_indices = edge_index[1]  # [E]
                nbr_pos = latent_coord[key_indices]  # [E, coord_dim]
                neighbors = {"edge_index": edge_index, "pos": nbr_pos}
                neighbors_per_scale.append(neighbors)
        else:
            batch_size = query_coord.shape[0]
            neighbors_per_batch = []

            for b in range(batch_size):
                neighbors_per_scale_batch = []
                for scale in self.scales:
                    scaled_radius = self.config.radius * scale
                    edge_index = self.nb_search.radius_search(
                        queries=query_coord[b],
                        keys=latent_coord,
                        radius=scaled_radius,
                        max_neighbors=self.max_neighbors,
                    )
                    # Add neighbor positions for geometric embedding
                    key_indices = edge_index[1]  # [E]
                    nbr_pos = latent_coord[key_indices]  # [E, coord_dim]
                    neighbors = {"edge_index": edge_index, "pos": nbr_pos}
                    neighbors_per_scale_batch.append(neighbors)
                neighbors_per_batch.append(neighbors_per_scale_batch)
            neighbors_per_scale = neighbors_per_batch

        self.neighbor_cache[cache_key] = neighbors_per_scale
        return neighbors_per_scale

    def forward(
        self,
        latent_tokens_coord: paddle.Tensor,
        rndata: paddle.Tensor,
        query_coord: paddle.Tensor,
        decoder_nbrs: Optional[Union[List, List[List]]] = None,
    ) -> paddle.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        latent_tokens_coord : paddle.Tensor [num_latent, coord_dim]
            Latent grid coordinates
        rndata : paddle.Tensor [batch_size, num_latent, in_channels]
            Latent features
        query_coord : paddle.Tensor
            Query coordinates
            - fx: [num_nodes, coord_dim]
            - vx: [batch_size, num_nodes, coord_dim]
        decoder_nbrs : Optional
            Precomputed neighbors

        Returns
        -------
        paddle.Tensor [batch_size, num_nodes, out_channels]
            Decoded features
        """
        coord_mode = self._detect_coordinate_mode(query_coord)
        rndata.shape[0]

        # Compute neighbors
        if self.precompute_edges:
            if decoder_nbrs is None:
                raise ValueError("decoder_nbrs required when precompute_edges=True")
            neighbors_per_scale = decoder_nbrs
        else:
            neighbors_per_scale = self._compute_neighbors(
                latent_tokens_coord, query_coord, coord_mode
            )

        # Prepare scale weights
        if self.use_scale_weights:
            if coord_mode == "fx":
                scale_weights = self.scale_weighting(query_coord)
            else:
                scale_weights = self.scale_weighting(query_coord[0])
            scale_weights = self.scale_weight_activation(scale_weights)

        # Process each scale
        if coord_mode == "fx":
            decoded_scales = self._forward_fx_mode(
                latent_tokens_coord, rndata, query_coord, neighbors_per_scale
            )
        else:
            decoded_scales = self._forward_vx_mode(
                latent_tokens_coord, rndata, query_coord, neighbors_per_scale
            )

        # Combine scales
        if len(decoded_scales) == 1:
            decoded = decoded_scales[0]
        else:
            if self.use_scale_weights:
                decoded = paddle.zeros_like(decoded_scales[0])
                for i, dec in enumerate(decoded_scales):
                    weights = scale_weights[:, i : i + 1].unsqueeze(0)
                    decoded += weights * dec
            else:
                decoded = paddle.stack(decoded_scales, axis=0).mean(axis=0)

        # Final projection
        # decoded: [batch, num_nodes, in_channels]
        # projection expects: [batch, num_nodes, in_channels]
        decoded = self.projection(decoded)  # [batch, num_nodes, out_channels]

        return decoded

    def _forward_fx_mode(self, latent_coord, rndata, query_coord, neighbors_per_scale):
        """Forward for fixed coordinates."""
        batch_size = rndata.shape[0]
        decoded_scales = []

        for neighbors in neighbors_per_scale:
            if self.node_embedding:
                latent_coord_proc = node_pos_encode(latent_coord)
                query_coord_proc = node_pos_encode(query_coord)
            else:
                latent_coord_proc = latent_coord
                query_coord_proc = query_coord

            # Call AGNO with new interface (y=coordinates, f_y=features)
            # rndata: [batch, num_latent, channels]
            decoded = self.agno(
                y=latent_coord_proc,  # 潜在点坐标 [num_latent, coord_dim]
                neighbors=neighbors,
                x=query_coord_proc,  # 查询点坐标 [num_queries, coord_dim]
                f_y=rndata,  # 输入特征 [batch, num_latent, channels]
            )  # Returns: [batch, num_queries, out_channels]

            # Apply geometric embedding
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    input_geom=latent_coord_proc,  # 潜在点坐标
                    latent_queries=query_coord_proc,  # 查询点坐标
                    spatial_nbrs=neighbors,  # 邻居信息
                )  # Returns: [num_queries, geoembed_channels]

                # Expand for batch
                geoembedding = geoembedding.unsqueeze(0).expand([batch_size, -1, -1])

                # Concatenate and recover
                decoded = paddle.concat(
                    [decoded, geoembedding], axis=-1
                )  # [batch, num_queries, 2*channels]
                # Recovery expects [batch, num_queries, 2*channels]
                decoded = self.recovery(decoded)  # [batch, num_queries, channels]

            decoded_scales.append(decoded)

        return decoded_scales

    def _forward_vx_mode(self, latent_coord, rndata, query_coord, neighbors_per_scale):
        """Forward for variable coordinates."""
        batch_size = query_coord.shape[0]
        decoded_scales = []

        for neighbors_batch in neighbors_per_scale:
            decoded_batch = []

            for b in range(batch_size):
                neighbors = neighbors_batch[b]

                if self.node_embedding:
                    latent_coord_proc = node_pos_encode(latent_coord)
                    query_coord_proc = node_pos_encode(query_coord[b])
                else:
                    latent_coord_proc = latent_coord
                    query_coord_proc = query_coord[b]

                # rndata: [batch, num_latent, channels] -> get batch b
                rndata_b = rndata[b].unsqueeze(0)  # [1, num_latent, channels]

                # Call AGNO with new interface
                decoded_b = self.agno(
                    y=latent_coord_proc,  # 潜在点坐标 [num_latent, coord_dim]
                    neighbors=neighbors,
                    x=query_coord_proc,  # 查询点坐标 [num_queries, coord_dim]
                    f_y=rndata_b,  # 输入特征 [1, num_latent, channels]
                )  # Returns: [num_queries, out_channels] (batch=1 so squeezed)

                # Apply geometric embedding
                if self.use_geoembed:
                    geoembedding = self.geoembed(
                        input_geom=latent_coord_proc,  # 潜在点坐标
                        latent_queries=query_coord_proc,  # 查询点坐标
                        spatial_nbrs=neighbors,  # 邻居信息
                    )  # [num_queries, geoembed_channels]

                    decoded_b = paddle.concat(
                        [decoded_b, geoembedding], axis=-1
                    )  # [num_queries, 2*channels]
                    # Recovery expects [num_queries, 2*channels]
                    decoded_b = self.recovery(decoded_b)  # [num_queries, channels]

                decoded_batch.append(
                    decoded_b.unsqueeze(0)
                )  # [1, num_queries, channels]

            decoded_scale = paddle.concat(decoded_batch, axis=0)
            decoded_scales.append(decoded_scale)

        return decoded_scales
