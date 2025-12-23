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
完整GAOT模型 - MAGNO Encoder + Patch ViT + MAGNO Decoder的端到端实现
输入: 配置GAOTConfig | 输出: GAOT模型实例 | 地位: 顶层模型，集成所有组件
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from dataclasses import dataclass
from typing import List
from typing import Optional

import paddle
import paddle.nn as nn

from .attn import Transformer
from .attn import TransformerConfig
from .magno import MAGNOConfig
from .magno import MAGNODecoder
from .magno import MAGNOEncoder


@dataclass
class GAOTConfig:
    """GAOT model configuration."""

    # Input/output
    input_size: int
    output_size: int

    # Coordinate configuration
    coord_dim: int = 2  # 2D or 3D
    latent_tokens_size: List[int] = None  # [H, W] for 2D or [H, W, D] for 3D

    # MAGNO configuration
    magno: MAGNOConfig = None

    # Transformer configuration
    transformer: TransformerConfig = None

    def __post_init__(self):
        if self.latent_tokens_size is None:
            # Default latent grid size
            if self.coord_dim == 2:
                self.latent_tokens_size = [32, 32]
            else:
                self.latent_tokens_size = [16, 16, 16]

        if self.magno is None:
            self.magno = MAGNOConfig(coord_dim=self.coord_dim)

        if self.transformer is None:
            self.transformer = TransformerConfig()


class GAOT(nn.Layer):
    """
    Geometry-Aware Operator Transformer.

    Architecture: MAGNO Encoder → Vision Transformer → MAGNO Decoder

    Supports:
    - 2D and 3D coordinate spaces
    - Fixed coordinates (fx) and variable coordinates (vx) modes
    - Multi-scale feature extraction
    - Geometric embedding

    Parameters
    ----------
    input_size : int
        Input feature dimension
    output_size : int
        Output feature dimension
    config : GAOTConfig
        Model configuration

    Examples
    --------
    >>> config = GAOTConfig(
    ...     input_size=1,
    ...     output_size=1,
    ...     coord_dim=2,
    ...     latent_tokens_size=[32, 32]
    ... )
    >>> model = GAOT(config=config)
    >>> # Fixed coordinates mode
    >>> x_coord = paddle.randn([1024, 2])  # Physical coordinates
    >>> pndata = paddle.randn([8, 1024, 1])  # Input features [batch, N, channels]
    >>> latent_coord = paddle.randn([1024, 2])  # Latent grid coordinates
    >>> output = model(latent_coord, x_coord, pndata)
    >>> print(output.shape)  # [8, 1024, 1]
    """

    def __init__(self, config: GAOTConfig):
        super().__init__()

        # Store configuration
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.coord_dim = config.coord_dim
        self.node_latent_size = config.magno.lifting_channels
        self.patch_size = config.transformer.patch_size

        # Validate and store latent token dimensions
        latent_tokens_size = config.latent_tokens_size
        if self.coord_dim == 2:
            if len(latent_tokens_size) != 2:
                raise ValueError(
                    f"For 2D, latent_tokens_size must have 2 dimensions, "
                    f"got {len(latent_tokens_size)}"
                )
            self.H = latent_tokens_size[0]
            self.W = latent_tokens_size[1]
            self.D = None
        else:  # 3D
            if len(latent_tokens_size) != 3:
                raise ValueError(
                    f"For 3D, latent_tokens_size must have 3 dimensions, "
                    f"got {len(latent_tokens_size)}"
                )
            self.H = latent_tokens_size[0]
            self.W = latent_tokens_size[1]
            self.D = latent_tokens_size[2]

        # Initialize encoder, processor, and decoder
        self.encoder = self._init_encoder(config)
        self.processor, self.patch_linear, self.positions = self._init_processor(config)
        self.decoder = self._init_decoder(config)

        # Store positional embedding type
        self.positional_embedding_name = config.transformer.positional_embedding

    def _init_encoder(self, config: GAOTConfig) -> MAGNOEncoder:
        """Initialize MAGNO encoder."""
        return MAGNOEncoder(
            in_channels=config.input_size,
            out_channels=self.node_latent_size,
            config=config.magno,
        )

    def _init_processor(self, config: GAOTConfig):
        """Initialize Vision Transformer processor."""
        # Calculate patch volume
        if self.coord_dim == 2:
            patch_volume = self.patch_size * self.patch_size
        else:  # 3D
            patch_volume = self.patch_size**3

        # Patch linear projection
        patch_input_dim = patch_volume * self.node_latent_size
        patch_linear = nn.Linear(patch_input_dim, patch_input_dim)

        # Get patch positions
        positions = self._get_patch_positions()

        # Initialize transformer
        processor = Transformer(
            input_size=patch_input_dim,
            output_size=patch_input_dim,
            config=config.transformer,
        )

        return processor, patch_linear, positions

    def _init_decoder(self, config: GAOTConfig) -> MAGNODecoder:
        """Initialize MAGNO decoder."""
        return MAGNODecoder(
            in_channels=self.node_latent_size,
            out_channels=config.output_size,
            config=config.magno,
        )

    def _get_patch_positions(self) -> paddle.Tensor:
        """
        Generate positional embeddings for patches.

        Returns
        -------
        paddle.Tensor
            Patch positions [num_patches, coord_dim]
        """
        P = self.patch_size

        if self.coord_dim == 2:
            num_patches_H = self.H // P
            num_patches_W = self.W // P

            # Create meshgrid
            h_idx = paddle.arange(num_patches_H, dtype="float32")
            w_idx = paddle.arange(num_patches_W, dtype="float32")

            grid_h, grid_w = paddle.meshgrid(h_idx, w_idx)
            positions = paddle.stack([grid_h, grid_w], axis=-1)
            positions = positions.reshape([-1, 2])
        else:  # 3D
            num_patches_H = self.H // P
            num_patches_W = self.W // P
            num_patches_D = self.D // P

            h_idx = paddle.arange(num_patches_H, dtype="float32")
            w_idx = paddle.arange(num_patches_W, dtype="float32")
            d_idx = paddle.arange(num_patches_D, dtype="float32")

            grid_h, grid_w, grid_d = paddle.meshgrid(h_idx, w_idx, d_idx)
            positions = paddle.stack([grid_h, grid_w, grid_d], axis=-1)
            positions = positions.reshape([-1, 3])

        return positions

    def _compute_absolute_embeddings(
        self, positions: paddle.Tensor, embed_dim: int
    ) -> paddle.Tensor:
        """
        Compute absolute positional embeddings using sinusoidal encoding.

        Parameters
        ----------
        positions : paddle.Tensor [num_patches, coord_dim]
            Patch positions
        embed_dim : int
            Embedding dimension

        Returns
        -------
        paddle.Tensor [num_patches, embed_dim]
            Positional embeddings
        """
        num_pos_dims = positions.shape[1]
        dim_per_coord = embed_dim // (2 * num_pos_dims)

        # Frequency sequence
        freq_seq = paddle.arange(dim_per_coord, dtype="float32")
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_per_coord))

        # Sinusoidal encoding
        sinusoid_inp = positions.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1
        )

        # Flatten
        pos_emb = pos_emb.reshape([positions.shape[0], -1])

        return pos_emb

    def encode(
        self,
        x_coord: paddle.Tensor,
        pndata: paddle.Tensor,
        latent_tokens_coord: paddle.Tensor,
        encoder_nbrs: Optional[List] = None,
    ) -> paddle.Tensor:
        """
        Encode physical nodes to latent grid.

        Parameters
        ----------
        x_coord : paddle.Tensor
            Physical coordinates
            - fx mode: [num_nodes, coord_dim]
            - vx mode: [batch, num_nodes, coord_dim]
        pndata : paddle.Tensor [batch, num_nodes, input_size]
            Physical node features
        latent_tokens_coord : paddle.Tensor [num_latent, coord_dim]
            Latent grid coordinates
        encoder_nbrs : List, optional
            Precomputed encoder neighbors

        Returns
        -------
        paddle.Tensor [batch, num_latent, node_latent_size]
            Encoded latent features
        """
        encoded = self.encoder(
            x_coord=x_coord,
            pndata=pndata,
            latent_tokens_coord=latent_tokens_coord,
            encoder_nbrs=encoder_nbrs,
        )
        return encoded

    def process(
        self, rndata: paddle.Tensor, condition: Optional[float] = None
    ) -> paddle.Tensor:
        """
        Process latent features through Vision Transformer.

        Parameters
        ----------
        rndata : paddle.Tensor [batch, num_latent, node_latent_size]
            Latent node features
        condition : float, optional
            Conditioning value (not used currently)

        Returns
        -------
        paddle.Tensor [batch, num_latent, node_latent_size]
            Processed features
        """
        batch_size = rndata.shape[0]
        n_latent = rndata.shape[1]
        C = rndata.shape[2]
        P = self.patch_size

        # Reshape to patches
        if self.coord_dim == 2:
            H, W = self.H, self.W

            assert n_latent == H * W, f"n_latent ({n_latent}) != H*W ({H}*{W})"
            assert (
                H % P == 0 and W % P == 0
            ), f"H({H}) and W({W}) must be divisible by P({P})"

            num_patches_H = H // P
            num_patches_W = W // P

            # Reshape: [batch, H*W, C] → [batch, num_patches, P*P*C]
            rndata = rndata.reshape([batch_size, H, W, C])
            rndata = rndata.reshape([batch_size, num_patches_H, P, num_patches_W, P, C])
            rndata = rndata.transpose([0, 1, 3, 2, 4, 5])
            rndata = rndata.reshape(
                [batch_size, num_patches_H * num_patches_W, P * P * C]
            )
        else:  # 3D
            H, W, D = self.H, self.W, self.D

            assert (
                n_latent == H * W * D
            ), f"n_latent ({n_latent}) != H*W*D ({H}*{W}*{D})"
            assert (
                H % P == 0 and W % P == 0 and D % P == 0
            ), f"H({H}), W({W}), D({D}) must be divisible by P({P})"

            num_patches_H = H // P
            num_patches_W = W // P
            num_patches_D = D // P

            # Reshape: [batch, H*W*D, C] → [batch, num_patches, P*P*P*C]
            rndata = rndata.reshape([batch_size, H, W, D, C])
            rndata = rndata.reshape(
                [batch_size, num_patches_H, P, num_patches_W, P, num_patches_D, P, C]
            )
            rndata = rndata.transpose([0, 1, 3, 5, 2, 4, 6, 7])
            rndata = rndata.reshape(
                [
                    batch_size,
                    num_patches_H * num_patches_W * num_patches_D,
                    P * P * P * C,
                ]
            )

        # Apply patch linear transformation
        rndata = self.patch_linear(rndata)

        # Add positional encoding
        pos = self.positions
        if self.positional_embedding_name == "absolute":
            patch_volume = P**self.coord_dim
            pos_emb = self._compute_absolute_embeddings(
                pos, patch_volume * self.node_latent_size
            )
            rndata = rndata + pos_emb.unsqueeze(0)
            relative_positions = None
        elif self.positional_embedding_name == "rope":
            relative_positions = pos
        else:
            relative_positions = None

        # Apply transformer
        rndata = self.processor(
            rndata, condition=condition, relative_positions=relative_positions
        )

        # Reshape back to latent grid
        if self.coord_dim == 2:
            rndata = rndata.reshape([batch_size, num_patches_H, num_patches_W, P, P, C])
            rndata = rndata.transpose([0, 1, 3, 2, 4, 5])
            rndata = rndata.reshape([batch_size, H * W, C])
        else:  # 3D
            rndata = rndata.reshape(
                [batch_size, num_patches_H, num_patches_W, num_patches_D, P, P, P, C]
            )
            rndata = rndata.transpose([0, 1, 4, 2, 5, 3, 6, 7])
            rndata = rndata.reshape([batch_size, H * W * D, C])

        return rndata

    def decode(
        self,
        latent_tokens_coord: paddle.Tensor,
        rndata: paddle.Tensor,
        query_coord: paddle.Tensor,
        decoder_nbrs: Optional[List] = None,
    ) -> paddle.Tensor:
        """
        Decode latent features to query points.

        Parameters
        ----------
        latent_tokens_coord : paddle.Tensor [num_latent, coord_dim]
            Latent grid coordinates
        rndata : paddle.Tensor [batch, num_latent, node_latent_size]
            Latent features
        query_coord : paddle.Tensor
            Query coordinates
            - fx mode: [num_nodes, coord_dim]
            - vx mode: [batch, num_nodes, coord_dim]
        decoder_nbrs : List, optional
            Precomputed decoder neighbors

        Returns
        -------
        paddle.Tensor [batch, num_nodes, output_size]
            Decoded output features
        """
        decoded = self.decoder(
            latent_tokens_coord=latent_tokens_coord,
            rndata=rndata,
            query_coord=query_coord,
            decoder_nbrs=decoder_nbrs,
        )
        return decoded

    def forward(
        self,
        latent_tokens_coord: paddle.Tensor,
        xcoord: paddle.Tensor,
        pndata: paddle.Tensor,
        query_coord: Optional[paddle.Tensor] = None,
        encoder_nbrs: Optional[List] = None,
        decoder_nbrs: Optional[List] = None,
        condition: Optional[float] = None,
    ) -> paddle.Tensor:
        """
        Forward pass for GAOT model.

        Parameters
        ----------
        latent_tokens_coord : paddle.Tensor [num_latent, coord_dim]
            Latent grid coordinates
        xcoord : paddle.Tensor
            Physical coordinates
            - fx mode: [num_nodes, coord_dim]
            - vx mode: [batch, num_nodes, coord_dim]
        pndata : paddle.Tensor [batch, num_nodes, input_size]
            Input features on physical nodes
        query_coord : paddle.Tensor, optional
            Query coordinates for output (defaults to xcoord)
        encoder_nbrs : List, optional
            Precomputed encoder neighbors
        decoder_nbrs : List, optional
            Precomputed decoder neighbors
        condition : float, optional
            Conditioning value

        Returns
        -------
        paddle.Tensor [batch, num_query, output_size]
            Output features on query points
        """
        # Encode: Physical nodes → Latent grid
        rndata = self.encode(
            x_coord=xcoord,
            pndata=pndata,
            latent_tokens_coord=latent_tokens_coord,
            encoder_nbrs=encoder_nbrs,
        )

        # Process: Apply Vision Transformer on latent grid
        rndata = self.process(rndata=rndata, condition=condition)

        # Decode: Latent grid → Query points
        if query_coord is None:
            query_coord = xcoord

        output = self.decode(
            latent_tokens_coord=latent_tokens_coord,
            rndata=rndata,
            query_coord=query_coord,
            decoder_nbrs=decoder_nbrs,
        )

        return output
