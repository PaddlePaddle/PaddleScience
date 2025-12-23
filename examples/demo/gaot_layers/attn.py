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
Transformer模块 - Patch Vision Transformer实现
输入: 潜在空间特征 | 输出: Transformer处理后特征 | 地位: 核心组件，被完整GAOT模型使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from dataclasses import dataclass
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


@dataclass
class TransformerConfig:
    """Transformer configuration."""

    patch_size: int = 8
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 8
    ffn_multiplier: int = 4
    positional_embedding: str = "absolute"  # 'absolute' or 'rope'
    use_attn_norm: bool = True
    use_ffn_norm: bool = True
    norm_eps: float = 1e-6
    atten_dropout: float = 0.0


class MultiHeadAttention(nn.Layer):
    """
    Multi-head self-attention.

    Parameters
    ----------
    input_size : int
        Input dimension
    hidden_size : int
        Hidden dimension for Q, K, V
    num_heads : int
        Number of attention heads
    dropout : float
        Attention dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.k_proj = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.v_proj = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.o_proj = nn.Linear(hidden_size, input_size, bias_attr=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self, x: paddle.Tensor, relative_positions: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : paddle.Tensor [batch, seq_len, input_size]
            Input tensor
        relative_positions : paddle.Tensor, optional
            Relative position embeddings (for RoPE)

        Returns
        -------
        paddle.Tensor [batch, seq_len, input_size]
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, hidden_size]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        q = q.transpose([0, 2, 1, 3])  # [batch, num_heads, seq_len, head_dim]

        k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        k = k.transpose([0, 2, 1, 3])

        v = v.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        v = v.transpose([0, 2, 1, 3])

        # Scaled dot-product attention
        attn_scores = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn_weights = F.softmax(attn_scores, axis=-1)

        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = paddle.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_len, -1])

        # Output projection
        output = self.o_proj(attn_output)

        return output


class FFN(nn.Layer):
    """
    Feed-forward network.

    Parameters
    ----------
    input_size : int
        Input dimension
    hidden_size : int
        Hidden dimension
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.w1 = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.w2 = nn.Linear(hidden_size, input_size, bias_attr=False)
        self.w3 = nn.Linear(input_size, hidden_size, bias_attr=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """SwiGLU activation."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Layer):
    """
    Transformer encoder block.

    Parameters
    ----------
    input_size : int
        Input dimension
    hidden_size : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    ffn_multiplier : int
        FFN hidden size multiplier
    use_attn_norm : bool
        Whether to use norm before attention
    use_ffn_norm : bool
        Whether to use norm before FFN
    norm_eps : float
        Epsilon for layer normalization
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        use_attn_norm: bool = True,
        use_ffn_norm: bool = True,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.use_attn_norm = use_attn_norm
        self.use_ffn_norm = use_ffn_norm

        # Attention
        if use_attn_norm:
            self.attn_norm = nn.LayerNorm(input_size, epsilon=norm_eps)
        self.attn = MultiHeadAttention(input_size, hidden_size, num_heads, dropout)

        # FFN
        if use_ffn_norm:
            self.ffn_norm = nn.LayerNorm(input_size, epsilon=norm_eps)
        self.ffn = FFN(input_size, input_size * ffn_multiplier)

    def forward(
        self, x: paddle.Tensor, relative_positions: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : paddle.Tensor
            Input tensor
        relative_positions : paddle.Tensor, optional
            Relative positions for RoPE

        Returns
        -------
        paddle.Tensor
            Output tensor
        """
        # Attention with residual
        if self.use_attn_norm:
            attn_input = self.attn_norm(x)
        else:
            attn_input = x
        x = x + self.attn(attn_input, relative_positions)

        # FFN with residual
        if self.use_ffn_norm:
            ffn_input = self.ffn_norm(x)
        else:
            ffn_input = x
        x = x + self.ffn(ffn_input)

        return x


class Transformer(nn.Layer):
    """
    Transformer processor for GAOT.

    Parameters
    ----------
    input_size : int
        Input dimension (patch_volume * node_latent_size)
    output_size : int
        Output dimension
    config : TransformerConfig
        Transformer configuration
    """

    def __init__(self, input_size: int, output_size: int, config: TransformerConfig):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = config.num_layers

        # Transformer blocks
        self.blocks = nn.LayerList(
            [
                TransformerBlock(
                    input_size=input_size,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    ffn_multiplier=config.ffn_multiplier,
                    use_attn_norm=config.use_attn_norm,
                    use_ffn_norm=config.use_ffn_norm,
                    norm_eps=config.norm_eps,
                    dropout=config.atten_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        x: paddle.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : paddle.Tensor [batch, num_patches, input_size]
            Input tensor
        condition : float, optional
            Conditioning value (not used currently)
        relative_positions : paddle.Tensor, optional
            Relative positions for RoPE

        Returns
        -------
        paddle.Tensor [batch, num_patches, output_size]
            Output tensor
        """
        for block in self.blocks:
            x = block(x, relative_positions)

        return x
