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
GAOT核心层包初始化 - 导出所有核心组件
输入: 无 | 输出: 导出所有模块 | 地位: 包入口
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from .agno import AGNO
from .attn import MultiHeadAttention
from .attn import Transformer
from .attn import TransformerBlock
from .attn import TransformerConfig
from .gaot import GAOT
from .gaot import GAOTConfig
from .gemb import GeometricEmbedding
from .gemb import node_pos_encode
from .magno import MAGNOConfig
from .magno import MAGNODecoder
from .magno import MAGNOEncoder
from .mlp import ChannelMLP
from .mlp import LinearChannelMLP
from .utils.neighbor_search import NeighborSearch
from .utils.scatter import scatter_add
from .utils.scatter import scatter_max
from .utils.scatter import scatter_mean
from .utils.scatter import scatter_sum
from .utils.scatter import segment_csr
from .utils.scatter import segment_softmax

__all__ = [
    # Scatter operations
    "scatter_add",
    "scatter_sum",
    "scatter_mean",
    "scatter_max",
    "segment_csr",
    "segment_softmax",
    # Neighbor search
    "NeighborSearch",
    # MLP
    "ChannelMLP",
    "LinearChannelMLP",
    # Geometric embedding
    "GeometricEmbedding",
    "node_pos_encode",
    # AGNO
    "AGNO",
    # MAGNO
    "MAGNOEncoder",
    "MAGNODecoder",
    "MAGNOConfig",
    # Transformer
    "Transformer",
    "TransformerConfig",
    "MultiHeadAttention",
    "TransformerBlock",
    # Complete GAOT model
    "GAOT",
    "GAOTConfig",
]
