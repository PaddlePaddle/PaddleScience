# GAOT核心模型层

## 极简架构（3行）
1. **基础层（utils/）**: scatter操作和邻居搜索（2个模块）
2. **组件层**: MLP、几何嵌入、AGNO、MAGNO、Transformer（7个模块）
3. **集成层**: 完整GAOT模型和评估指标（2个模块）

## 文件清单

| 文件 | 地位 | 功能 | 依赖 |
|------|------|------|------|
| **utils/scatter.py** | 基础 | Scatter操作（替代torch_scatter） | paddle |
| **utils/neighbor_search.py** | 基础 | 邻居搜索（替代torch_cluster） | scipy |
| **mlp.py** | 组件 | MLP模块（ChannelMLP和LinearChannelMLP） | paddle |
| **gemb.py** | 组件 | 几何嵌入（statistical和pointnet方法） | paddle, mlp |
| **agno.py** | 组件 | AGNO图神经算子（核心消息传递） | paddle, mlp, scatter |
| **magno.py** | 组件 | MAGNO编解码器（多尺度图注意力）✅ v1.1.0修复 | paddle, agno, gemb |
| **attn.py** | 组件 | Patch Vision Transformer | paddle |
| **gaot.py** | 集成 | 完整GAOT模型 | magno, attn |
| **metrics.py** | 集成 | L1+median评估指标 | paddle |

## 架构层次

```
完整GAOT模型 (gaot.py)
    ↓
├─ MAGNO Encoder (magno.py)
│   ├─ AGNO (agno.py) → MLP + scatter
│   └─ GeometricEmbedding (gemb.py) → MLP
├─ Patch ViT (attn.py)
└─ MAGNO Decoder (magno.py)
    ├─ AGNO (agno.py)
    └─ GeometricEmbedding (gemb.py)

基础工具 (utils/)
├─ scatter.py: 图操作
└─ neighbor_search.py: 邻居查找
```

## 最近更新

### 2025-12-23 (v1.1.0)
- ✅ **修复projection层维度问题**（magno.py第577-584行）
  - **问题**: MAGNODecoder的projection层遇到维度不匹配
  - **修复**: 移除decoder中不必要的transpose操作
  - **结果**: 前向传播测试通过，输出形状正确 [batch, nodes, output_dim]
  - **验证**: complete_validation_test.py全部测试通过

### 2025-12-22 (v1.0.0)
- ✅ AGNO和GeometricEmbedding接口修复完成
- ✅ 接口与PyTorch版本100%一致
- ✅ 添加Apache 2.0许可证头部
- ✅ 代码规范化（Black + isort + Ruff）

## 关键接口

### AGNO接口（与PyTorch一致）
```python
def forward(y, neighbors, x=None, f_y=None):
    """
    Args:
        y: [n, coord_dim] - 物理点坐标
        x: [m, coord_dim] - 查询点坐标
        f_y: [batch, n, in_channels] - 输入特征
        neighbors: Dict - 邻居信息
    Returns:
        [batch, m, out_channels] - 输出特征
    """
```

### GeometricEmbedding接口（与PyTorch一致）
```python
def forward(input_geom, latent_queries, spatial_nbrs):
    """
    Args:
        input_geom: [n, coord_dim] - 输入点坐标
        latent_queries: [m, coord_dim] - 查询点坐标
        spatial_nbrs: Dict - 邻居信息
    Returns:
        [m, output_dim] - 几何嵌入特征
    """
```

## 最近更新

### 2025-12-23
- ✅ **修复projection层维度问题**（magno.py）
  - 移除MAGNODecoder中不必要的transpose操作
  - 前向传播测试通过
  - 输出形状验证正确 [batch, nodes, output_dim]

### 2025-12-22
- ✅ AGNO和GeometricEmbedding接口修复完成
- ✅ 接口与PyTorch版本100%一致

## 维护规则

⚠️ **重要**: 一旦本文件所属目录有变化，应当立即更新本文档

- 新增模块时更新文件清单
- 修改接口时更新关键接口说明
- 调整架构时更新架构层次图
- **重要修复需在"最近更新"章节记录**