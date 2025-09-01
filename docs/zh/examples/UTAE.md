# 卫星时间序列影像分割：U-TAE

## 案例简介
本案例实现了 **U-TAE (Unsupervised Temporal Attention Encoder)** 模型在 **PASTIS 数据集** 上的语义分割与全景分割任务。  
该模型最早发表于 *ICCV 2021*（Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks）。  
我们将原始 PyTorch 实现迁移至 **PaddlePaddle**，并提供完整的训练与测试脚本。  

## 数据集
- **PASTIS**：一个专为遥感时间序列影像语义分割设计的数据集，包含 Sentinel-2 多时相影像及对应的地块标注。  
- 任务类型：
  - **语义分割 (Semantic Segmentation)**
  - **全景分割 (Panoptic Segmentation)**

数据集可通过 [PASTIS 官网](https://paperswithcode.com/dataset/pastis) 下载。  

## 模型结构
- **U-TAE Backbone**：采用卷积 + 时间注意力机制对卫星影像时序特征进行建模。  
- **ConvLSTM & LTAE**：用于时序特征编码与解码。  
- **Panoptic Head**：在语义分割结果上进一步进行全景分割。  

迁移实现包括以下核心组件：
- `src/backbones/`：网络骨干（ConvLSTM, LTAE, Positional Encoding, UTAE）  
- `src/learning/`：训练过程相关（loss、miou、权重初始化）  
- `src/panoptic/`：全景分割模块（PaPs, loss, FocalLoss, metrics, utils）  
- 训练/测试脚本：`train_semantic.py`、`test_semantic.py`、`train_panoptic.py`、`test_panoptic.py`

## 使用方法

### 语义分割任务
训练：
```bash
python train_semantic.py --config configs/utae_semantic.yaml
```

测试：
```bash
python test_semantic.py --config configs/utae_semantic.yaml --weights output/utae_semantic/best_model.pdparams
```

### 全景分割任务
训练：
```bash
python train_panoptic.py --config configs/utae_panoptic.yaml
```

测试：
```bash
python test_panoptic.py --config configs/utae_panoptic.yaml --weights output/utae_panoptic/best_model.pdparams
```

## 实验结果
在 PASTIS 数据集上，本案例复现了以下性能（PaddlePaddle 实现）：  

- **SQ (Segmentation Quality)**: 83.8  
- **RQ (Recognition Quality)**: 58.9  
- **PQ (Panoptic Quality)**: 49.7  

对比 PyTorch 原版实现，性能相近，说明迁移有效。

## 文件结构
```
UTAE
 ├── src
 │   ├── backbones/              # 模型骨干
 │   ├── learning/               # 训练相关工具
 │   ├── panoptic/               # 全景分割模块
 │   ├── dataset.py              # 数据加载
 │   ├── model_utils.py          # 模型工具函数
 │   └── utils.py                # 通用工具
 ├── train_semantic.py           # 语义分割训练
 ├── test_semantic.py            # 语义分割测试
 ├── train_panoptic.py           # 全景分割训练
 └── test_panoptic.py            # 全景分割测试
```
