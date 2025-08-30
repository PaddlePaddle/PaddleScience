
# Satellite Time-Series Panoptic Segmentation
基于 **ConvTemporalAttentionNet** 的 PaddleScience 官方示例。

## 数据集
- Sentinel-2 12 期影像  
- 下载地址：[Google Drive](https://drive.google.com/xxx)  
- 目录结构
  ```plaintext
  data/satellite_panoptic/
  ├── images/   # (T, C, H, W)
  └── labels/   # (T, H, W)
  ```

## 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练
python train.py

# 3. 验证
python validate.py --pretrained ./pretrained/model.pdparams
```

## 性能
| 指标 | 值 |
| ---- | --- |
| SQ   | 0.806 |
| RQ   | 0.494 |
| PQ   | 0.403 |

## 文件说明
| 文件 | 作用 |
| ---- | ---- |
| pretrained/model.pdparams | 训练好的权重 |
| src/model_paddle.py | ConvTemporalAttentionNet 网络 |
| src/dataset_paddle.py | 数据加载 |
| src/utils_paddle.py | 工具函数 |
| train.py | 训练脚本 |
| validate.py | 验证脚本 |
| conf/satellite_panoptic.yaml | 超参配置 |

## 参考
原始 PyTorch 实现：[GitHub Link](https://github.com/xxx)

---
