# Predicting the Strength of Composites

## 参考
Po-Hao Lai, et al. "Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets"
<https://doi.org/10.5281/zenodo.14803929>

## 目录结构
```
CNN_UTS/
│
├─ conf/  
│    └─ resnet.yaml
├─ data_utils.py  
├─ model_utils.py  
├─ main.py  
├─ requirements.txt  
├─ readme.md  
├─ resnet18-v5-finetune/  
├─ outputs/  
├─ Saved_Output/  
└─ Dataset/  
     ├─ Train_val/  
     └─ Test/  
```

## 环境依赖

见 requirements.txt

## 数据格式说明
数据集下载链接:https://zenodo.org/records/14803929

- `Dataset/Train_val/` 和 `Dataset/Test/` 下为若干子文件夹，每个子文件夹代表一个样本组。
- 每个子文件夹内包含若干 `.jpg` 图像和一个 `.csv` 文件。
- `.csv` 文件示例（每行对应一张图片，需包含 `Image Name`、若干特征列、`UTS (MPa)` 等标签）：

| Image Name         | ...特征列... | UTS (MPa) | ... |
|--------------------|--------------|-----------|-----|
| IPP_10__40060.jpg  | ...          | 0.56      | ... |
| ...                | ...          | ...       | ... |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `conf/resnet.yaml`，可自定义训练/评估参数：

```yaml
mode: "eval"
seed: 42
device: "cuda:0"
data:
  train_path: "./Dataset/Train_val"
  test_path: "./Dataset/Test"
  N: 1
train:
  epochs: 32
  n_splits: 5
  batch_size: 32
  lr: 0.0009761248347350309
output_dir: "./Saved_Output"
```

### 3. 训练模型

```bash
python main.py mode=train
```

### 4. 评估模型

```bash
python main.py mode=eval
```

### 5. 可视化与结果

- 训练和评估后，预测结果、统计指标、可视化图片会自动保存在 `Saved_Output/` 目录下。
。
