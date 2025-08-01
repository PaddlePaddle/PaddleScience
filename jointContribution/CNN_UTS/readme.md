# PaddlePaddle-ResNet18 图像回归项目

本项目基于 Paddle框架复现的材料微观结构强度预测项目，通过 X 射线 CT 图像预测聚合物 - 陶瓷复合材料的极限抗拉强度（UTS），实现了论文《Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets》中的核心方法，实现了图像的回归预测，支持ResNet18、五折交叉验证、超参数配置、自动训练与评估、可视化等功能

## 目录结构
```
CNN_UTS/
│
├─ conf/                  # 配置文件
│    └─ resnet.yaml
├─ data_utils.py          # 数据集加载与处理工具
├─ model_utils.py         # 模型相关工具（如随机种子设置）
├─ main.py                # 主程序，包含训练与评估流程
├─ requirements.txt       # 依赖包列表
├─ readme.md              # 项目说明文档
├─ resnet18-v5-finetune/  # 各折训练得到的模型参数
├─ outputs/               # 日志与输出目录
├─ Saved_Output/          # 保存的预测结果与可视化图片
└─ Dataset/               # 数据集目录
     ├─ Train_val/        # 训练/验证集
     └─ Test/             # 测试集
```

## 环境依赖

见 requirements.txt

## 数据格式说明

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
- 支持parity plot、小提琴图等多种可视化。

## 主要功能

- 支持ResNet18的回归任务
- 五折交叉验证与模型集成
- 配置化超参数与数据路径
- 自动保存/加载模型与预测结果
- 多种可视化与统计指标输出
