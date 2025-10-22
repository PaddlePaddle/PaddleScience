# XRDMatch

## 概述

XRDMatch 是一个基于 PaddleScience 的 XRD 数据半监督学习示例，使用 FlexMatch 算法进行材料分类。该示例展示了如何使用少量有标签数据和大量无标签数据来训练高性能的分类模型，特别适用于材料科学中的 XRD 谱线分析。本工作目的是利用锂离子固态电解质材料的XRD数据训练，得到相应的结构和性能关系。

## 背景

X射线衍射（XRD）是材料科学中重要的表征技术，能够提供材料的晶体结构信息。在实际应用中，获取大量有标签的 XRD 数据成本高昂且耗时，而半监督学习可以充分利用大量无标签数据来提升模型性能，降低标注成本。

## 方法

本示例采用 FlexMatch 算法，结合以下核心技术：

- **数据增强**：对无标签数据进行弱增强和强增强，提高模型泛化能力
- **伪标签生成**：基于模型预测生成伪标签，扩展训练数据
- **动态阈值**：根据类别置信度动态调整选择阈值，平衡各类别样本
- **一致性正则化**：确保模型对增强数据的一致性预测，提高鲁棒性

## 快速开始

### 环境要求

```bash
pip install paddlepaddle ppsci numpy pandas scikit-learn tqdm
```

### 数据准备

```bash
cd examples/ai4material/xrdmatch
# 确保 xrd_data/ 目录下包含 lbs.csv 和 ulbs.csv 文件
```

### 运行示例

```bash
# 快速演示（单次实验，适合测试）
python main.py --epochs 1 --batch_size 32

# 完整训练（100次实验，完整性能评估）
python main.py --epochs 100 --batch_size 32
```

### 配置参数

主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--num_labels` | 20 | 有标签数据数量 |
| `--lr` | 3e-4 | 学习率 |
| `--data_dir` | `./xrd_data` | 数据目录路径 |

## 数据格式

### 输入数据

- **`xrd_data/lbs.csv`**: 有标签数据
  - 包含样本名称、ID、标签和 XRD 谱线数据（4501维特征）
  - 标签：0（正类）、1（负类）

- **`xrd_data/ulbs.csv`**: 无标签数据
  - 包含样本名称、ID 和 XRD 谱线数据（4501维特征）
  - 无标签信息，用于半监督学习

### 数据预处理

1. **归一化**：将 XRD 强度值归一化到 [0,1] 范围
2. **噪声处理**：去除低强度噪声（阈值 < 0.1）
3. **数据增强**：
   - **弱增强**：添加少量噪声（10%）和位移（100像素）
   - **强增强**：缩放（15%）、消除（15%）、大幅位移（500像素）

## 模型架构

使用 VGG 网络作为特征提取器：

```python
model = ppsci.arch.VGG(in_channel=1, num_classes=2)
```

网络结构：
- 输入：1×4501 的 XRD 谱线数据
- 特征提取：VGG 卷积层
- 分类：2类分类（正类/负类）

## 训练策略

### FlexMatch 算法流程

1. **有标签数据训练**：使用交叉熵损失进行监督学习
2. **无标签数据处理**：
   - 生成弱增强和强增强版本
   - 基于弱增强版本生成伪标签
   - 使用强增强版本进行一致性训练
3. **动态阈值**：根据类别置信度动态调整选择阈值

### 损失函数

```python
total_loss = loss_lb + lambda_u * loss_ulb
```

其中：
- `loss_lb`: 有标签数据的交叉熵损失
- `loss_ulb`: 无标签数据的一致性损失
- `lambda_u`: 无标签损失权重（默认1.0）

### 训练配置

- **优化器**：AdamW (lr=3e-4, weight_decay=0.01)
- **学习率调度**：固定学习率
- **批次大小**：有标签32，无标签96
- **训练轮数**：100轮（每轮10个迭代）

## 评估指标

- **准确率 (Accuracy)**：正确分类的样本比例
- **精确率 (Precision)**：预测为正类中实际为正类的比例
- **召回率 (Recall)**：实际正类中被正确预测的比例
- **F1 分数 (F1-Score)**：精确率和召回率的调和平均
- **混淆矩阵 (Confusion Matrix)**：各类别预测结果的详细分布

## 结果示例

### 训练日志示例

```
[2025-07-14 20:59:47 INFO] Starting experiment 1/100
[2025-07-14 20:59:47 INFO] unlabeled data number: 10000, labeled data number: 20
[2025-07-14 20:59:47 INFO] Epoch: 0
[2025-07-14 20:59:47,744 INFO] confusion matrix
[2025-07-14 20:59:47,744 INFO] [[0.77777778 0.22222222]
 [0.2        0.8       ]]
[2025-07-14 20:59:47,744 INFO] evaluation metric
[2025-07-14 20:59:47,745 INFO] acc: 0.7969
[2025-07-14 20:59:47,745 INFO] precision: 0.6727
[2025-07-14 20:59:47,745 INFO] recall: 0.7889
[2025-07-14 20:59:47,746 INFO] f1: 0.6949
Best model saved at epoch 1, score: 0.6949028236156949

### 性能指标

在标准测试集上的典型性能：

| 指标 | 值 |
|------|-----|
| 准确率 | 0.797 |
| 精确率 | 0.673 |
| 召回率 | 0.789 |
| F1分数 | 0.695 |

## 文件结构

```
examples/ai4material/xrdmatch/
├── main.py                    # 主训练脚本
├── README.md                  # 使用说明
├── configs/
│   └── xrdmatch.yaml         # 配置文件
├── xrd_data/                 # 数据目录
│   ├── lbs.csv              # 有标签数据
│   └── ulbs.csv             # 无标签数据
├── saved_models_ppsci/       # 模型保存目录
│   └── exp_0/               # 实验0的模型文件
├── loss/                     # 损失函数模块
├── datasets/                 # 数据集模块
```

## 使用说明

### 1. 数据准备

确保 `xrd_data/` 目录下包含：
- `lbs.csv`: 有标签数据文件
- `ulbs.csv`: 无标签数据文件

### 2. 快速测试

```bash
# 单次实验，快速验证环境
python main.py --epochs 1
```

### 3. 完整训练

```bash
# 100次实验，完整性能评估
python main.py --epochs 100
```

### 4. 结果查看

训练完成后，查看以下文件：
- `diver.txt`: 训练过程记录
- `pred.txt`: 预测结果
- `saved_models_ppsci/`: 保存的模型文件

## 注意事项

1. **数据获取**：XRD 数据需要手动下载或联系作者获取
2. **计算资源**：完整训练需要较多计算资源，建议使用 GPU
3. **参数调优**：可根据具体数据调整超参数
4. **结果复现**：设置随机种子确保结果可复现
5. **内存要求**：大数据集可能需要较大内存，建议分批处理


## 参考文献

-Zheng Wan., et al. "XRDMatch: a semi-supervised learning framework to efficiently discover room temperature lithium superionic conductors." Energy Environ. Sci., 2024, 17, 9487(https://pubs.rsc.org/en/content/articlelanding/2024/ee/d4ee02970d)
