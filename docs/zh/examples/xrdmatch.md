# XRDMatch

## 1.模型训练与评估
=== "模型训练命令"
``` sh
    python main.py
```
=== "模型评估命令"
``` sh
    python main.py --mode eval --exp_id x --epoch x
```

## 2.背景简介

XRDMatch 是一个基于 PaddleScience 的 XRD 数据半监督学习示例，使用 FlexMatch 算法进行材料分类。该示例展示了如何使用少量有标签数据和大量无标签数据来训练高性能的分类模型，特别适用于材料科学中的 XRD 谱线分析。

X射线衍射（XRD）是材料科学中重要的表征技术，能够提供材料的晶体结构信息。在实际应用中，获取大量有标签的 XRD 数据成本高昂且耗时，而半监督学习可以充分利用大量无标签数据来提升模型性能，降低标注成本。

本工作目的是利用锂离子固态电解质材料的XRD数据训练，得到相应的结构和性能关系。通过FlexMatch算法，结合数据增强、伪标签生成、动态阈值和一致性正则化等技术，实现高效的半监督学习。


## 3.模型原理

该方法的主要思想是通过卷积神经网络建立XRD谱线数据与材料性能之间的非线性映射关系。模型采用VGG网络作为特征提取器，结合FlexMatch半监督学习算法，能够有效利用大量无标签数据提升模型性能。

本案例采用VGG网络作为基础模型架构，主要包括以下几个部分：

1. 输入层：接收 1×4501 的 XRD 谱线数据
2. 卷积层：多层卷积块，提取局部特征模式
3. 池化层：降维和特征聚合
4. 全连接层：特征映射到分类结果
5. 输出层：2类分类（正类/负类）

通过FlexMatch算法，模型能够：
- 基于弱增强数据生成伪标签
- 使用强增强数据进行一致性训练
- 动态调整选择阈值，平衡各类别样本

### 3.1数据格式说明

数据集包含材料XRD谱线数据和对应的性能标签：
- **数据连接**:
```
https://paddle-org.bj.bcebos.com/paddlescience/datasets/xrdmatch/lbs.csv
https://paddle-org.bj.bcebos.com/paddlescience/datasets/xrdmatch/ulbs.csv
```
- **`xrd_data/lbs.csv`**: 有标签数据
  - 包含样本名称、ID、标签和 XRD 谱线数据（4501维特征）
  - 标签：0（正类）、1（负类）

- **`xrd_data/ulbs.csv`**: 无标签数据
  - 包含样本名称、ID 和 XRD 谱线数据（4501维特征）
  - 无标签信息，用于半监督学习

### 3.2数据预处理与增强策略

1. **归一化**：将 XRD 强度值归一化到 [0,1] 范围
2. **噪声处理**：去除低强度噪声（阈值 < 0.1）
3. **数据增强**：
   - **弱增强**：添加少量噪声（10%）和位移（100像素）
   - **强增强**：缩放（15%）、消除（15%）、大幅位移（200像素）和噪声（20%）
``` py linenums="42" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:42:89
--8<--
```
### 3.3自定义数据集类
``` py linenums="201" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:201:239
--8<--
```
### 3.4 FlexMatch 半监督损失函数

1. **有标签数据训练**：使用交叉熵损失进行监督学习
2. **无标签数据处理**：
   - 生成弱增强和强增强版本
   - 基于弱增强版本生成伪标签
   - 使用强增强版本进行一致性训练
3. **动态阈值**：根据类别置信度动态调整选择阈值
``` py linenums="242" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:242:327
--8<--
```

### 3.5 损失函数
```python
total_loss = loss_lb + lambda_u * loss_ulb
```

其中：
- `loss_lb`: 有标签数据的交叉熵损失
- `loss_ulb`: 无标签数据的一致性损失
- `lambda_u`: 无标签损失权重（默认1.0）

### 3.6 训练配置

- **优化器**：AdamW (lr=3e-4, weight_decay=0.01)
- **批次大小**：有标签32，无标签96
- **实验次数**：100次独立实验
- **训练轮数**：每个实验100轮（每轮10个迭代）
- **数据划分**：正类前20个，负类前75个用于训练
- **模型保存**：仅当F1分数≥0.7时保存模型

## 3.7 评估指标

- **准确率 (Accuracy)**：正确分类的样本比例
- **精确率 (Precision)**：预测为正类中实际为正类的比例
- **召回率 (Recall)**：实际正类中被正确预测的比例
- **F1 分数 (F1-Score)**：精确率和召回率的调和平均
- **混淆矩阵 (Confusion Matrix)**：各类别预测结果的详细分布
- **评估方法**：支持训练时评估和独立评估两种模式
- **训练时评估**：在训练过程中自动调用，将日志保存到每个实验的 saved_models_ppsci/exp_*/log.txt 文件中
- **独立评估**：使用 `--mode eval` 参数对已保存的模型进行评估，结果保存到 eval_log.txt 文件中
- **模型保存策略**：仅当F1分数≥0.7时保存模型
- **内置评估实现**：该函数会在训练过程中自动调用，并将日志保存到每个实验的 saved_models_ppsci/exp_*/log.txt 文件中。代码实现：
``` py linenums="412" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:412:457
--8<--
```

## 4.结果示例

### 训练日志示例

```
Epoch: 0
[2025-8-27 02:40:12,747 INFO] confusion matrix
[2025-8-27 02:40:12,748 INFO] [[0.22222222 0.77777778]
 [0.2        0.8       ]]
[2025-8-27 02:40:12,748 INFO] evaluation metric
[2025-8-27 02:40:12,748 INFO] acc: 0.7188
[2025-8-27 02:40:12,748 INFO] precision: 0.5083
[2025-8-27 02:40:12,750 INFO] recall: 0.5111
[2025-8-27 02:40:12,750 INFO] f1: 0.5060
F1 score 0.5060 < 0.7, model not saved at epoch 0
```

### 性能指标

在标准测试集上的典型性能：

| 指标 | 值 |
|------|-----|
| 准确率 | 0.797 |
| 精确率 | 0.673 |
| 召回率 | 0.789 |
| F1分数 | 0.695 |

### 评估日志示例

```
Evaluating experiment 0 epoch 11 model...
Starting prediction...
confusion matrix
[[0.77777778 0.22222222]
 [0.16363636 0.83636364]]
evaluation metric
acc: 0.6480
precision: 0.6480
recall: 0.6480
f1: 0.6480
```

## 5.完整代码
``` py linenums="1" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py
--8<--
```

## 参考文献

Zheng Wan., et al.** "XRDMatch: a semi-supervised learning framework to efficiently discover room temperature lithium superionic conductors." *Energy Environ. Sci.*, 2024, 17, 9487. (https://pubs.rsc.org/en/content/articlelanding/2024/ee/d4ee02970d)
