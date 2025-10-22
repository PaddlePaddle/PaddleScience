# Predicting the Strength of Composites

=== "模型训练命令"

    ``` sh
    python main.py mode=train
    ```

=== "模型评估命令"

    ``` sh
    python python main.py mode=eval
    ```

## 下载预训练模型

| [resnet18-v5-fold1](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold2](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold3](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold4](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold5](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) ||

## 下载模型必要参数

| [Saved_Output](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/Saved_Output.tar.gz) |

## 背景简介

材料的极限抗拉强度（UTS）是衡量复合材料抗拉伸破坏的核心指标，直接决定其应用安全性与可靠性。它是结构设计的关键依据，确保构件在拉伸载荷下不失效；也是材料选型的重要标准，匹配不同场景的强度需求，最终保障复合材料制品的性能上限。但由于复杂的形态-性能关系，预测其机械性能仍然较为困难，使用传统机器学习方法很难对其做出有效的预测。

针对材料科学领域中材料结构强度预测这一问题，通过X射线CT图像预测聚合物-陶瓷复合材料的极限抗拉强度（UTS）。相较于传统材料强度预测方法对于数据和模型的需求严苛，且需要耗费较长的时间成本，本项目通过深度学习技术，在小样本数据集的条件下，实现了较高精度的UTS值预测，提供了更快速且准确的工具。帮助研究人员快速了解材料的特性，并优化材料设计

本研究中使用卷积神经网络（CNN） 来分析冷烧结聚合物-陶瓷复合材料的 X 射线计算机断层扫描 （CT） 图像来应对这一问题。以形态特征作为输入的传统机器学习模型产生的准确性有限，而使用预训练的卷积神经网络，并使用集成学习进一步优化了模型。使用小型数据集来揭示复合材料中形态-结构-性能关系的替代机器学习方法，为衡量复合材料的性能提供了更精确且高效的解决方案。

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

## 2. 模型原理

本章节对基于卷积神经网络的材料拉伸强度预测模型的原理进行介绍。

该方法的主要思想是通过卷积神经网络建立材料微观结构图像与拉伸强度（UTS）之间的非线性映射关系。模型采用ResNet架构，能够有效提取图像中的深层特征信息。

本案例采用ResNet-18作为基础模型架构，主要包括以下几个部分：

1. 输入层：接收 224×224×3 的RGB图像数据
2. 卷积层：多个卷积块，包含残差连接
3. 池化层：最大池化操作，降低特征图尺寸
4. 全连接层：将特征映射到最终的预测值
5. 输出层：输出预测的UTS值（MPa）

通过这种方式，我们可以自动学习材料微观结构图像中的关键特征，建立图像与性能之间的映射关系，实现准确的拉伸强度预测。

## 3. 模型实现

本章节我们讲解如何基于 PaddleScience 代码实现材料拉伸强度预测模型。本案例使用5折交叉验证进行模型训练和评估，并使用 PaddleScience 内置的各种功能模块。

### 3.1 数据格式说明

数据集下载链接:<https://paddle-org.bj.bcebos.com/paddlescience/datasets/CNN_UTS/Dataset.zip>

| Image Name         | ...特征列... | UTS (MPa) | ... |
|--------------------|--------------|-----------|-----|
| IPP_10__40060.jpg  | ...          | 0.56      | ... |
| ...                | ...          | ...       | ... |

本案例使用的数据集包含材料微观结构图像和对应的拉伸强度标签。数据集分为以下几个部分：

1. 训练集：`Dataset/Train_val/`
2. 测试集：`Dataset/Test/`

数据集结构如下：

- 每个样本包含RGB图像和对应的UTS标签
- 图像经过预处理，统一调整为224×224尺寸
- 使用ImageNet预训练权重的标准化参数进行归一化

为了方便数据处理，我们使用了 `make_dataset` 函数来创建数据集：

``` py linenums="63" title="examples/CNN_UTS/data_utils.py"
--8<--
examples/CNN_UTS/data_utils.py:63:155
--8<--
```

### 3.2 模型构建

本案例使用 PaddleScience 内置的 `ppsci.arch.ResNet` 构建ResNet-18模型。模型的主要参数包括：

1. 网络结构：ResNet-18 (2,2,2,2)
2. 输入通道：3（RGB图像）
3. 输出维度：1（UTS预测值）
4. 预训练权重：ImageNet

模型定义代码如下：

``` py linenums="84" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:84:88
--8<--
```

### 3.3 数据增强

为了提高模型的泛化能力，我们实现了多种数据增强策略：

1. 随机水平翻转
2. 随机垂直翻转
3. 中心裁剪到224×224
4. 标准化处理

数据增强配置如下：

``` py linenums="28" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:28:45
--8<--
```

### 3.4 训练策略

本案例采用5折交叉验证策略进行模型训练：

1. 将训练数据分为5个fold
2. 每个fold训练一个独立的模型
3. 最终使用所有fold的预测结果进行集成

训练过程包括：

``` py linenums="60" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:60:93
--8<--
```

### 3.5 损失函数和优化器

使用均方误差损失函数进行回归任务：

``` py linenums="90" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:90:93
--8<--
```

使用Adam优化器进行参数更新：

``` py linenums="91" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:91:93
--8<--
```

### 3.6 模型评估

评估过程包括：

1. 计算MSE和R²指标
2. 生成parity plot和violin plot
3. 进行集成预测

评估器构建代码如下：

``` py linenums="191" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:191:216
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py
--8<--
```

## 参考文献

- [Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets](<https://pubs.acs.org/doi/10.1021/acsmaterialslett.4c02424>)
