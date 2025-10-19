# psc_NN(Machine Learning for Perovskite Solar Cells: An Open-Source Pipeline)

!!! note "注意事项"

    1. 开始训练前，请确保数据集已正确放置在 `data/cleaned/` 目录下。
    2. 训练和评估需要安装额外的依赖包，请使用 `pip install -r requirements.txt` 安装。
    3. 为获得最佳性能，建议使用 GPU 进行训练。

=== "模型训练命令"

    ``` sh
    python psc_nn.py mode=train
    ```

=== "模型评估命令"

    ``` sh
    # 使用本地预训练模型
    python psc_nn.py mode=eval eval.pretrained_model_path="Your pdparams path"
    ```

    ``` sh
    # 或使用远程预训练模型
    python psc_nn.py mode=eval eval.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/PerovskiteSolarCells/solar_cell_pretrained.pdparams"
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [solar_cell_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/PerovskiteSolarCells/solar_cell_pretrained.pdparams) | RMSE: 3.91798 |

## 1. 背景简介

太阳能电池是一种通过光电效应将光能直接转换为电能的关键能源器件，其性能预测是优化和设计太阳能电池的重要环节。然而，传统的性能预测方法往往依赖于复杂的物理模拟和大量的实验测试，不仅成本高昂，且耗时较长，制约了研究与开发的效率。

近年来，深度学习和机器学习技术的快速发展，为太阳能电池性能预测提供了创新的方法。通过机器学习技术，可以显著加快开发速度，同时实现与实验结果相当的预测精度。特别是在钙钛矿太阳能电池研究中，材料的化学组成和结构多样性为模型训练带来了新的挑战。为了解决这一问题，研究者们通常将材料的特性转换为固定长度的特征向量，以适配机器学习模型。尽管如此，不同性能指标的特征表示设计仍需不断优化，同时对模型预测结果的可解释性要求也更为严格。

本研究中，通过利用包含钙钛矿太阳能电池特性信息的全面数据库（PDP），我们构建并评估了包括 XGBoost、psc_nn 在内的多种机器学习模型，专注于预测短路电流密度（Jsc）。研究结果表明，结合深度学习与超参数优化工具（如 Optuna）能够显著提升太阳能电池设计的效率，为新型太阳能电池研发提供了更精确且高效的解决方案。

## 2. 模型原理

本章节仅对太阳能电池性能预测模型的原理进行简单地介绍，详细的理论推导请阅读 [Machine Learning for Perovskite Solar Cells: An Open-Source Pipeline](https://onlinelibrary.wiley.com/doi/10.1002/apxr.202400060)。

该方法的主要思想是通过人工神经网络建立光谱响应数据与短路电流密度（Jsc）之间的非线性映射关系。人工神经网络模型的总体结构如下图所示：

![psc_nn_overview](https://paddle-org.bj.bcebos.com/paddlescience/docs/psc_nn/psc_nn_overview.png)

本案例采用多层感知机（MLP）作为基础模型架构，主要包括以下几个部分：

1. 输入层：接收 2808 维的光谱响应数据
2. 隐藏层：4-6 层全连接层，每层的神经元数量通过 Optuna 优化
3. 激活函数：使用 ReLU 激活函数引入非线性特性
4. 输出层：输出预测的 Jsc 值

通过这种方式，我们可以自动找到最适合当前任务的模型配置，提高模型的预测性能。

## 3. 模型实现

本章节我们讲解如何基于 PaddleScience 代码实现钙钛矿太阳能电池性能预测模型。本案例结合 Optuna 框架进行超参数优化，并使用 PaddleScience 内置的各种功能模块。为了快速理解 PaddleScience，接下来仅对模型构建、约束构建、评估器构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

本案例使用的数据集包含 [Perovskite Database Project(PDP) 数据](https://paddle-org.bj.bcebos.com/paddlescience%2Fdatasets%2Fpsc%2Fdata.zip)。数据集分为以下几个部分：

1. 训练集：
   - 特征数据：`data/cleaned/training.csv`
   - 标签数据：`data/cleaned/training_labels.csv`
2. 验证集：
   - 特征数据：`data/cleaned/validation.csv`
   - 标签数据：`data/cleaned/validation_labels.csv`

为了方便数据处理，我们实现了一个辅助函数 `create_tensor_dict` 来创建输入和标签的 tensor 字典：

``` py linenums="84" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:84:89
--8<--
```

数据集的读取和预处理代码如下：

``` py linenums="172" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:172:191
--8<--
```

为了进行超参数优化，我们将训练集进一步划分为训练集和验证集：

``` py linenums="185" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:185:187
--8<--
```

### 3.2 模型构建

本案例使用 PaddleScience 内置的 `ppsci.arch.MLP` 构建多层感知机模型。模型的超参数通过 Optuna 框架进行优化，主要包括：

1. 网络层数：4-6层
2. 每层神经元数量：10-input_dim/2
3. 激活函数：ReLU
4. 输入维度：2808（光谱响应数据维度）
5. 输出维度：1（Jsc 预测值）

模型定义代码如下：

``` py linenums="152" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:152:168
--8<--
```

### 3.3 损失函数设计

考虑到数据集中不同样本的重要性可能不同，我们设计了一个加权均方误差损失函数。该函数对较大的 Jsc 值赋予更高的权重，以提高模型在高性能太阳能电池上的预测准确性：

``` py linenums="72" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:72:81
--8<--
```

### 3.4 约束构建

本案例基于数据驱动的方法求解问题，因此使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。为了减少代码重复，我们实现了 `create_constraint` 函数来创建监督约束：

``` py linenums="92" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:92:111
--8<--
```

### 3.5 评估器构建

为了实时监测模型的训练情况，我们实现了 `create_validator` 函数来创建评估器：

``` py linenums="114" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:114:129
--8<--
```

### 3.6 优化器构建

为了统一管理优化器和学习率调度器的创建，我们实现了 `create_optimizer` 函数：

``` py linenums="132" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:132:149
--8<--
```

### 3.7 模型训练与评估

在训练过程中，我们使用上述封装的函数来创建数据字典、约束、评估器和优化器：

``` py linenums="258" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:258:262
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py
--8<--
```

## 5. 参考文献

- [Machine Learning for Perovskite Solar Cells: An Open-Source Pipeline](https://onlinelibrary.wiley.com/doi/10.1002/apxr.202400060)
