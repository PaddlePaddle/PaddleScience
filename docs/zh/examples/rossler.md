# Rossler System

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6209280?sUid=455441&shared=1&ts=1684495132419" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 --output ./datasets/rossler_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 --output ./datasets/rossler_valid.hdf5
    python train_enn.py
    python train_transformer.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 --output ./datasets/rossler_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 --output ./datasets/rossler_valid.hdf5
    python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams
    python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams
    ```

| 模型 | MSE |
| :-- | :-- |
| [rossler_transformer_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_transformer_pretrained.pdparams) | 0.022 |

## 1. 背景简介

Rossler System，最早由德国科学家 Rossler 提出，也是常见的混沌系统。该系统在混沌理论的研究中具有重要地位，为混沌现象提供了一种数学描述和理解方法。同时由于该系统对数值扰动极为敏感，因此也是是评估机器学习（深度学习）模型准确性的良好基准。

## 2. 问题定义

Rossler 系统的状态方程：

$$
\begin{cases}
  \dfrac{\partial x}{\partial t} = -\omega y - z, & \\
  \dfrac{\partial y}{\partial t} = \omega x + \alpha y, & \\
  \dfrac{\partial z}{\partial t} = \beta + z(x - \gamma)
\end{cases}
$$

当参数取以下值时，系统表现出经典的混沌特性：

$$\omega = 1.0, \alpha = 0.165, \beta = 0.2, \gamma = 10$$

在这个案例中，要求给定初始时刻点的坐标，预测未来一段时间内点的运动轨迹。

## 3. 问题求解

接下来开始讲解如何基于 PaddleScience 代码，用深度学习的方法求解该问题。本案例基于论文 [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957) 方法进行求解，关于该方法的理论部分请参考[此文档](lorenz.md#31)或[原论文](https://arxiv.org/abs/2010.03957)。接下来首先会对使用的数据集进行介绍，然后对该方法两个训练步骤（Embedding 模型训练、Transformer 模型训练）的监督约束构建、模型构建等进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集采用了 [Transformer-Physx](https://github.com/zabaras/transformer-physx) 中提供的数据。该数据集使用龙格－库塔（Runge-Kutta）传统数值求解方法得到，数据集的划分如下：

|数据集 |时间序列的数量|时间步的数量|下载地址|
|:----:|:---------:|:--------:|:--------:|
|训练集 |256        |1025      |[rossler_training.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5)|
|验证集 |32         |1025      |[rossler_valid.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5)|

数据集官网为：<https://zenodo.org/record/5148524#.ZDe77-xByrc>

### 3.2 Embedding 模型

首先展示代码中定义的各个参数变量，每个参数的具体含义会在下面使用到时进行解释。

``` yaml linenums="22" title="examples/rossler/conf/enn.yaml"
--8<--
examples/rossler/conf/enn.yaml:22:34
--8<--
```

#### 3.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数，代码如下：

``` py linenums="55" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:55:74
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `RosslerDataset`，另外还指定了该类初始化时参数的取值：

1. `file_path`：代表训练数据集的文件路径，指定为变量 `train_file_path` 的值；
2. `input_keys`：代表模型输入数据的变量名称，此处填入变量 `input_keys`；
3. `label_keys`：代表真实标签的变量名称，此处填入变量 `output_keys`；
4. `block_size`：代表使用多长的时间步进行训练，指定为变量 `train_block_size` 的值；
5. `stride`：代表连续的两个训练样本之间的时间步间隔，指定为16；
6. `weight_dict`：代表模型输出各个变量与真实标签损失函数的权重，此处使用 `output_keys`、`weights` 生成。

"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，另外还指定了该类初始化时参数 `drop_last`、`shuffle` 均为 `True`。

`train_dataloader_cfg` 还定义了 `batch_size`、`num_workers` 的值。

定义监督约束的代码如下：

``` py linenums="76" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:76:86
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用带有 L2Decay 的 MSELoss，类名为 `MSELossWithL2Decay`，`regularization_dict` 设置了正则化的变量名称和对应的权重；

第三个参数表示在训练时如何计算需要被约束的中间变量，此处我们约束的变量就是网络的输出；

第四个参数是约束条件的名字，方便后续对其索引。此处命名为 "Sup"。

#### 3.2.2 模型构建

在该案例中，Embedding 模型的输入输出都是物理空间中点的位置坐标 $(x, y, z)$ ，使用了全连接层实现 Embedding 模型，如下图所示。

<figure markdown>
  ![rossler_embedding](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/rossler_embedding.png){ loading=lazy }
  <figcaption>Embedding 网络模型</figcaption>
</figure>

用 PaddleScience 代码表示如下：

``` py linenums="92" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:92:99
--8<--
```

其中，`RosslerEmbedding` 的前两个参数在前文中已有描述，这里不再赘述，网络模型的第三、四个参数是训练数据集的均值和方差，用于归一化输入数据。计算均值、方差的的代码表示如下：

``` py linenums="32" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:32:43
--8<--
```

#### 3.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `ExponentialDecay` ，学习率大小设置为0.001。优化器使用 `Adam`，梯度裁剪使用了 Paddle 内置的 `ClipGradByGlobalNorm` 方法。用 PaddleScience 代码表示如下

``` py linenums="101" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:101:110
--8<--
```

#### 3.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="114" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:114:141
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用 `ppsci.metric.MSE` 。

#### 3.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="143" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:143:157
--8<--
```

### 3.3 Transformer 模型

上文介绍了如何构建 Embedding 模型的训练、评估，在本节中将介绍如何使用训练好的 Embedding 模型训练 Transformer 模型。因为训练 Transformer 模型的步骤与训练 Embedding 模型的步骤基本相似，因此本节在两者的重复部分的各个参数不再详细介绍。首先将代码中定义的各个参数变量展示如下，每个参数的具体含义会在下面使用到时进行解释。

``` yaml linenums="23" title="examples/rossler/conf/transformer.yaml"
--8<--
examples/rossler/conf/transformer.yaml:23:34
--8<--
```

#### 3.3.1 约束构建

Transformer 模型同样基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数，代码如下：

``` py linenums="64" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:64:82
--8<--
```

数据加载的各个参数与 Embedding 模型中的基本一致，不再赘述。需要说明的是由于 Transformer 模型训练的输入数据是 Embedding 模型 Encoder 模块的输出数据，因此我们将训练好的 Embedding 模型作为 `RosslerDataset` 的一个参数，在初始化时首先将训练数据映射到编码空间。

定义监督约束的代码如下：

``` py linenums="84" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:84:89
--8<--
```

#### 3.3.2 模型构建

在该案例中，Transformer 模型的输入输出都是编码空间中的向量，使用的 Transformer 结构如下：

<figure markdown>
  ![rossler_transformer](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/rossler_transformer.png){ loading=lazy }
  <figcaption>Transformer 网络模型</figcaption>
</figure>

用 PaddleScience 代码表示如下：

``` py linenums="95" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:95:95
--8<--
```

类 `PhysformerGPT2` 除了需要填入 `input_keys`、`output_keys` 外，还需要设置 Transformer 模型的层数 `num_layers`、上下文的大小 `num_ctx`、输入的 Embedding 向量的长度 `embed_size`、多头注意力机制的参数 `num_heads`，在这里填入的数值为4、64、32、4。

#### 3.3.3 学习率与优化器构建

本案例中使用的学习率方法为 `CosineWarmRestarts`，学习率大小设置为0.001。优化器使用 `Adam`，梯度裁剪使用了 Paddle 内置的 `ClipGradByGlobalNorm` 方法。用 PaddleScience 代码表示如下：

``` py linenums="97" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:97:104
--8<--
```

#### 3.3.4 评估器构建

训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。用 PaddleScience 代码表示如下：

``` py linenums="107" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:107:132
--8<--
```

#### 3.3.5 可视化器构建

本案例中可以通过构建可视化器在模型评估时将评估结果可视化出来，由于 Transformer 模型的输出数据是预测的编码空间的数据无法直接进行可视化，因此需要额外将输出数据使用 Embedding 网络的 Decoder 模块变换到物理状态空间。

在本文中首先定义了对 Transformer 模型输出数据变换到物理状态空间的代码：

``` py linenums="34" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:34:52
--8<--
```

``` py linenums="63" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:63:64
--8<--
```

可以看到，程序首先载入了训练好的 Embedding 模型，然后在 `OutputTransform` 的 `__call__` 函数内实现了编码向量到物理状态空间的变换。

在定义好了以上代码之后，就可以实现可视化器代码的构建了：

``` py linenums="134" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:134:152
--8<--
```

首先使用上文中的 `mse_validator` 中的数据集进行可视化，另外还引入了 `vis_data_nums` 变量用于控制需要可视化样本的数量。最后通过 `VisualizerScatter3D` 构建可视化器。

#### 3.3.6 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="154" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:154:172
--8<--
```

## 4. 完整代码

``` py linenums="1" title="rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py
--8<--
```

``` py linenums="1" title="rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py
--8<--
```

## 5. 结果展示

下图中展示了两个不同初始条件下的模型预测结果和传统数值微分的预测结果。

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/result_states0.png){ loading=lazy }
  <figcaption>模型预测结果（"pred_states"）与传统数值微分结果（"states"）</figcaption>
</figure>

<figure markdown>
  ![result_states1](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/result_states1.png){ loading=lazy }
  <figcaption>模型预测结果（"pred_states"）与传统数值微分结果（"states"）</figcaption>
</figure>
