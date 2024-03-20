# Lorenz System

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6206798?contributionType=1&sUid=455441&shared=1&ts=1684477535039" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 --output ./datasets/lorenz_training_rk.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 --output ./datasets/lorenz_valid_rk.hdf5
    python train_enn.py
    python train_transformer.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 --output ./datasets/lorenz_training_rk.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 --output ./datasets/lorenz_valid_rk.hdf5
    python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
    python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python train_transformer.py mode=export EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
    ```

=== "模型推理命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 --output ./datasets/lorenz_training_rk.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 --output ./datasets/lorenz_valid_rk.hdf5
    python train_transformer.py mode=infer
    ```

| 模型 | MSE |
| :-- | :-- |
| [lorenz_transformer_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_transformer_pretrained.pdparams) | 0.054 |

## 1. 背景简介

Lorenz System，中文名称可译作“洛伦兹系统”，又称“洛伦兹混沌系统”，最早由美国气象学家爱德华·洛伦兹（Edward N.Lorenz）在1963年的一篇文章中提出。著名的“蝴蝶效应”，即“一只南美洲亚马逊河流域热带雨林中的蝴蝶，偶尔扇动几下翅膀，可以在两周以后引起美国得克萨斯州的一场龙卷风”，也是最早起源于这篇文章。洛伦兹系统的特点是在一定参数条件下展现出复杂、不确定的动态行为，包括对初始条件的敏感性和长期行为的不可预测性。这种混沌行为在自然界和许多实际应用领域中都存在，例如气候变化、股票市场波动等。洛伦兹系统对数值扰动极为敏感，是评估机器学习（深度学习）模型准确性的良好基准。

## 2. 问题定义

洛伦兹系统的状态方程：

$$
\begin{cases}
  \dfrac{\partial x}{\partial t} = \sigma(y - x), & \\
  \dfrac{\partial y}{\partial t} = x(\rho - z) - y, & \\
  \dfrac{\partial z}{\partial t} = xy - \beta z
\end{cases}
$$

当参数取以下值时，系统表现出经典的混沌特性：

$$\rho = 28, \sigma = 10, \beta = \frac{8}{3}$$

在这个案例中，要求给定初始时刻点的坐标，预测未来一段时间内点的运动轨迹。

## 3. 问题求解

接下来开始讲解如何基于 PaddleScience 代码，用深度学习的方法求解该问题。本案例基于论文 [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957) 方法进行求解，接下来首先会对该论文的理论方法进行简单介绍，然后对使用的数据集进行介绍，最后对该方法两个训练步骤（Embedding 模型训练、Transformer 模型训练）的监督约束构建、模型构建等进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 方法介绍

Transformer 结构在 NLP、CV 领域中取得了巨大的成功，但是其在建模物理系统方面还没有得到更多的探索。在 [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957) 这篇文章中，作者提出了基于 Transformer 的网络结构用于建模物理系统。实验结果表明，提出的方法能够准确的建模不同的动态系统，并且比其他传统的方法更好。

如下图所示，该方法主要包含两个网络模型：Embedding 模型和 Transformer 模型。其中，Embedding 模型的 Encoder 模块负责将物理状态变量进行编码映射为编码向量，Decoder 模块则负责将编码向量映射为物理状态变量；Transformer 模型作用于编码空间，其输入是 Embedding 模型 Encoder 模块的输出，利用当前时刻的编码向量预测下一时刻的编码向量，预测得到的编码向量可以被 Embedding 模型的 Decoder 模块解码，得到对应的物理状态变量。在模型训练时，首先训练 Embedding 模型，然后将 Embedding 模型的参数冻结训练 Transformer 模型。关于该方法的细节请参考论文 [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957)。

<figure markdown>
  ![trphysx-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/trphysx-arch.png){ loading=lazy }
  <figcaption>左：Embedding 网络结构，右：Transformer 网络结构</figcaption>
</figure>

### 3.2 数据集介绍

数据集采用了 [Transformer-Physx](https://github.com/zabaras/transformer-physx) 中提供的数据。该数据集使用龙格－库塔（Runge-Kutta）传统数值求解方法得到，每个时间步大小为0.01，初始位置从以下范围中随机选取：

$$x_{0} \sim(-20, 20), y_{0} \sim(-20, 20), z_{0} \sim(10, 40)$$

数据集的划分如下：

|数据集 |时间序列的数量|时间步的数量|下载地址|
|:----:|:---------:|:--------:|:--------:|
|训练集 |2048       |256       |[lorenz_training_rk.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5)|
|验证集 |64         |1024      |[lorenz_valid_rk.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5)|

数据集官网为：<https://zenodo.org/record/5148524#.ZDe77-xByrc>

### 3.3 Embedding 模型

首先展示代码中定义的各个参数变量，每个参数的具体含义会在下面使用到时进行解释。

``` yaml linenums="26" title="examples/conf/enn.yaml"
--8<--
examples/lorenz/conf/enn.yaml:26:34
--8<--
```

#### 3.3.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数，代码如下：

``` py linenums="51" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:51:70
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `LorenzDataset`，另外还指定了该类初始化时参数的取值：

1. `file_path`：代表训练数据集的文件路径，指定为变量 `train_file_path` 的值；
2. `input_keys`：代表模型输入数据的变量名称，此处填入变量 `input_keys`；
3. `label_keys`：代表真实标签的变量名称，此处填入变量 `output_keys`；
4. `block_size`：代表使用多长的时间步进行训练，指定为变量 `train_block_size` 的值；
5. `stride`：代表连续的两个训练样本之间的时间步间隔，指定为16；
6. `weight_dict`：代表模型输出各个变量与真实标签损失函数的权重，此处使用 `output_keys`、`weights` 生成。

"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，另外还指定了该类初始化时参数 `drop_last`、`shuffle` 均为 `True`。

`train_dataloader_cfg` 还定义了 `batch_size`、`num_workers` 的值。

定义监督约束的代码如下：

``` py linenums="72" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:72:85
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用带有 L2Decay 的 MSELoss，类名为 `MSELossWithL2Decay`，`regularization_dict` 设置了正则化的变量名称和对应的权重；

第三个参数表示在训练时如何计算需要被约束的中间变量，此处我们约束的变量就是网络的输出；

第四个参数是约束条件的名字，方便后续对其索引。此处命名为 "Sup"。

#### 3.3.2 模型构建

在该案例中，Embedding 模型的输入输出都是物理空间中点的位置坐标 $(x, y, z)$ ，使用了全连接层实现 Embedding 模型，如下图所示。

<figure markdown>
  ![lorenz_embedding](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/lorenz_embedding.png){ loading=lazy }
  <figcaption>Embedding 网络模型</figcaption>
</figure>

用 PaddleScience 代码表示如下：

``` py linenums="91" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:91:97
--8<--
```

其中，`LorenzEmbedding` 的前两个参数在前文中已有描述，这里不再赘述，网络模型的第三、四个参数是训练数据集的均值和方差，用于归一化输入数据。计算均值、方差的的代码表示如下：

``` py linenums="32" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:32:39
--8<--
```

#### 3.3.3 学习率与优化器构建

本案例中使用的学习率方法为 `ExponentialDecay` ，学习率大小设置为0.001。优化器使用 `Adam`，梯度裁剪使用了 Paddle 内置的 `ClipGradByGlobalNorm` 方法。用 PaddleScience 代码表示如下

``` py linenums="99" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:99:108
--8<--
```

#### 3.3.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="112" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:112:139
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用 `ppsci.metric.MSE` 。

#### 3.3.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="142" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:142:156
--8<--
```

### 3.4 Transformer 模型

上文介绍了如何构建 Embedding 模型的训练、评估，在本节中将介绍如何使用训练好的 Embedding 模型训练 Transformer 模型。因为训练 Transformer 模型的步骤与训练 Embedding 模型的步骤基本相似，因此本节在两者的重复部分的各个参数不再详细介绍。首先将代码中定义的各个参数变量展示如下，每个参数的具体含义会在下面使用到时进行解释。

``` yaml linenums="36" title="examples/lorenz/conf/transformer.yaml"
--8<--
examples/lorenz/conf/transformer.yaml:36:43
--8<--
```

#### 3.4.1 约束构建

Transformer 模型同样基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数，代码如下：

``` py linenums="68" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:68:85
--8<--
```

数据加载的各个参数与 Embedding 模型中的基本一致，不再赘述。需要说明的是由于 Transformer 模型训练的输入数据是 Embedding 模型 Encoder 模块的输出数据，因此我们将训练好的 Embedding 模型作为 `LorenzDataset` 的一个参数，在初始化时首先将训练数据映射到编码空间。

定义监督约束的代码如下：

``` py linenums="87" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:87:92
--8<--
```

#### 3.4.2 模型构建

在该案例中，Transformer 模型的输入输出都是编码空间中的向量，使用的 Transformer 结构如下：

<figure markdown>
  ![lorenz_transformer](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/lorenz_transformer.png){ loading=lazy }
  <figcaption>Transformer 网络模型</figcaption>
</figure>

用 PaddleScience 代码表示如下：

``` py linenums="98" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:98:98
--8<--
```

类 `PhysformerGPT2` 除了需要填入 `input_keys`、`output_keys` 外，还需要设置 Transformer 模型的层数 `num_layers`、上下文的大小 `num_ctx`、输入的 Embedding 向量的长度 `embed_size`、多头注意力机制的参数 `num_heads`，在这里填入的数值为4、64、32、4。

#### 3.4.3 学习率与优化器构建

本案例中使用的学习率方法为 `CosineWarmRestarts`，学习率大小设置为0.001。优化器使用 `Adam`，梯度裁剪使用了 Paddle 内置的 `ClipGradByGlobalNorm` 方法。用 PaddleScience 代码表示如下：

``` py linenums="101" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:101:107
--8<--
```

#### 3.4.4 评估器构建

训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。用 PaddleScience 代码表示如下：

``` py linenums="110" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:110:135
--8<--
```

#### 3.4.5 可视化器构建

本案例中可以通过构建可视化器在模型评估时将评估结果可视化出来，由于 Transformer 模型的输出数据是预测的编码空间的数据无法直接进行可视化，因此需要额外将输出数据使用 Embedding 网络的 Decoder 模块变换到物理状态空间。

在本文中首先定义了对 Transformer 模型输出数据变换到物理状态空间的代码：

``` py linenums="34" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:34:52
--8<--
```

``` py linenums="64" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:64:65
--8<--
```

可以看到，程序首先载入了训练好的 Embedding 模型，然后在 `OutputTransform` 的 `__call__` 函数内实现了编码向量到物理状态空间的变换。

在定义好了以上代码之后，就可以实现可视化器代码的构建了：

``` py linenums="138" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:138:155
--8<--
```

首先使用上文中的 `mse_validator` 中的数据集进行可视化，另外还引入了 `vis_data_nums` 变量用于控制需要可视化样本的数量。最后通过 `VisualizerScatter3D` 构建可视化器。

#### 3.4.6 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="157" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:157:175
--8<--
```

## 4. 完整代码

``` py linenums="1" title="lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py
--8<--
```

``` py linenums="1" title="lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py
--8<--
```

## 5. 结果展示

下图中展示了两个不同初始条件下的模型预测结果和传统数值微分的预测结果。

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/result_states0.png){ loading=lazy }
  <figcaption>模型预测结果（"pred_states"）与传统数值微分结果（"states"）</figcaption>
</figure>

<figure markdown>
  ![result_states1](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/result_states1.png){ loading=lazy }
  <figcaption>模型预测结果（"pred_states"）与传统数值微分结果（"states"）</figcaption>
</figure>
