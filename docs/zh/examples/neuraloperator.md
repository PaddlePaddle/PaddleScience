# neuraloperator


=== "模型训练命令"

    ``` sh
    # sfno 预训练模型
    python examples/neuraloperator/train_sfno.py
    # tfno 预训练模型
    python examples/neuraloperator/train_tfno.py
    # uno 预训练模型
    python examples/neuraloperator/train_uno.py
    ```

=== "模型评估命令"

    ``` sh
    # sfno 模型评估
    python examples/neuraloperator/train_sfno.py mode=eval
    # tfno 模型评估
    python examples/neuraloperator/train_tfno.py mode=eval
    # uno 模型评估
    python examples/neuraloperator/train_uno.py mode=eval
    ```

=== "模型推理命令"

    ``` sh
    # sfno 模型推理
    python examples/neuraloperator/train_sfno.py mode=infer
    # tfno 模型推理
    python examples/neuraloperator/train_tfno.py mode=infer
    # uno 模型推理
    python examples/neuraloperator/train_uno.py mode=infer
    ```
| 模型 | 16_h1 | 16_l2 | 32_h1 | 32_l2 |
| :-- | :-- | :-- | :-- | :-- |
| [tfno 模型]() | 0.13113 | 0.08514 | 0.30353 | 0.12408

| 模型 | 16_h1 | 16_l2 | 32_h1 | 32_l2 |
| :-- | :-- | :-- | :-- | :-- |
| [uno 模型]() | 0.18360 | 0.11040 | 0.74840 | 0.60193

| 模型 | 32x64_l2 | 64x128_l2 |
| :-- | :-- | :-- |
| [sfno 模型]() | 1.01075 | 2.33481 |

## 1. 背景简介
许多科学和工程问题涉及反复求解复杂的偏微分方程 (PDE) 系统，以获取某些参数的不同值。例如分子动力学、微力学和湍流流动。通常这样的系统需要精细的离散化才能捕捉所模拟的现象。因此，传统数值求解器速度慢，有时效率低下。机器学习方法可能通过提供快速的求解器来革新科学领域，这些求解器可以近似或增强传统求解器。然而，经典神经网络在有限维空间之间进行映射，因此只能学习与特定离散化相关的解决方案。这通常是实际应用中的一个限制，因此需要开发与网格无关的神经网络。最近，一项新的工作提出了用神经网络学习无网格、无限维算子。神经算子通过产生一组用于不同离散化、且与网格无关的参数，来弥补有限维算子方法中网格依赖性的问题。  neuraloperator 通过直接在傅里叶空间 (Fourier space) 中参数化 (parameterize) 积分核 (integral kernel) 来制定一个新的神经算子，从而实现了富有表现力和高效的架构。论文对 Burgers 方程、Darcy 流和 Navier-Stokes 方程进行了实验。傅里叶神经算子是第一个基于机器学习的方法，成功地用零样本超分辨率模拟湍流。与传统 PDE 求解器相比，它快达三个数量级。
## 2. 模型原理
本章节仅对 NeuralOperator 的模型原理进行简单地介绍，详细的理论推导请阅读
[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)。
NeuralOperator 引入了傅里叶神经算子 (Fourier neural operator)，这是一种新颖的深度学习架构，能够学习函数之间无限维空间的映射；积分算子被限制为卷积，并通过傅里叶域中的线性变换实例化。傅里叶神经算子是第一个学习湍流状态下 Navier-Stokes 方程族的分辨率不变解算子的工作，其中以前基于图形的神经算子不收敛。该方法共享相同的学习网络参数，而不考虑输入和输出空间上使用的离散化。


模型的总体结构如图所示：

<figure markdown>
  ![NeuralOperator-arch](https://github.com/PaddlePaddle/PaddleScience/assets/71805205/299b9244-fbb6-4bdd-ab9b-a017034b2ef9){ loading=lazy style="margin:0 auto"}
  <figcaption>NeuralOperator 网络模型</figcaption>
</figure>

NeuralOperator 论文中使用 TFNO 和 UNO 模型训练 Darcy-Flow  数据集，并进行验证和推理；使用 SFNO 模型训练 Spherical Shallow Water(SWE) 数据集，并进行验证和推理。接下来分别进行介绍。

### 2.1 模型训练、推理过程

模型预训练阶段是基于随机初始化的网络权重对模型进行训练，如下图所示，其中 $X_[w,h]$ 表示大小为 $w*h$ 的二维偏微分数据，$Y_[w,h]$ 表示预测的大小为 $w*h$ 的二维偏微分方程数值解，$Y_{true[w,h]}$ 表示真实二维偏微分方程数值解。最后网络模型预测的输出和真值计算 LpLoss 或者 H1 损失函数。

<figure markdown>
  ![FNO-pretraining](https://github.com/PaddlePaddle/PaddleScience/assets/71805205/ae66f124-04cb-4b5d-b45a-c916cc4f22b7){ loading=lazy style="margin:0 auto;height:70%;width:70%"}
  <figcaption>FNO 模型预训练</figcaption>
</figure>

在推理阶段，给定大小为 $w*h$ 的二维偏微分数据，预测得到大小为 $w*h$ 的二维偏微分方程数值解。

<figure markdown>
  ![FNO-infer](https://github.com/PaddlePaddle/PaddleScience/assets/71805205/21b67b89-d87a-4dda-a354-91606f73296a){ loading=lazy style="margin:0 auto;height:60%;width:60%"}
  <figcaption>FNO 模型推理</figcaption>
</figure>

## 3. TFNO 模型训练 darcy-flow 实现

接下来开始讲解如何基于 PaddleScience 代码，实现 TFNO 模型对 darcy-flow 数据的训练与推理。关于该案例中的其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍<a id="3.1"></a>

使用 二维达西流 (darcy-flow) 数据集，这个问题的偏微分方程为：

$-\nabla\cdot (k(x)\nabla u(x))=f(x),x\in D$

其中，x 是位置，u(x) 是流体的压力，k(x) 是渗透率场，f(x) 是压力的函数。达西流问题可以被用来描述多孔介质的流动、弹性材料和热传导。在这里，我们定义了一个二维的平面区域 $D=[0,1]×[0,1]$，我们希望得到一个模型，可以在给定 k 渗透率场的情况下，估算出 u 流体压力。

**训练数据和测试数据：**

数据集包括 1000 条 16x16 分辨率大小的训练数据；50 条 32x32 和 50 条 32x32分辨率大小的测试数据。数据格式采用 NPY 格式保存。

### 3.2 模型预训练

#### 3.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="12" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:12:35
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `DarcyFlowDataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 16，`num_works` 为 0。

定义监督约束的代码如下：

``` py linenums="37" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:37:51
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数 `h1`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 3.2.2 模型构建

在该案例中，darcy-flow 基于 TFNO 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="131" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:131:133
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="46" title="examples/neuraloperator/conf/tfno_darcyflow_pretrain.yaml"
--8<--
examples/neuraloperator/conf/tfno_darcyflow_pretrain.yaml:46:75
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 3.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `StepDecay`，学习率大小设置为 `5e-3`。优化器使用 `Adam`,用 PaddleScience 代码表示如下：

``` py linenums="134" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:134:157
--8<--
```

#### 3.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="55" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:55:129
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `hlLoss` 和 `LpLoss`。

#### 3.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="159" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:159:178
--8<--
```

### 3.3 模型评估可视化

#### 3.3.1 测试集上评估模型

构建模型的代码为：

``` py linenums="265" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:265:267
--8<--
```

构建评估器的代码为：

``` py linenums="182" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:182:263
--8<--
```

#### 3.3.2 模型导出

构建模型的代码为：

``` py linenums="285" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:285:288
--8<--
```

实例化 `ppsci.solver.Solver`：

``` py linenums="290" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:290:294
--8<--
```

构建模型输入格式并导出静态模型：

``` py linenums="295" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:295:304
--8<--
```

`InputSpec` 函数中第一个设置模型输入尺寸，第二个参数设置输入数据类型，第三个设置输入数据的 `Key`.

#### 3.3.3 模型推理

创建预测器:

``` py linenums="309" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:309:311
--8<--
```

准备预测数据：

``` py linenums="313" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:313:316
--8<--
```

进行模型预测与预测值显示:

``` py linenums="318" title="examples/neuraloperator/train_tfno.py"
--8<--
examples/neuraloperator/train_tfno.py:318:341
--8<--
```

## 4. UNO 模型训练 darcy-flow 实现

### 4.1 数据集介绍

数据集同 [3.1 节](#3.1)。

### 4.2 模型预训练

#### 4.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="12" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:12:35
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `DarcyFlowDataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 16，`num_works` 为 0。

定义监督约束的代码如下：

``` py linenums="37" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:37:51
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数 `h1`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 4.2.2 模型构建

在该案例中，darcy-flow 基于 UNO 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="131" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:131:133
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="46" title="examples/neuraloperator/conf/uno_darcyflow_pretrain.yaml"
--8<--
examples/neuraloperator/conf/uno_darcyflow_pretrain.yaml:46:79
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 4.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `StepDecay`，学习率大小设置为 `5e-3`。优化器使用 `Adam`,用 PaddleScience 代码表示如下：

``` py linenums="134" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:134:157
--8<--
```

#### 4.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="55" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:55:129
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `hlLoss` 和 `LpLoss`。

#### 4.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="159" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:159:178
--8<--
```

### 4.3 模型评估可视化

#### 4.3.1 测试集上评估模型

构建模型的代码为：

``` py linenums="265" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:265:267
--8<--
```

构建评估器的代码为：

``` py linenums="182" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:182:263
--8<--
```

#### 4.3.2 模型导出

构建模型的代码为：

``` py linenums="285" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:285:288
--8<--
```

实例化 `ppsci.solver.Solver`：

``` py linenums="290" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:290:294
--8<--
```

构建模型输入格式并导出静态模型：

``` py linenums="295" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:295:304
--8<--
```

`InputSpec` 函数中第一个设置模型输入尺寸，第二个参数设置输入数据类型，第三个设置输入数据的 `Key`.

#### 4.3.3 模型推理

创建预测器:

``` py linenums="309" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:309:311
--8<--
```

准备预测数据：

``` py linenums="313" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:313:316
--8<--
```

进行模型预测与预测值显示:

``` py linenums="318" title="examples/neuraloperator/train_uno.py"
--8<--
examples/neuraloperator/train_uno.py:318:341
--8<--
```

## 5. SFNO 模型训练 spherical Shallow Water equations(SWE) 实现

### 5.1 数据集介绍

球面浅水方程（Spherical Shallow Water Equations，简称SWE）是一组描述在旋转地球表面上的浅水流动的偏微分方程。浅水方程通常用于模拟海洋、湖泊和河流中的流体运动，当流体的垂直尺度远小于其水平尺度时，可以忽略流体的垂直结构，只考虑其水平运动。

球面浅水方程在数学上可以由以下方程组表示：

$\frac{\partial u}{\partial t} +u\cdot \nabla u=-g\nabla h-fu+F$

$\frac{\partial h}{\partial t}+\nabla \cdot (hu)=0$

其中：

𝑢 是水平速度场，通常包含经度和纬度方向的速度分量。

ℎ 是流体高度（或水面高度）相对于参考水平面的位移。

𝑔 是重力加速度。

𝑓 是科里奥利参数，它与地球自转和纬度有关，f=2Ωsinϕ，其中 Ω 是地球自转的角度，𝜙 是纬度。

𝐹 是摩擦力和其他外部力（如风力）的向量。

∇ 是水平梯度算子。

球面浅水方程考虑了地球的球形几何，因此使用的是球面坐标系。在实际应用中，这些方程通常需要进行离散化和数值求解，以便于在计算机上进行模拟。

**训练数据和测试数据：**

数据集包括 200 条 32x64 分辨率大小的训练数据；50 条 32x64 和 50 条 64x128 分辨率大小的测试数据。数据格式采用 NPY 格式保存。

### 5.2 模型预训练

#### 5.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="12" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:12:30
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `DarcyFlowDataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 4，`num_works` 为 0。

定义监督约束的代码如下：

``` py linenums="32" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:32:41
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数 `Lp`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 5.2.2 模型构建

在该案例中，SWE 基于 SFNO 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="104" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:104:106
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="41" title="examples/neuraloperator/conf/sfno_swe_pretrain.yaml"
--8<--
examples/neuraloperator/conf/sfno_swe_pretrain.yaml:41:69
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 5.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `StepDecay`，学习率大小设置为 `5e-3`。优化器使用 `Adam`,用 PaddleScience 代码表示如下：

``` py linenums="108" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:108:131
--8<--
```

#### 5.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="45" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:45:102
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标是 `LpLoss`。

#### 5.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="133" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:133:152
--8<--
```

### 5.3 模型评估可视化

#### 5.3.1 测试集上评估模型

构建模型的代码为：

``` py linenums="217" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:217:219
--8<--
```

构建评估器的代码为：

``` py linenums="156" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:156:215
--8<--
```

#### 5.3.2 模型导出

构建模型的代码为：

``` py linenums="237" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:237:240
--8<--
```

实例化 `ppsci.solver.Solver`：

``` py linenums="242" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:242:246
--8<--
```

构建模型输入格式并导出静态模型：

``` py linenums="247" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:247:256
--8<--
```

`InputSpec` 函数中第一个设置模型输入尺寸，第二个参数设置输入数据类型，第三个设置输入数据的 `Key`.

#### 5.3.3 模型推理

创建预测器:

``` py linenums="261" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:261:263
--8<--
```

准备预测数据：

``` py linenums="265" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:265:267
--8<--
```

进行模型预测与预测值显示:

``` py linenums="269" title="examples/neuraloperator/train_sfno.py"
--8<--
examples/neuraloperator/train_sfno.py:269:291
--8<--
```

## 6. 结果展示

下图展示了 TFNO 对 Darcy-flow 数据的预测结果和真值结果。
k(x) 的黑色区域就是可以渗透的地方，白色为不可渗透的区域。右侧是目标结果，颜色越亮，压力越大。

<figure markdown>
  ![TFNO-predict](https://github.com/PaddlePaddle/PaddleScience/assets/71805205/d0c07ef6-cad3-4db2-8e03-fbd62458f740){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>TFNO 的预测结果（"Model prediction"）与真值结果（"Ground-truth y"）</figcaption>
</figure>

下图展示了 UNO 对 Darcy-flow 数据的预测结果和真值结果。

<figure markdown>
  ![UNO-predict](https://github.com/PaddlePaddle/PaddleScience/assets/71805205/8088aaf5-3479-4dde-b498-6ce123c10b4f){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>UNO 的预测结果（"Model prediction"）与真值结果（"Ground-truth y"）</figcaption>
</figure>

下图展示了 SFNO 对 SWE 数据的预测结果和真值结果。

<figure markdown>
  ![SFNO-predict](https://github.com/PaddlePaddle/PaddleScience/assets/71805205/a80b004a-009f-43e4-bb0f-919e1d9de4e5){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>SFNO的预测结果（"Model prediction"）与真值结果（"Ground-truth y"）</figcaption>
</figure>
