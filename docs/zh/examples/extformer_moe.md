# Extformer-MoE

!!! note

    1. 开始训练、评估前，请先下载 [ICAR-ENSO数据集](https://tianchi.aliyun.com/dataset/98942)，并对应修改 yaml 配置文件中的 `FILE_PATH` 为解压后的数据集路径。
    2. 开始训练、评估前，请安装 `xarray` 和 `h5netcdf`：`pip install requirements.txt`
    3. 若训练时显存不足，可指定 `MODEL.checkpoint_level` 为 `1` 或 `2`，此时使用 recompute 模式运行，以训练时间换取显存。

=== "模型训练命令"

    ``` sh
    # ICAR-ENSO 数据预训练模型: Extformer-MoE
    python extformer_moe_enso_train.py
    # python extformer_moe_enso_train.py MODEL.checkpoint_level=1 # using recompute to run in device with small GPU memory
    # python extformer_moe_enso_train.py MODEL.checkpoint_level=2 # using recompute to run in device with small GPU memory
    ```

=== "模型评估命令"

    ``` sh
    # ICAR-ENSO 模型评估: Extformer-MoE
    python extformer_moe_enso_train.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/extformer-moe/extformer_moe_pretrained.pdparams
    ```

| 模型 | 变量名称 | C-Nino3.4-M | C-Nino3.4-WM | MSE(1E-4) | MAE(1E-1) | RMSE |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [Extformer-MoE](https://paddle-org.bj.bcebos.com/paddlescience/models/extformer-moe/extformer_moe_pretrained.pdparams) | sst | 0.7651 | 2.39771 | 3.0000 | 0.1291 | 0.50243 |

## 1. 背景简介

地球是一个复杂的系统。地球系统的变化，从温度波动等常规事件到干旱、冰雹和厄尔尼诺/南方涛动 (ENSO) 等极端事件，影响着我们的日常生活。在所有后果中，地球系统的变化会影响农作物产量、航班延误、引发洪水和森林火灾。对这些变化进行准确及时的预测可以帮助人们采取必要的预防措施以避免危机，或者更好地利用风能和太阳能等自然资源。因此，改进地球变化（例如天气和气候）的预测模型具有巨大的社会经济影响。

近年来，深度学习模型在天气和气候预报任务中显示出了巨大的潜力。相较于传统的数值模拟方法，深度学习方法通过利用视觉神经网络 (ViT) 或图神经网络 (GNN) 等新兴技术直接从海量再分析数据中学习当前和未来天气或气候状态之间的复杂映射关系，在预测效率和精度方面均取得了显著的提升。然而，地球变化中发生的极端事件往往呈现出长距离时空同步关联、时空分布规律多样以及极值观测信号稀疏等特点，给基于深度学习的地球系统极端事件预测模型的构建带来了诸多新的技术挑战。

### 1.1 长距离时空同步关联

面对复杂耦合的地球变化系统，现有基于视觉和图深度学习的技术在建模极端天气呈现出的长距离时空关联性时存在诸多不足。具体而言，基于视觉深度学习的智能预报模型（例如华为的盘古气象大模型）仅限于计算局部区域内的信息交互，无法高效利用来自遥远区域的全局信息。相比之下，基于图神经网络的天气预报方法（例如谷歌的GraphCast）可以通过预定义的图结构进行远程信息传播，然而先验图结构难以有效识别影响极端天气的关键长距离信息且容易受到噪声影响，导致模型产生有偏甚至错误的预测结果。此外，地球系统的气象数据一般具有海量的网格点，在挖掘全局的长距离时空关联信息的同时，可能会导致模型复杂度的激增，如何高效建模时空数据中的长距离关联成为地球系统极端事件预测的重大挑战。

Earthformer，一种用于地球系统预测的时空转换器。为了更好地探索时空注意力的设计，其中设计了 Cuboid Attention ，它是高效时空注意力的通用构建块。这个想法是将输入张量分解为不重叠的长方体，并行应用长方体级自注意力。由于我们将 O(N<sup>2</sup>) 自注意力限制在局部长方体内，因此模型整体复杂度大大降低。不同类型的相关性可以通过不同的长方体分解来捕获。同时 Earthformer 引入了一组关注所有局部长方体的全局向量，从而收集系统的整体状态。通过关注全局向量，局部长方体可以掌握系统的总体动态并相互共享信息，从而捕获到地球系统的长距离关联信息。

### 1.2 时空分布规律多样

精准建模时空分布规律的多样性是提升地球系统极端事件预测的关键。现有方法在时域和空域均使用共享的参数，无法有效捕捉特定于时段和地理位置独特的的极端天气特征模式。

混合专家（MoE, Mixture-of-Experts）网络，它包含一组专家网络和门控网络。每个专家网络都是独立的神经网络，拥有独立的参数，门控网络自适应地为每个输入单元选择一个独特的专家网络子集。在训练和推理过程中，每个输入单元只需要利用一个很小的专家网络子集，因此可以扩大专家网络的总数，在增强模型表达能力的同时维持相对较小的计算复杂度。在地球系统中，MoE 可以通过学习与时间、地理位置、模型输入相关的独有参数集合，从而增强模型捕捉时空分布差异性的能力。

### 1.3 极值观测信号稀疏

气象数据的不均衡分布会导致模型偏向于预测频繁出现的正常气象状况，而低估了观测值稀少的极端状况，因为模型训练中常用的回归损失函数比如均方误差（MSE）损失会导致预测结果的过平滑现象。与具有离散标签空间的不平衡分类问题不同，不平衡回归问题具有连续的标签空间，为极端预测问题带来了更大的挑战。

Rank-N-Contrast（RNC）是一种表征学习方法，旨在学习一种回归感知的样本表征，该表征以连续标签空间中的距离为依据，对嵌入空间中的样本间距离进行排序，然后利用它来预测最终连续的标签。在地球系统极端预测问题中，RNC 可以对气象数据的表征进行规范，使其满足嵌入空间的连续性，和标签空间对齐，最终缓解极端事件的预测结果的过平滑问题。

## 2. 模型原理

### 2.1 Earthformer

本章节仅对 EarthFormer 的模型原理进行简单地介绍，详细的理论推导请阅读 [Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](https://arxiv.org/abs/2207.05833)。

Earthformer 的网络模型使用了基于 Cuboid Attention 的分层 Encoder-Decoder 架构Transformer，它将数据分解为长方体并并行应用长方体级自注意力，这些长方体进一步与全局向量的集合交互以捕获全局信息。

Earthformer 的总体结构如图所示：

<center class ='img'>
<img title="Earthformer" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/extformer-moe/Earthformer.png" width="60%">
</center>

### 2.2 Mixture-of-Experts

本章节仅对 Mixture-of-Experts 的原理进行简单地介绍，详细的理论推导请阅读 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
](https://arxiv.org/abs/1701.06538)。

混合专家（MoE, Mixture-of-Experts）网络，它包含一组参数独立的专家网络 $E_1,E_2,...,E_n$ 和门控网络 $G$。给定输入 $x$，MoE 网络的输出为 $y=\sum_{i=1}^n G(x)_iE_i(x)$。

MoE 的总体结构如图所示：

<center class ='img'>
<img title="MoE" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/extformer-moe/MoE.png" width="60%">
</center>

### 2.3 Rank-N-Contrast

Rank-N-Contrast（RNC）是一种根据样本在标签空间中的相互间的排序，通过对比来学习以学习连续性表征的的回归方法。RNC 的一个简单示例如图所示：

<center class ='img'>
<img title="RNC" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/extformer-moe/RNC.png" width="70%">
</center>

### 2.4 Extformer-MoE 模型的训练、推理过程

模型预训练阶段是基于随机初始化的网络权重对模型进行训练，如下图所示，其中 $[x_{i}]_{i=1}^{T}$ 表示长度为 $T$ 时空序列的输入气象数据，$[y_{i}]_{i=1}^{K}$ 表示预测未来 $K$ 步的气象数据，$[y_{i_True}]_{i=1}^{K}$ 表示未来 $K$ 步的真实数据，如海面温度数据和云总降水量数据。最后网络模型预测的输出和真值计算 mse 损失函数。在推理阶段，给定长度序列为 $T$ 的数据，得到长度序列为 $K$ 的预测结果。

## 3. 海面温度模型实现

接下来开始讲解如何基于 PaddleScience 代码，实现 Extformer-MoE 模型的训练与推理。关于该案例中的其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集采用了 [EarthFormer](https://github.com/amazon-science/earth-forecasting-transformer/tree/main) 处理好的 ICAR-ENSO 数据集。

本数据集由气候与应用前沿研究院 ICAR 提供。数据包括 CMIP5/6 模式的历史模拟数据和美国 SODA 模式重建的近100多年历史观测同化数据。每个样本包含以下气象及时空变量：海表温度异常 (SST) ，热含量异常 (T300)，纬向风异常 (Ua)，经向风异常 (Va)，数据维度为 (year,month,lat,lon)。训练数据提供对应月份的 Nino3.4 index 标签数据。测试用的初始场数据为国际多个海洋资料同化结果提供的随机抽取的 n 段 12 个时间序列，数据格式采用 NPY 格式保存。

**训练数据：**

每个数据样本第一维度 (year) 表征数据所对应起始年份，对于 CMIP 数据共 291 年，其中 1-2265 为 CMIP6 中 15 个模式提供的 151 年的历史模拟数据 (总共：151年 *15 个模式=2265) ；2266-4645 为 CMIP5 中 17 个模式提供的 140 年的历史模拟数据 (总共：140 年*17 个模式=2380)。对于历史观测同化数据为美国提供的 SODA 数据。

**训练数据标签**

标签数据为 Nino3.4 SST 异常指数，数据维度为 (year,month)。

CMIP(SODA)_train.nc 对应的标签数据当前时刻 Nino3.4 SST 异常指数的三个月滑动平均值，因此数据维度与维度介绍同训练数据一致。

注：三个月滑动平均值为当前月与未来两个月的平均值。

**测试数据**

测试用的初始场 (输入) 数据为国际多个海洋资料同化结果提供的随机抽取的 n 段 12 个时间序列，数据格式采用NPY格式保存，维度为 (12，lat，lon, 4), 12 为 t 时刻及过去 11 个时刻，4 为预测因子，并按照 SST,T300,Ua,Va 的顺序存放。

EarthFFormer 模型对于 ICAR-ENSO 数据集的训练中，只对其中海面温度 (SST) 进行训练和预测。训练海温异常观测的 12 步 (一年) ，预测海温异常最多 14 步。

### 3.2 模型预训练

#### 3.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="25" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:25:47
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `ExtMoEENSODataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 16，`num_works` 为 8。

定义监督约束的代码如下：

``` py linenums="49" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:49:55
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 3.2.2 模型构建

在该案例中，海面温度模型基于 ExtFormerMoECuboid 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="88" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:88:92
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="47" title="examples/earthformer/conf/earthformer_enso_pretrain.yaml"
--8<--
examples/extformer_moe/conf/extformer_moe_enso_pretrain.yaml:47:129
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 3.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `Cosine`，学习率大小设置为 `2e-4`。优化器使用 `AdamW`，并将参数进行分组，使用不同的
`weight_decay`,用 PaddleScience 代码表示如下：

``` py linenums="94" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:94:119
--8<--
```

#### 3.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="59" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:59:86
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `MAE`、`MSE`、`RMSE`、`corr_nino3.4_epoch` 和 `corr_nino3.4_weighted_epoch`。

#### 3.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="121" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:121:137
--8<--
```

### 3.3 模型评估

构建模型的代码为：

``` py linenums="138" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:138:139
--8<--
```

构建评估器的代码为：

``` py linenums="142" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:142:182
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py
--8<--
```
