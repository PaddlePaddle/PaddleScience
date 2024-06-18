# EarthFormer

开始训练、评估前，请先下载

[ICAR-ENSO数据集](https://tianchi.aliyun.com/dataset/98942)

[SEVIR数据集](https://nbviewer.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb#download
)

=== "模型训练命令"

    ``` sh
    # ICAR-ENSO 数据模型训练
    python examples/earthformer/earthformer_enso_train.py
    # SEVIR 数据模型训练
    python examples/earthformer/earthformer_sevir_train.py

    ```

=== "模型评估命令"

    ``` sh
    # ICAR-ENSO 模型评估
    python examples/earthformer/earthformer_enso_train.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_enso.pdparams
    # SEVIR 模型评估
    python examples/earthformer/earthformer_sevir_train.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_sevir.pdparams
    ```

=== "模型导出命令"

    ``` sh
    # ICAR-ENSO 模型推理
    python examples/earthformer/earthformer_enso_train.py mode=export
    # SEVIR 模型推理
    python examples/earthformer/earthformer_sevir_train.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # ICAR-ENSO 模型推理
    python examples/earthformer/earthformer_enso_train.py mode=infer
    # SEVIR 模型推理
    python examples/earthformer/earthformer_sevir_train.py mode=infer
    ```
| 模型 | 变量名称 | C-Nino3.4-M | C-Nino3.4-WM | MSE(1E-4) |
| :-- | :-- | :-- | :-- | :-- |
| [ENSO 模型](https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_enso.pdparams) | sst | 0.74130 | 2.28990 | 2.5000 |

| 模型 | 变量名称 | CSI-M | CSI-219 | CSI-181 | CSI-160 | CSI-133 | CSI-74 | CSI-16 | MSE(1E-4) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [SEVIR 模型](https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_sevir.pdparams) | vil | 0.4419 | 0.1791 | 0.2848 | 0.3232 | 0.4271 | 0.6860 | 0.7513 | 3.6957 |

## 1. 背景简介

地球是一个复杂的系统。地球系统的变化，从温度波动等常规事件到干旱、冰雹和厄尔尼诺/南方涛动 (ENSO) 等极端事件，影响着我们的日常生活。在所有后果中，地球系统的变化会影响农作物产量、航班延误、引发洪水和森林火灾。对这些变化进行准确及时的预测可以帮助人们采取必要的预防措施以避免危机，或者更好地利用风能和太阳能等自然资源。因此，改进地球变化（例如天气和气候）的预测模型具有巨大的社会经济影响。

Earthformer，一种用于地球系统预测的时空转换器。为了更好地探索时空注意力的设计，论文提出了 Cuboid Attention ，它是高效时空注意力的通用构建块。这个想法是将输入张量分解为不重叠的长方体，并行应用长方体级自注意力。由于我们将 O(N<sup>2</sup>) 自注意力限制在局部长方体内，因此整体复杂度大大降低。不同类型的相关性可以通过不同的长方体分解来捕获。同时论文引入了一组关注所有局部长方体的全局向量，从而收集系统的整体状态。通过关注全局向量，局部长方体可以掌握系统的总体动态并相互共享信息。

## 2. 模型原理

本章节仅对 EarthFormer 的模型原理进行简单地介绍，详细的理论推导请阅读 [Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](https://arxiv.org/abs/2207.05833)。

Earthformer 的网络模型使用了基于 Cuboid Attention 的分层 Transformer incoder-decoder 。这个想法是将数据分解为长方体并并行应用长方体级自注意力。这些长方体进一步与全局向量的集合连接。

模型的总体结构如图所示：

<figure markdown>
  ![Earthformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/earthformer_arch.png){ loading=lazy style="margin:0 auto;height:150%;width:150%"}
  <figcaption>EarthFormer 网络模型</figcaption>
</figure>

EarthFormer 原代码中训练了 ICAR-ENSO 数据集中海面温度 (sst) 和 SEVIR 数据集中对云总降水量 (vil) 的估计模型，接下来将介绍这两个模型的训练、推理过程。

### 2.1 ICAR-ENSO 和 SEVIR 模型的训练、推理过程

模型预训练阶段是基于随机初始化的网络权重对模型进行训练，如下图所示，其中 $[x_{i}]_{i=1}^{T}$ 表示长度为 $T$ 时空序列的输入气象数据，$[y_{T+i}]_{i=1}^{K}$ 表示预测未来 $K$ 步的气象数据，$[y_{T+i_true}]_{i=1}^{K}$ 表示未来 $K$ 步的真实数据，如海面温度数据和云总降水量数据。最后网络模型预测的输出和真值计算 mse 损失函数。

<figure markdown>
  ![earthformer-pretraining](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/earthformer-pretrain.png){ loading=lazy style="margin:0 auto;height:70%;width:70%"}
  <figcaption>earthformer 模型预训练</figcaption>
</figure>

在推理阶段，给定长度序列为 $T$ 的数据，得到长度序列为 $K$ 的预测结果。

<figure markdown>
  ![earthformer-pretraining](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/earthformer-infer.png){ loading=lazy style="margin:0 auto;height:60%;width:60%"}
  <figcaption>earthformer 模型推理</figcaption>
</figure>

## 3. 海面温度模型实现

接下来开始讲解如何基于 PaddleScience 代码，实现 EarthFormer 模型的训练与推理。关于该案例中的其余细节请参考 [API文档](../api/arch.md)。

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

``` py linenums="35" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:35:56
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `ENSODataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 16，`num_works` 为 8。

定义监督约束的代码如下：

``` py linenums="58" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:58:64
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数 `mse_loss`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 3.2.2 模型构建

在该案例中，海面温度模型基于 CuboidTransformer 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="97" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:97:99
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="46" title="examples/earthformer/conf/earthformer_enso_pretrain.yaml"
--8<--
examples/earthformer/conf/earthformer_enso_pretrain.yaml:46:105
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 3.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `Cosine`，学习率大小设置为 `2e-4`。优化器使用 `AdamW`，并将参数进行分组，使用不同的
`weight_decay`,用 PaddleScience 代码表示如下：

``` py linenums="101" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:101:126
--8<--
```

#### 3.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="68" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:68:95
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `MAE`、`MSE`、`RMSE`、`corr_nino3.4_epoch` 和 `corr_nino3.4_weighted_epoch`。

#### 3.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="128" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:128:146
--8<--
```

### 3.3 模型评估可视化

#### 3.3.1 测试集上评估模型

构建模型的代码为：

``` py linenums="179" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:179:181
--8<--
```

构建评估器的代码为：

``` py linenums="150" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:150:177
--8<--
```

#### 3.3.2 模型导出

构建模型的代码为：

``` py linenums="199" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:199:202
--8<--
```

实例化 `ppsci.solver.Solver`：

``` py linenums="204" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:204:208
--8<--
```

构建模型输入格式并导出静态模型：

``` py linenums="212" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:212:218
--8<--
```

`InputSpec` 函数中第一个设置模型输入尺寸，第二个参数设置输入数据类型，第三个设置输入数据的 `Key`.

#### 3.3.3 模型推理

创建预测器:

``` py linenums="222" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:222:224
--8<--
```

准备预测数据：

``` py linenums="226" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:226:249
--8<--
```

进行模型预测与预测值保存:

``` py linenums="253" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:253:258
--8<--
```

## 4. 云总降水量 vil 模型实现

### 4.1 数据集介绍

数据集采用了 [EarthFormer](https://github.com/amazon-science/earth-forecasting-transformer/tree/main) 处理好的 SEVIR 数据集。

The Storm Event ImagRy(SEVIR) 数据集是由麻省理工林肯实验室和亚马逊收集并提供的。SEVIR 是一个经过注释、整理和时空对齐的数据集，包含 10,000 多个天气事件，每个事件由 384 千米 x 384 千米的图像序列组成，时间跨度为 4 小时。SEVIR 中的图像通过五种不同的数据类型进行采样和对齐：GOES-16 高级基线成像仪的三个通道 (C02、C09、C13)、NEXRAD 垂直液态水含量 (vil) 和 GOES-16 地球静止闪电成像 (GLM) 闪烁图。

SEVIR数据集的结构包括两部分：目录 (Catalog) 和数据文件 (Data File)。目录是一个 CSV 文件，其中包含描述事件元数据的行。数据文件是一组 HDF5 文件，包含特定传感器类型的事件。这些文件中的数据以 4D 张量形式存储，形状为 N x L x W x T，其中 N 是文件中的事件数，LxW 是图像大小，T 是图像序列中的时间步数。
<figure markdown>
  ![SEVIR](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/sevir.png){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>SEVIR 传感器类型说明</figcaption>
</figure>

EarthFormer 采用 SEVIR 中的 NEXRAD 垂直液态水含量 (VIL) 作为降水预报的基准，即在 65 分钟的垂直综合液体背景下，预测未来 60 分钟的垂直综合液体。因此，分辨率为 13x384x384&rarr;12x384x384。

### 4.2 模型预训练

#### 4.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="27" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:27:59
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `ENSODataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 1，`num_works` 为 8。

定义监督约束的代码如下：

``` py linenums="61" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:61:67
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数 `mse_loss`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

### 4.2.2 模型构建

在该案例中，云总降水量模型基于 CuboidTransformer 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="117" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:117:119
--8<--
```

定义模型的参数通过配置进行设置，如下：

``` yaml linenums="58" title="examples/earthformer/conf/earthformer_sevir_pretrain.yaml"
--8<--
examples/earthformer/conf/earthformer_sevir_pretrain.yaml:58:117
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 4.2.3 学习率与优化器构建

本案例中使用的学习率方法为 `Cosine`，学习率大小设置为 `1e-3`。优化器使用 `AdamW`，并将参数进行分组，使用不同的 `weight_decay`,用 PaddleScience 代码表示如下：

``` py linenums="121" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:121:146
--8<--
```

#### 4.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="71" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:71:115
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `MAE`、`MSE`、`csi`、`pod`、`sucr`和 `bias`，且后四个评价指标分别使用不同的阈值 `[16,74,133,160,181,219]`。

#### 4.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="148" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:148:164
--8<--
```

#### 4.2.6 模型评估

由于目前 `paddlescience` 中的验证策略分为两类，一类是直接对验证数据集进行模型输出拼接，然后计算评价指标。另一类是按照每个 batch_size 计算评价指标，然后拼接，最后对所有结果求平均，该方法默认数据之间没有关联性。但是 `SEVIR` 数据集数据之间有关联性，所以不适用第二种方法；又由于 `SEVIR` 数据集量大，使用第一种方法验证显存需求大，因此验证 `SEVIR` 数据集使用的方法如下：

- 1.对一个 batch size 计算 `hits`、`misses` 和 `fas` 三个数据
- 2.对数据集所有数据保存所有 `batch` 的三个值的累加和.
- 3.对三个值的累加和计算 `csi`、`pod`、`sucr`和 `bias` 四个指标。

``` py linenums="165" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:165:181
--8<--
```

### 4.3 模型评估可视化

#### 4.3.1 测试集上评估模型

构建模型的代码为：

``` py linenums="231" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:231:233
--8<--
```

构建评估器的代码为：

``` py linenums="185" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:185:229
--8<--
```

模型评估：

``` py linenums="246" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:246:262
--8<--
```

#### 4.3.2 模型导出

构建模型的代码为：

``` py linenums="266" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:266:269
--8<--
```

实例化 `ppsci.solver.Solver`：

``` py linenums="271" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:271:275
--8<--
```

构建模型输入格式并导出静态模型：

``` py linenums="279" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:279:285
--8<--
```

`InputSpec` 函数中第一个设置模型输入尺寸，第二个参数设置输入数据类型，第三个设置输入数据的 `Key`.

#### 4.3.3 模型推理

创建预测器:

``` py linenums="293" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:293:294
--8<--
```

准备预测数据并进行对应模式的数据预处理：

``` py linenums="295" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:295:314
--8<--
```

进行模型预测并可视化:

``` py linenums="318" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:318:330
--8<--
```

## 5. 完整代码

``` py linenums="1" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py
--8<--
```

``` py linenums="1" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py
--8<--
```

## 6. 结果展示

下图展示了云总降水量模型按照65分钟的输入数据，得到60分钟间隔的预测结果和真值结果。

<figure markdown>
  ![SEVIR-predict](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/sevir-predict.png){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>SEVIR 中 vil 的预测结果（"prediction"）与真值结果（"target"）</figcaption>
</figure>

说明：

Hit:TP, Miss:FN, False Alarm：FP

第一行: 输入数据；

第二行: 真值结果；

第三行: 预测结果；

第四行: 设定阈值为 `74` 情况下，TP、FN、FP 三种情况标记

第五行： 在所有阈值情况下，TP、FN、FP 三种情况标记
