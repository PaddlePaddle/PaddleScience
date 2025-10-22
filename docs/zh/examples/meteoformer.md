# Meteoformer

开始训练、评估前，请下载[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)数据集文件。

开始评估前，请下载或训练生成预训练模型。

用于评估的数据集已保存，可通过以下链接进行下载、评估：
[ERA5_201601.tar.gz](https://paddle-org.bj.bcebos.com/paddlescience/datasets/meteoformer/ERA5_201601.tar.gz)、
[mean.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/mean.nc)、
[std.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/std.nc)。

下载或解压完成后，请保持以下目录形式：
ERA5/
├── mean.nc
├── std.nc
└── 2016/
    ├── r_2016010100.npy
    ├── ...

=== "模型训练命令"

    ``` sh
    python main.py
    ```

=== "模型评估命令"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/meteoformer/meteoformer.pdparams"
    ```

## 1. 背景简介

短中期气象预测主要涉及对未来几小时至几天内的天气变化进行预测。这类预测通常需要涵盖多个气象要素，如温度、湿度、风速等，这些要素对气象变化有着复杂的时空依赖关系。准确的短中期气象预测对于防灾减灾、农业生产、航空航天等领域具有重要意义。传统的气象预测模型主要依赖于物理公式和数值天气预报（NWP），但随着深度学习的快速发展，基于数据驱动的模型逐渐展现出更强的预测能力。

为了有效捕捉这些多维时空特征，Meteoformer应运而生。Meteoformer是一种基于Transformer架构的模型，专门针对短中期多气象要素的预测任务进行优化。该模型能够处理多个气象变量的时空依赖关系，采用自注意力机制来捕捉不同时空尺度的关联性，从而实现更准确的温度、湿度、风速等气象要素的多步预测。通过Meteoformer，气象预报可以实现更加高效和精确的多要素预测，为气象服务提供更加可靠的数据支持。

## 2. 模型原理

本章节对 Meteoformer 的模型原理进行简单地介绍。

### 2.1 编码器

该模块使用两层Transformer，提取空间特征更新节点特征：

``` py linenums="243" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:243:277
--8<--
```

### 2.2 演变器

该模块使用两层Transformer，学习全局时间动态特性：

``` py linenums="280" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:280:325
--8<--
```

### 2.3 解码器

该模块使用两层卷积，将时空表征解码为未来多气象要素：

``` py linenums="329" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:329:344
--8<--
```

### 2.4 Meteoformer模型结构

模型的总体结构如图所示：

<figure markdown>
  ![meteoformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/climateformer/climateformer.png){ loading=lazy style="margin:0 auto"}
  <figcaption>Meteoformer 模型结构</figcaption>
</figure>

Meteoformer模型首先使用特征嵌入层对输入信号（过去几个时间帧的多气象要素）进行空间特征编码：

``` py linenums="418" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:418:420
--8<--
```

然后模型利用演变器将学习空间特征的动态特性，预测未来几个时间帧的气象特征：

``` py linenums="422" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:422:425
--8<--
```

最后模型将时空动态特性与初始气象底层特征结合，使用两层卷积预测未来短中期内的多气象要素值：

``` py linenums="427" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:427:429
--8<--
```

## 3. 模型训练

### 3.1 数据集介绍

案例中使用了预处理的ERA5Meteo数据集，属于ERA5再分析数据的一个子集。ERA5Meteo包含了全球大气、陆地和海洋的多种变量，研究区域从东经 140° 到西经 70°，从北纬 55° 到赤道，空间分辨率为 0.25°。该数据集从2016年开始到2020年，每小时提供一次天气状况的估计，非常适合用于短中期多气象要素预测等任务。在实际应用过程中，时间间隔选取为1小时。

数据集被保存为 T x C x H x W 的矩阵，记录了相应地点和时间的对应气象要素的值，其中 T 为时间序列长度，C代表通道维，案例中选取了3个不同气压层的温度、相对湿度、东向风速、北向风速等气象信息，H 和 W 代表按照经纬度划分后的矩阵的高度和宽度。根据年份，数据集按照 7:2:1 划分为训练集、验证集，和测试集。案例中预先计算了气象要素数据的均值与标准差，用于后续的正则化操作。

### 3.2 模型训练

#### 3.2.1 模型构建

该案例基于 Meteoformer 模型实现，用 PaddleScience 代码表示如下：

``` py linenums="94" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:94:95
--8<--
```

#### 3.2.2 约束器构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束器。在定义约束器之前，需要首先指定约束器中用于数据加载的各个参数。

训练集数据加载的代码如下:

``` py linenums="23" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:23:56
--8<--
```

定义监督约束的代码如下：

``` py linenums="58" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:58:64
--8<--
```

#### 3.2.3 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。

验证集数据加载的代码如下:

``` py linenums="69" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:69:80
--8<--
```

定义监督评估器的代码如下：

``` py linenums="82" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:82:92
--8<--
```

#### 3.2.4 学习率与优化器构建

本案例中学习率大小设置为 `1e-3`，优化器使用 `Adam`，用 PaddleScience 代码表示如下：

``` py linenums="97" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:97:102
--8<--
```

#### 3.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="104" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:104:120
--8<--
```

#### 3.2.6 训练时评估

通过设置 `ppsci.solver.Solver` 中的 `eval_during_train` 参数，可以自动保存在验证集上效果最优的模型参数。

``` py linenums="113" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:113:113
--8<--
```

### 3.3 评估模型

#### 3.3.1 评估器构建

测试集数据加载的代码如下:

``` py linenums="126" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:126:137
--8<--
```

定义监督评估器的代码如下：

``` py linenums="139" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:139:149
--8<--
```

与验证集的 `SupervisedValidator` 相似，在这里使用的评价指标是 `MAE` 和 `MSE`。

#### 3.3.2 加载模型并进行评估

设置预训练模型参数的加载路径并加载模型。

``` py linenums="151" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:151:152
--8<--
```

实例化 `ppsci.solver.Solver`，然后启动评估。

``` py linenums="154" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:154:165
--8<--
```

## 4. 完整代码

数据集接口：

``` py linenums="1" title="ppsci/data/dataset/era5meteo_dataset.py"
--8<--
ppsci/data/dataset/era5meteo_dataset.py
--8<--
```

模型结构：

``` py linenums="1" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py
--8<--
```

模型训练：

``` py linenums="1" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py
--8<--
```

配置文件：

``` py linenums="1" title="examples/meteoformer/conf/meteoformer.yaml"
--8<--
examples/meteoformer/conf/meteoformer.yaml
--8<--
```

## 5. 结果展示

下图展示了Meteoformer模型在1000 hPa层风速预测任务中的预测结果与真值结果对比。横轴表示不同的预测时间步，时间间隔为1小时，模型一次可预测未来6个时间步。

<figure markdown>
  ![result_precip](https://paddle-org.bj.bcebos.com/paddlescience/docs/meteoformer/result.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Meteoformer模型预测结果（"Pred"）与真值结果（"GT"）</figcaption>
</figure>
