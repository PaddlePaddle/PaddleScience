# Climateformer

开始训练、评估前，请下载[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)数据集文件。

开始评估前，请下载或训练生成预训练模型。

用于评估的ERA5数据集2018年数据已保存，可通过以下链接进行下载、评估：
[2018.h5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/2018.h5)、
[mean.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/mean.nc)、
[std.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/std.nc)。

=== "模型训练命令"

    ``` sh
    python main.py
    ```

=== "模型评估命令"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/climateformer/climateformer.pdparams"
    ```

## 1. 背景简介

长期气候预测主要涉及对未来几周乃至几个月内的天气变化进行预测。这类预测通常需要涵盖多个气象要素，如温度、湿度、风速等，这些要素对气象变化有着复杂的时空依赖关系。准确的气候预测对于防灾减灾、农业生产、航空航天等领域具有重要意义。传统的气象预测模型主要依赖于物理公式和数值天气预报（NWP），但随着深度学习的快速发展，基于数据驱动的模型逐渐展现出更强的预测能力。

Climateformer，正是一种面向长期气候预测的时空深度学习框架。该模型的设计目标是学习并模拟气象系统在长周期下的演化范式。其架构通常包含三大模块：编码器将海温、气压、风场等多源、多圈层的气候变量编码为一个能够表征当前“气候态”的全局向量。核心的演变模块（基于Transformer结构）则致力于捕捉这些气候态之间跨越数周甚至数月的长程时间依赖性。最终，解码器根据演变后的状态向量，预测未来多个周期内的平均关键气候指数。通过Climateformer，气候预报可以实现更加高效和精确的多要素预测，为气象服务提供更加可靠的数据支持。

## 2. 模型原理

本章节对 Climateformer 的模型原理进行简单地介绍。

### 2.1 编码器

该模块使用两层Transformer，提取空间特征更新节点特征：

``` py linenums="243" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:243:277
--8<--
```

### 2.2 演变器

该模块使用两层Transformer，学习全局时间动态特性：

``` py linenums="280" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:280:325
--8<--
```

### 2.3 解码器

该模块使用两层卷积，将时空表征解码为未来多气象要素：

``` py linenums="329" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:329:344
--8<--
```

### 2.4 Climateformer模型结构

模型的总体结构如图所示：

<figure markdown>
  ![climateformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/climateformer/climateformer.png){ loading=lazy style="margin:0 auto"}
  <figcaption>Climateformer 网络模型</figcaption>
</figure>

Climateformer模型首先使用特征嵌入层对输入信号（多气象要素的过去几个周平均时间帧）进行空间特征编码：

``` py linenums="418" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:418:420
--8<--
```

然后模型利用演变器将学习空间特征的动态特性，预测未来几个周平均时间帧的气象特征：

``` py linenums="422" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:422:425
--8<--
```

最后模型将时空动态特性与初始气象底层特征结合，使用两层卷积预测未来数周至数月的多气象要素周平均值：

``` py linenums="427" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:427:429
--8<--
```

## 3. 模型训练

### 3.1 数据集介绍

案例中使用了预处理的ERA5Climate数据集，属于ERA5再分析数据的一个子集。ERA5Climate包含了全球大气、陆地和海洋的多种变量，研究区域从东经 140° 到西经 70°，从北纬 55° 到赤道，空间分辨率为 0.25°。该数据集从2016年开始到2020年，每小时提供一次天气状况的估计，非常适合用于短中期多气象要素预测等任务。在实际应用过程中，时间间隔为一周，每帧选取为 7*24 小时内的周平均值。

数据集被保存为 T x C x H x W 的矩阵，记录了相应地点和时间的对应气象要素的值，其中 T 为时间序列长度，C代表通道维，案例中选取了3个不同气压层的温度、相对湿度、东向风速、北向风速等气象信息，H 和 W 代表按照经纬度划分后的矩阵的高度和宽度。根据年份，数据集按照 7:2:1 划分为训练集、验证集，和测试集。案例中预先计算了气象要素数据的均值与标准差，用于后续的正则化操作。

### 3.2 模型训练

#### 3.2.1 模型构建

该案例基于 Climateformer 模型实现，用 PaddleScience 代码表示如下：

``` py linenums="97" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:97:98
--8<--
```

#### 3.2.2 约束器构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束器。在定义约束器之前，需要首先指定约束器中用于数据加载的各个参数。

训练集数据加载的代码如下:

``` py linenums="23" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:23:58
--8<--
```

定义监督约束的代码如下：

``` py linenums="60" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:60:66
--8<--
```

#### 3.2.3 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。

验证集数据加载的代码如下:

``` py linenums="71" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:71:83
--8<--
```

定义监督评估器的代码如下：

``` py linenums="85" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:85:95
--8<--
```

#### 3.2.4 学习率与优化器构建

本案例中学习率大小设置为 `1e-3`，优化器使用 `Adam`，用 PaddleScience 代码表示如下：

``` py linenums="100" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:100:105
--8<--
```

#### 3.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="107" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:107:123
--8<--
```

#### 3.2.6 训练时评估

通过设置 `ppsci.solver.Solver` 中的 `eval_during_train` 参数，可以自动保存在验证集上效果最优的模型参数。

``` py linenums="116" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:116:116
--8<--
```

### 3.3 评估模型

#### 3.3.1 评估器构建

测试集数据加载的代码如下:

``` py linenums="129" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:129:141
--8<--
```

定义监督评估器的代码如下：

``` py linenums="143" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:143:153
--8<--
```

与验证集的 `SupervisedValidator` 相似，在这里使用的评价指标是 `MAE` 和 `MSE`。

#### 3.3.2 加载模型并进行评估

设置预训练模型参数的加载路径并加载模型。

``` py linenums="155" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:155:156
--8<--
```

实例化 `ppsci.solver.Solver`，然后启动评估。

``` py linenums="158" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:158:169
--8<--
```

## 4. 完整代码

数据集接口：

``` py linenums="1" title="ppsci/data/dataset/era5climate_dataset.py"
--8<--
ppsci/data/dataset/era5climate_dataset.py
--8<--
```

模型结构：

``` py linenums="1" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py
--8<--
```

模型训练：

``` py linenums="1" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py
--8<--
```

配置文件：

``` py linenums="1" title="examples/climateformer/conf/climateformer.yaml"
--8<--
examples/climateformer/conf/climateformer.yaml
--8<--
```

## 5. 结果展示

下图展示了Climateformer模型在1000 hPa等气压层温度预测任务中的预测结果与真值对比。横轴表示不同的预测时间步，时间间隔为1周，每次模型预测未来6周的周平均值。

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/climateformer/result.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Climateformer模型预测结果（"Pred"）与真值结果（"GT"）</figcaption>
</figure>
