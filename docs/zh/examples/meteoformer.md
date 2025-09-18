# Meteoformer

开始训练、评估前，请下载ERA5数据集文件

开始评估前，请下载或训练生成预训练模型

=== "模型训练命令"

    ``` sh
    python main.py
    ```

=== "模型评估命令"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path=./outputs_meteoformer/checkpoints/best_model.pdparams
    ```

## 1. 背景简介

短中期气象预测主要涉及对未来几小时至几天内的天气变化进行预测。这类预测通常需要涵盖多个气象要素，如温度、湿度、风速等，这些要素对气象变化有着复杂的时空依赖关系。准确的短中期气象预测对于防灾减灾、农业生产、航空航天等领域具有重要意义。传统的气象预测模型主要依赖于物理公式和数值天气预报（NWP），但随着深度学习的快速发展，基于数据驱动的模型逐渐展现出更强的预测能力。

为了有效捕捉这些多维时空特征，Meteoformer应运而生。Meteoformer是一种基于Transformer架构的模型，专门针对短中期多气象要素的预测任务进行优化。该模型能够处理多个气象变量的时空依赖关系，采用自注意力机制来捕捉不同时空尺度的关联性，从而实现更准确的温度、湿度、风速等气象要素的多步预测。通过Meteoformer，气象预报可以实现更加高效和精确的多要素预测，为气象服务提供更加可靠的数据支持。

## 2. 模型原理

本章节对 Meteoformer 的模型原理进行简单地介绍。

### 2.1 编码器

该模块使用两层Transformer，提取空间特征更新节点特征：

``` py linenums="8" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:233:267
--8<--
```

### 2.2 演变器

该模块使用两层Transformer，学习全局时间动态特性：

``` py linenums="29" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:269:314
--8<--
```

### 2.3 解码器

该模块使用两层卷积，将时空表征解码为未来多气象要素：

``` py linenums="29" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:317:332
--8<--
```

### 2.4 Meteoformer模型结构

Meteoformer模型首先使用特征嵌入层对输入信号（过去几个时间帧的气象要素）进行空间特征编码：

``` py linenums="73" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:405:406
--8<--
```

``` py linenums="94" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:233:267
--8<--
```

然后模型利用演变器将学习空间特征的动态特性，预测未来几个时间帧的气象特征：

``` py linenums="75" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:409:411
--8<--
```

``` py linenums="96" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:269:314
--8<--
```

最后模型将时空动态特性与初始气象底层特征结合，使用两层卷积预测未来短中期内的多气象要素值：

``` py linenums="112" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:414:415
--8<--
```

``` py linenums="35" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:317:332
--8<--
```

## 3. 模型训练

### 3.1 数据集介绍

案例中使用了预处理的ERA5Meteo数据集，属于ERA5再分析数据的一个子集。ERA5Meteo包含了全球大气、陆地和海洋的多种变量，分辨率为31公里。该数据集从1979年开始到2020年，每小时提供一次天气状况的估计，非常适合用于短中期多气象要素预测等任务。在实际应用过程中，时间间隔选取为1小时。

数据集被保存为 T x C x H x W 的矩阵，记录了相应地点和时间的对应气象要素的值，其中 T 为时间序列长度，C代表通道维，案例中选取了3个不同气压层的温度、相对湿度、东向风速、北向风速等气象信息，H 和 W 代表按照经纬度划分后的矩阵的高度和宽度。根据年份，数据集按照 7:2:1 划分为训练集、验证集，和测试集。案例中预先计算了气象要素数据的均值与标准差，用于后续的正则化操作。

### 3.2 模型训练

#### 3.2.1 模型构建

该案例基于 Meteoformer 模型实现，用 PaddleScience 代码表示如下：

``` py linenums="79" title="examples/meteoformer/mian.py"
--8<--
examples/meteoformer/main.py:92:92
--8<--
```

#### 3.2.2 约束器构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束器。在定义约束器之前，需要首先指定约束器中用于数据加载的各个参数。

训练集数据加载的代码如下:

``` py linenums="20" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:23:38
--8<--
```

定义监督约束的代码如下：

``` py linenums="40" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:57:61
--8<--
```

#### 3.2.3 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。

验证集数据加载的代码如下:

``` py linenums="44" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:68:78
--8<--
```

定义监督评估器的代码如下：

``` py linenums="65" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:81:88
--8<--
```

#### 3.2.4 学习率与优化器构建

本案例中学习率大小设置为 `1e-3`，优化器使用 `Adam`，用 PaddleScience 代码表示如下：

``` py linenums="83" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:95:99
--8<--
```

#### 3.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="88" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:115:117
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py
--8<--
