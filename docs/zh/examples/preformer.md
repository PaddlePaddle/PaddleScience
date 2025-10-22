# Preformer

开始训练、评估前，请下载[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)数据集文件。

开始评估前，请下载或训练生成预训练模型。

用于评估的数据集已保存，可通过下面的链接进行下载、评估：
[rain_2016_01.h5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/preformer/rain_2016_01.h5)、
[ERA5_201601.tar.gz](https://paddle-org.bj.bcebos.com/paddlescience/datasets/meteoformer/ERA5_201601.tar.gz)、
[mean.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/mean.nc)、
[std.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/std.nc)。

下载或解压完成后，请保持以下目录形式：
ERA5/
├── mean.nc
├── std.nc
├── rain_2016_01.h5
└── 2016/
    ├── r_2016010100.npy
    ├── ...

=== "模型训练命令"

    ``` sh
    python main.py
    ```

=== "模型评估命令"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/preformer/preformer.pdparams"
    ```

## 1. 背景简介

降水是一种与人类生产生活密切相关的天气现象。准确预测短临降水不仅为农业管理、交通规划以及灾害预防等公共服务提供关键技术支持，也是一项具有挑战性的学术研究任务。近年来，深度学习在气象预测领域取得了重大突破。以多模态三维（高度、经度及纬度）气象数据为研究对象，研究基于深度学习的短临降水预测方法，具有重要的理论研究价值和广阔的应用前景。

Preformer，一种用于短临降水预测的时空Transformer网络，该模型由编码器、演变器和解码器组成。具体而言，编码器通过探索embedding之间的依赖来编码空间特征。通过演变器，从重新排列的embedding中学习全局时间动态特性。最后在解码器中，将时空表征解码为未来降水量。

## 2. 模型原理

本章节对 Preformer 的模型原理进行简单地介绍。

### 2.1 编码器

该模块使用两层Transformer，提取空间特征更新节点特征：

``` py linenums="243" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:243:277
--8<--
```

### 2.2 演变器

该模块使用两层Transformer，学习全局时间动态特性：

``` py linenums="280" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:280:325
--8<--
```

### 2.3 解码器

该模块使用两层卷积，将时空表征解码为未来降水量：

``` py linenums="329" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:329:344
--8<--
```

### 2.4 Preformer模型结构

模型的总体结构如图所示：

<figure markdown>
  ![preformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/preformer/preformer.png){ loading=lazy style="margin:0 auto"}
  <figcaption>Preformer 网络模型</figcaption>
</figure>

Preformer模型首先使用特征嵌入层对输入信号（过去几小时的气象要素）进行空间特征编码：

``` py linenums="415" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:415:417
--8<--
```

然后模型利用演变器将学习空间特征的动态特性，预测未来几小时的气象特征：

``` py linenums="419" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:419:422
--8<--
```

最后模型将时空动态特性与初始气象底层特征结合，使用两层卷积预测未来短时降水强度：

``` py linenums="424" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:424:428
--8<--
```

## 3. 模型训练

### 3.1 数据集介绍

案例中使用了预处理的ERA5SQ数据集，属于ERA5再分析数据的一个子集。ERA5SQ包含了全球大气、陆地和海洋的多种变量，研究区域从东经 140° 到西经 70°，从北纬 55° 到赤道，空间分辨率为 0.25°。该数据集从2016年开始到2020年，每小时提供一次天气状况的估计，非常适合用于降水预测和水汽总量的分析等任务。

数据集被保存为 T x C x H x W 的矩阵，记录了相应地点和时间的降雨量和气象要素值，其中 T 为时间序列长度，C代表通道维，案例中选取了3个不同气压层的温度、相对湿度、东向风速、北向风速等气象信息，H 和 W 代表按照经纬度划分后的矩阵的高度和宽度。根据年份，数据集按照 7:2:1 划分为训练集、验证集，和测试集。案例中预先计算了降雨数据等的均值与标准差，用于后续的正则化操作。

### 3.2 模型训练

#### 3.2.1 模型构建

该案例基于 Preformer 模型实现，用 PaddleScience 代码表示如下：

``` py linenums="94" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:94:95
--8<--
```

#### 3.2.2 约束器构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束器。在定义约束器之前，需要首先指定约束器中用于数据加载的各个参数。

训练集数据加载的代码如下:

``` py linenums="23" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:23:56
--8<--
```

定义监督约束的代码如下：

``` py linenums="58" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:58:64
--8<--
```

#### 3.2.3 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。

验证集数据加载的代码如下:

``` py linenums="69" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:69:80
--8<--
```

定义监督评估器的代码如下：

``` py linenums="82" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:82:92
--8<--
```

#### 3.2.4 学习率与优化器构建

本案例中学习率大小设置为 `1e-3`，优化器使用 `Adam`，用 PaddleScience 代码表示如下：

``` py linenums="97" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:97:102
--8<--
```

#### 3.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="104" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:104:121
--8<--
```

#### 3.2.6 训练时评估

通过设置 `ppsci.solver.Solver` 中的 `eval_during_train` 参数，可以自动保存在验证集上效果最优的模型参数。

``` py linenums="113" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:113:113
--8<--
```

### 3.3 评估模型

#### 3.3.1 评估器构建

测试集数据加载的代码如下:

``` py linenums="127" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:127:138
--8<--
```

定义监督评估器的代码如下：

``` py linenums="140" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:140:150
--8<--
```

与验证集的 `SupervisedValidator` 相似，在这里使用的评价指标是 `MAE` 和 `MSE`。

#### 3.3.2 加载模型并进行评估

设置预训练模型参数的加载路径并加载模型。

``` py linenums="152" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:152:153
--8<--
```

实例化 `ppsci.solver.Solver`，然后启动评估。

``` py linenums="155" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:155:166
--8<--
```

## 4. 完整代码

数据集接口：

``` py linenums="1" title="ppsci/data/dataset/era5sq_dataset.py"
--8<--
ppsci/data/dataset/era5sq_dataset.py
--8<--
```

模型结构：

``` py linenums="1" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py
--8<--
```

模型训练：

``` py linenums="1" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py
--8<--
```

配置文件：

``` py linenums="1" title="examples/preformer/conf/preformer.yaml"
--8<--
examples/preformer/conf/preformer.yaml
--8<--
```

## 5. 结果展示

下图展示了Preformer模型在短时降水预测任务中的预测结果与真值结果对比。图中的横轴表示不同的时间段，每个时间段间隔为1小时，每次模型预测6帧降水量。

<figure markdown>
  ![result_precip](https://paddle-org.bj.bcebos.com/paddlescience/docs/preformer/result.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Preformer模型预测结果（"Ours"）与真值结果（"GT"）</figcaption>
</figure>

## 6. 参考资料

- [Preformer: Simple and Efficient Design for Precipitation Nowcasting With Transformers](https://ieeexplore.ieee.org/document/10288072)
