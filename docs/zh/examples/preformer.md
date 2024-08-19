# Preformer

开始训练、评估前，请下载数据集文件

开始评估前，请下载或训练生成预训练模型

=== "模型训练命令"

    ``` sh
    # 模型训练
    python examples/preformer/train.py
    ```

=== "模型评估命令"

    ``` sh
    # 模型评估
    python examples/preformer/train.py mode=eval
    ```

## 1. 背景简介

降水是一种与人类生产生活密切相关的天气现象。准确预测短临降水不仅为农业管理、交通规划以及灾害预防等公共服务提供关键技术支持，也是一项具有挑战性的学术研究任务。近年来，深度学习在气象预测领域取得了重大突破。以多模态三维（高度、经度及纬度）气象数据为研究对象，研究基于深度学习的短临降水预测方法，具有重要的理论研究价值和广阔的应用前景。

Preformer，一种用于短临降水预测的时空Transformer网络，该模型由编码器、演变器和解码器组成。具体而言，编码器通过探索embedding之间的依赖来编码空间特征。通过演变器，从重新排列的embedding中学习全局时间动态特性。最后在解码器中，将时空表征解码为未来降水量。


## 2. 模型原理

本章节对 Preformer 的模型原理进行简单地介绍。

### 2.1 编码器

该模块使用两层Transformer，提取空间特征更新节点特征：

``` py linenums="8" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:194:217
--8<--
```

### 2.2 演变器

该模块使用两层Transformer，学习全局时间动态特性：

``` py linenums="29" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:220:254
--8<--
```

### 2.3 解码器

该模块使用两层卷积，将时空表征解码为未来降水量：

``` py linenums="29" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:257:273
--8<--
```

### 2.4 Preformer模型结构

Preformer模型首先使用特征嵌入层对输入信号（过去几小时的气象要素）进行空间特征编码：

``` py linenums="73" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:293:293
--8<--
```

``` py linenums="94" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:194:217
--8<--
```

然后模型利用演变器将学习空间特征的动态特性，预测未来几小时的气象特征：

``` py linenums="75" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:310:313
--8<--
```

``` py linenums="96" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:220:254
--8<--
```

最后模型将时空动态特性与初始气象底层特征结合，使用两层卷积预测未来短时降水强度：

``` py linenums="112" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:83:86
--8<--
```

``` py linenums="35" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:112:116
--8<--
```

## 3. 模型训练

### 3.1 数据集介绍

案例中使用了预处理的 PEMSD4 和 PEMSD8 数据集。PEMSD4 为旧金山湾区交通数据，选取 29 条道路上 307 个传感器记录的交通数据，时间为 2018 年 1 月至 2 月。PEMSD8 为圣贝纳迪诺 8 条道路上 170 个检测器收集的交通数据，时间为 2016 年 7 月至 8 月。

两个数据集均被保存为 N x T x 1 的矩阵，记录了相应交通节点与时间的流量数据，其中 N 为交通节点数量，T 为时间序列长度。两个数据集分别按照 7:2:1 划分为训练集、验证集，和测试集。案例中预先计算了流量数据的均值与标准差，用于后续的正则化操作。

### 3.2 模型训练

#### 3.2.1 模型构建

该案例基于 Preformer 模型实现，用 PaddleScience 代码表示如下：

``` py linenums="79" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:133:133
--8<--
```

#### 3.2.2 约束器构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束器。在定义约束器之前，需要首先指定约束器中用于数据加载的各个参数。

训练集数据加载的代码如下:

``` py linenums="20" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:44:79
--8<--
```

定义监督约束的代码如下：

``` py linenums="40" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:81:86
--8<--
```

#### 3.2.3 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。

验证集数据加载的代码如下:

``` py linenums="44" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:177:191
--8<--
```

定义监督评估器的代码如下：

``` py linenums="65" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:195:203
--8<--
```

#### 3.2.4 学习率与优化器构建

本案例中学习率大小设置为 `1e-3`，优化器使用 `Adam`，用 PaddleScience 代码表示如下：

``` py linenums="83" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:136:140
--8<--
```

#### 3.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="88" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:143:156
--8<--
```

#### 3.2.6 模型导出

通过设置 `ppsci.solver.Solver` 中的 `eval_during_train` 和 `eval_freq` 参数，可以自动保存在验证集上效果最优的模型参数。

``` py linenums="100" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:158:158
--8<--
```

#### 3.2.7 测试集上评估模型

训练完成后，启动评估流程在测试集上评估模型。

``` py linenums="112" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py:160:160
--8<--
```


## 4. 完整代码

数据集接口：

``` py linenums="1" title="ppsci\data\dataset\era5sq_dataset.py"
--8<--
ppsci\data\dataset\era5sq_dataset.py
--8<--
```

模型结构：

``` py linenums="1" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py
--8<--
```

模型训练：

``` py linenums="1" title="examples/preformer/train.py"
--8<--
examples/preformer/train.py
--8<--
```

配置文件：

``` py linenums="1" title="examples/preformer/conf/train.yaml"
--8<--
examples/preformer/conf/train.yaml
--8<--
```
