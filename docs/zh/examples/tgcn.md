# TGCN

=== "模型训练命令"

    ``` sh
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip
    unzip tgcn_data.zip
    python run.py data_name=PEMSD8
    # python run.py data_name=PEMSD4
    ```

=== "模型评估命令"

    ``` sh
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip
    unzip tgcn_data.zip
    wget https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD8_pretrained_model.pdparams
    python run.py data_name=PEMSD8 mode=eval EVAL.pretrained_model_path=PEMSD8_pretrained_model.pdparams
    # wget https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD4_pretrained_model.pdparams
    # python run.py data_name=PEMSD4 mode=eval EVAL.pretrained_model_path=PEMSD4_pretrained_model.pdparams
    ```

| 预训练模型                                                   | 指标                    |
| ------------------------------------------------------------ | ----------------------- |
| [PEMSD4_pretrained_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD4_pretrained_model.pdparams) | MAE: 21.48; RMSE: 34.06 |
| [PEMSD8_pretrained_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD8_pretrained_model.pdparams) | MAE: 15.57; RMSE: 24.52 |


## 1. 背景简介

交通预测旨在通过分析历史观测数据（例如，交通网络上的传感器记录）来预测未来的交通时间序列状况（例如，交通流量或交通速度）。作为智能交通系统（ITS）的重要组成部分，交通预测任务是实现智慧城市的核心基础，包括主动动态交通控制和智能路线引导，有助于减少道路安全隐患并提高城市交通系统的运营效率。

TGCN，一种用于交通流量预测的时空图卷积网络（Temporal Graph Convolutional Network）。具体而言，通过将交通网络建模为图结构数据，使用图卷积网络（GCN）模块提取空间特征；通过将交通信号建模为时序信息，使用时间卷积网络（TCN）模块捕获时间特征。TGCN通过迭代执行两个模块，最终完成交通流量预测任务。

## 2. 模型原理

本章节对 TGCN 的模型原理进行简单的介绍。

### 2.1 图卷积网络模块

该模块使用两层消息传递网络，提取空间特征更新节点特征：

``` py linenums="12" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:12:35
--8<--
```

### 2.2 时间卷积网络模块

该模块使用三层一维卷积网络，提取时间特征更新节点特征：

``` py linenums="38" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:38:71
--8<--
```

### 2.3 TGCN模型结构

TGCN 模型首先使用特征嵌入层对输入信号（即交通节点在过去一段时间内的流量数据）进行编码：

``` py linenums="140" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:140:145
--8<--
```

``` py linenums="173" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:173:176
--8<--
```

然后模型交替堆叠前述 TCN 模块与 GCN 模块，更新节点特征：

``` py linenums="147" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:147:157
--8<--
```

``` py linenums="178" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:178:192
--8<--
```

最后模型将初始节点特征与两个 GCN 模块的输入拼接，使用两层 MLP 得到目标输出（即交通节点在未来一段时间内的流量预测）：

``` py linenums="159" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:159:170
--8<--
```

``` py linenums="194" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:194:198
--8<--
```

## 3. 模型训练

### 3.1 数据集介绍

案例中使用了预处理的 PEMSD4 和 PEMSD8 数据集。PEMSD4 为旧金山湾区交通数据，选取 29 条道路上 307 个传感器记录的交通数据，时间为 2018 年 1 月至 2 月。PEMSD8 为圣贝纳迪诺 8 条道路上 170 个检测器收集的交通数据，时间为 2016 年 7 月至 8 月。

两个数据集均被保存为 N x T x 1 的矩阵，记录了相应交通节点与时间的流量数据，其中 N 为交通节点数量，T 为时间序列长度。两个数据集分别按照 7:2:1 划分为训练集、验证集，和测试集。案例中预先计算了流量数据的均值与标准差，用于后续的正则化操作。

### 3.2 模型训练

#### 3.2.1 模型构建

该案例基于 TGCN 模型实现，用 PaddleScience 代码表示如下：

``` py linenums="67" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:67:82
--8<--
```

#### 3.2.2 约束器构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束器。在定义约束器之前，需要首先指定约束器中用于数据加载的各个参数。

训练集数据加载的代码如下:

``` py linenums="10" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:10:29
--8<--
```

定义监督约束的代码如下：

``` py linenums="31" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:31:35
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数 `L1_loss`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `train`。

#### 3.2.3 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。

验证集数据加载的代码如下:

``` py linenums="37" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:37:54
--8<--
```

定义监督评估器的代码如下：

``` py linenums="56" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:56:63
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 约束器比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用的评价指标分别是 `MAE` 和 `RMSE`。

#### 3.2.4 学习率与优化器构建

本案例中学习率大小设置为 `1e-2`，优化器使用 `Adam`，用 PaddleScience 代码表示如下：

``` py linenums="83" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:83:84
--8<--
```

#### 3.2.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="88" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:88:104
--8<--
```

#### 3.2.6 模型导出

通过设置 `ppsci.solver.Solver` 中的 `eval_during_train` 参数，可以自动保存在验证集上效果最优的模型参数。

``` py linenums="97" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:97:97
--8<--
```

### 3.3 评估模型

#### 3.3.1 评估器构建

测试集数据加载的代码如下:

``` py linenums="108" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:108:125
--8<--
```

定义监督评估器的代码如下：

``` py linenums="127" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:127:134
--8<--
```

与验证集的 `SupervisedValidator` 相似，在这里使用的评价指标分别是 `MAE` 和 `RMSE`。

#### 3.3.2 加载模型并进行评估

设置预训练模型参数的加载路径并加载模型。

``` py linenums="138" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:138:153
--8<--
```

实例化 `ppsci.solver.Solver`，然后启动评估。

``` py linenums="155" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:155:166
--8<--
```

## 4. 完整代码

数据集接口：

``` py linenums="1" title="ppsci/data/dataset/pems_dataset.py"
--8<--
ppsci/data/dataset/pems_dataset.py
--8<--
```

模型结构：

``` py linenums="1" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py
--8<--
```

模型训练：

``` py linenums="1" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py
--8<--
```

配置文件：

``` py linenums="1" title="examples/tgcn/conf/run.yaml"
--8<--
examples/tgcn/conf/run.yaml
--8<--
```

## 5. 结果展示

下表展示了 TGCN 在 PEMSD4 和 PEMSD8 两个数据集上的评估结果。

| 数据集 | MAE   | RMSE  |
| :----- | :---- | :---- |
| PEMSD4 | 21.48 | 34.06 |
| PEMSD8 | 15.57 | 24.52 |
