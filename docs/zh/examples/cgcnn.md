# CGCNN (Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties)

开始训练、评估前，请先下载[数据集](https://cmr.fysik.dtu.dk/c2db/c2db.html)并进行划分。数据读取需要额外安装依赖 `pymatgen`，请运行安装命令 `pip install pymatgen`。

| 预训练模型  | 指标 |
|:--| :--|
| [cgcnn_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/CGCNN/cgcnn_pretrained.pdparams) | loss(MAE): 0.4195 |

=== "模型训练命令"

    ``` sh
    python CGCNN.py TRAIN_DIR="Your train dataset path" VALID_DIR="Your evaluate dataset path"
    ```
=== "模型评估命令"

    ``` sh
    python CGCNN.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/CGCNN/cgcnn_pretrained.pdparams" TEST_DIR="Your test dataset path"
    ```

## 1. 背景简介

机器学习方法在加速新材料设计方面变得越来越流行，其预测材料性质的精度接近于从头计算，但计算速度要快几个数量级。晶体系统的任意尺寸带来了挑战，因为它们需要表示为固定长度的向量，以便与大多数算法兼容。这个问题通常是通过使用简单的材料属性手动构造固定长度的特征向量或设计原子坐标的对称不变变换来解决的。然而，前者需要逐个设计来预测不同的性质，而后者由于复杂的变换使得模型难以解释。CGCNN是一个广义的晶体图卷积神经网络框架框架，用于表示周期性晶体系统，它既提供了具有密度泛函理论(DFT)精度的材料性质预测，又提供了原子水平的化学见解。因此本案例使用CGNN对二维半导体材料的能带性质进行预测。

## 2. 模型原理

本章节仅对 CGCNN 的模型原理进行简单地介绍，详细的理论推导请阅读 [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)。

该方法的主要思想是通过一个同时编码原子信息和原子间成键相互作用的晶体图来表示晶体结构，然后在该图上构建一个卷积神经网络，通过使用DFT计算数据进行训练，自动提取最适合预测目标性质的表示。如图所示，晶体图是一个无向多图，它由代表原子的节点和代表晶体中原子之间连接的边所定义。晶体图不同于普通图，因为它允许在同一对端点之间有多条边，这是晶体图的一个特点，同时因为它们的周期性，其与分子图也有所不同。图中每个节点由一个特征向量表示，编码节点对应原子的属性。类似地，每条边同样用特征向量表示；对应于连接原子的键。

模型的总体结构如图所示：

![CGCNN_overview](https://paddle-org.bj.bcebos.com/paddlescience/docs/CGCNN/CGCNN.png)

CGCNN 论文中预测了七种不同性质，接下来将介绍如何使用 PaddleScience 代码实现 CGCNN 网络预测二维半导体间隙性质

### 3.1 数据集介绍

CGCNN 原文中使用的是 数据集 (<https://next-gen.materialsproject.org/>) 和 数据集(<https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html>)。本案例使用自行收集的数据集进行训练测试，如果用户需要使用本案例进行相关任务，可以参考以下数据集格式:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) 用于记录用户所需的晶体结构的文件。
- [id _ prop.csv] 每个晶体的目标属性。

您可以通过创建一个目录`root_dir`来创建一个自定义数据集，该目录包含以下文件:

1. `id_prop.csv`: [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) 第一列为每个晶体重新编码一个唯一的`ID`，第二列重新编码目标属性的值。

2. `atom_init.json`: [JSON](https://en.wikipedia.org/wiki/JSON) 存储每个元素的初始向量。

3. `ID.cif`: [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) 对晶体结构进行重新编码的文件，其中`ID`是晶体在数据集中的唯一ID。

`root_dir`的结构应该是(`root_dir`泛指训练/评估/测试数据文件夹):

``` log
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

### 3.2 模型构建

CGCNN 需要通过所使用的数据进行模型构造，因此需要先实例化`CGCNNDataset`。在实例化`CGCNNDataset`后可以得到训练样本的长度和输入维度等信息，根据此信息和设定的模型超参数`cfg.MODEL.atom_fea_len`、`cfg.MODEL.n_conv`、`cfg.MODEL.h_fea_len`、`cfg.MODEL.n_h`完成`CrystalGraphConvNet`的实例化。

``` py linenums="18" title="examples/cgcnn/CGCNN.py"
--8<--
examples/cgcnn/CGCNN.py:18:32
--8<--
```

其中超参数`cfg.MODEL.atom_fea_len`、`cfg.MODEL.n_conv`、`cfg.MODEL.h_fea_len`、`cfg.MODEL.n_h`默认设定如下：

``` yaml linenums="36" title="examples/cgcnn/conf/CGCNN.yaml"
--8<--
examples/cgcnn/conf/CGCNN.yaml:36:41
--8<--
```

### 3.3 约束构建

本问题模型为回归模型，采用监督学习方式进行训练，因此可以使用PaddleScience内置监督约束`SupervisedConstraint`构建监督约束。代码如下：

``` py linenums="34" title="examples/cgcnn/CGCNN.py"
--8<--
examples/cgcnn/CGCNN.py:34:51
--8<--
```

其中`root_dir`为训练集路径，`batch_size`为批训练大小。为了能够正常的批次训练，`collate_fn`需要根据模型进行重新设计。`collate_pool`代码如下：

``` py linenums="33" title="ppsci/data/dataset/cgcnn_dataset.py"
--8<--
ppsci/data/dataset/cgcnn_dataset.py:33:97
--8<--
```

### 3.4 评估器构建

为了实时监测模型的训练情况，我们将在每轮训练后对上一轮训练完毕的模型进行评估。与训练过程保持一致，我们使用PaddleScience内置的`SupervisedValidator`函数构建监督数据评估器。具体代码如下：

``` py linenums="53" title="examples/cgcnn/CGCNN.py"
--8<--
examples/cgcnn/CGCNN.py:53:70
--8<--
```

### 3.5 优化器构建

训练时使用`SGD`优化器进行训练，相关代码如下：

``` py linenums="72" title="examples/cgcnn/CGCNN.py"
--8<--
examples/cgcnn/CGCNN.py:72:76
--8<--
```

训练超参数`cfg.TRAIN.lr`、`cfg.TRAIN.momentum`、`cfg.TRAIN.weight_decay`等默认设定如下：

``` yaml linenums="49" title="examples/cgcnn/conf/CGCNN.yaml"
--8<--
examples/cgcnn/conf/CGCNN.yaml:49:51
--8<--
```

### 3.6 模型训练

由于本问题被建模为回归问题，因此可以使用PaddleScience内置的`psci.loss.MAELoss('mean')`作为训练过程的损失函数。同时选择使用随机梯度下降法对网络进行优化。并且将训练过程封装至PaddleScience内置的`Solver`中，具体代码如下：

``` py linenums="78" title="examples/cgcnn/CGCNN.py"
--8<--
examples/cgcnn/CGCNN.py:78:86
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/cgcnn/CGCNN.py"
--8<--
examples/cgcnn/CGCNN.py
--8<--
```

## 5. 参考资料

- [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
