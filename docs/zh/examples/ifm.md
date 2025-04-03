# IFM-MLP

!!! note

    1. 开始训练、评估前，请先下载 molecules 数据集 [dataset.zip](https://paddle-org.bj.bcebos.com/paddlescience/datasets/IFM/dataset.zip) ，或[Google Drive(作者原始链接)](https://drive.google.com/drive/folders/1ZYdYQ0TtmShJC-z6dr4BU1aPfeQSE9gD?usp=sharing)，并对应修改 yaml 配置文件中的 `data_dir` 为解压后的数据集路径。
    2. 如果需要使用预训练模型进行评估，请先下载预训练模型[pretrained.zip](https://paddle-org.bj.bcebos.com/paddlescience/models/IFM/pretrained.zip)并解压，例如pretrained路径。
    3. 开始训练、评估前，请安装 `rdkit` 和 `scikit-learn`等，相关依赖请执行`pip install requirements.txt`安装。

=== "模型训练命令"

    ``` sh
    # 在tox21/sider/hiv/bace/bbbp等数据上训练模型MLP-IFM,embed_name可选IFM/None
    # mode/data_label/MODEL.embed_name 等参数可在conf/ifm.yaml进行配置
    python ifm.py mode=train data_label=tox21 MODEL.embed_name='IFM'
    ```

=== "模型评估命令"

    ``` sh
    # 在tox21/sider/hiv/bace/bbbp等数据上评估模型MLP-IFM,embed_name可选IFM/None
    # 预训练模型的路径例如： pretrained/IFM/bace/model.pdparams 或使用自行训练的模型路径
    python ifm.py mode=eval data_label=tox21 MODEL.embed_name='IFM' EVAL.pretrained_model_path=pretrained/IFM/bace/model.pdparams
    ```

## 1. 背景简介

分子特性预测（MPP）是计算药物发现中的一项关键任务，旨在识别具有理想药理学和 ADMET（吸收、分布、代谢、排泄和毒性）特性。机器学习模型已被广泛应用在这个快速发展的领域，常用的模型有两种：传统的非深度模型和深度模型。在非深度模型中，分子被输入到传统机器学习模型，例如计算得到的或手动设计的分子指纹到随机森林和支持向量机等。另一类利用深度模型以数据驱动的方式来提取表征分子。具体来说，例如使用多层感知器（MLP）可应用于计算得到的或手动设计的分子指纹；基于序列的神经网络架构包括循环神经网络（RNN）、一维卷积神经网络(1D CNN) 和Transformers等可被用来编码表征的分子SMILES字符串。

此外，分子可以自然地表示为以原子为节点、键为边的图结构，激发了一系列致力于利用这种结构化归纳偏差来获得更好的分子表示。这些方法的关键成果是图神经网络（GNN），它在学习过程中同时考虑图结构和属性特征。最近，研究人员将分子的3D构象纳入其表示中取得了更好的性能，然而基于现实的因素考虑，例如计算成本、对齐不变性，构象生成的不确定性以及目标分子不可用的构象限制了这些模型的实际适用性。作者总结了被广泛使用的分子的描述符及其相应的模型来做基准测试。之前的大量研究，观察到深度模型在分子数据集上很难超越非深度模型。但是这些研究并没有考虑新兴的深度模型（例如Transformer、SphereNet）等，也没有研究不同分子描述符（例如3D分子图）的影响，也没有研究模型经常在分子上效果不佳的深层次的原因。

因此，作者进行了全面的分子特性预测基准研究，以及数据集和超参数调整的精确方法。结果证实了之前研究的观察结果，即深度模型通常很难超越传统的非深度模型，即使不考虑深度学习算法训练速度较慢的情况。因此，作者基于上述问题，提出了一种简单而有效的特征映射方法IFM，以帮助深度模型在理论情况下学习非平滑目标函数，取得了更好的效果。

## 2. IFM模型原理

### 2.1 IFM方法

本章节仅对 IFM 的模型原理进行简单地介绍，详细的理论推导请阅读 [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt)。

正如作者论文中所解释的，深度模型很难学习分子的非平滑目标函数数据，这种现象在文献中被称为“光谱偏差”。为了克服这种偏差，之前的一些工作通过实验发现输入特征的启发式正弦映射允许 MLP 学习非平滑目标函数。然而，这些映射方法将不可避免地混合进了原始特征。为了解决这种情况，作者引入了一种名为独立特征映射的新方法（IFM），在将分子特征的每个维度输入模型之前分别实现嵌入。将分子特征表示为 $x ∈ \mathbb{R}^d$，我们将 IFM 表示为：

$$
\begin{equation}
f_x = [\sin(v)|| \cos(v)], v = [2πc_1x, . . . , 2πc_kx]
\end{equation}
$$

其中 $||$ 表示两个向量的串联，$c = [c_1, c_2, ···, c_k]$ 是可学习参数，从 $N(0, σ)$ 和 $f_x ∈ \mathbb{R}^{2k×d}$ 初始化。作者研究了超参数 $k$ 和 $σ$ 的影响。由于 $\cos(a − b) = \cos a \cos b + \sin a \sin b$，我们有：

$$
\begin{equation}
f_x · f_{x^′} =\sum_{i=1}^k cos(2πc_i(x − x^′)) := g_c(x − x^′)
\end{equation}
$$

其中 · 是点积，$x^′$是另一个分子特征。因此，IFM 可以映射数据点到向量空间，以便它们的点积达到一定的距离度量，这是预期的特征映射方法的特征。根据之前的研究，作者提供了 IFM 有效性的理论依据。正如先前一些工作证明的有效性，深度模型可以用神经正切核（NTK）来近似。具体来说，让 $I$ 代表一个全连接的深度网络，其权重 $θ$ 是从高斯初始化的分布 $N$ ，NTK 理论表明，随着 $I$ 中层的宽度变得无穷大，并且随机梯度下降 (SGD) 的学习率接近零，在训练时函数 $I(x; θ)$ 收敛为使用神经正切核 (NTK)的核回归解，即：

$$
\begin{equation}
h_{NTK}(x, x^′) = E_{θ∼N} \langle \frac{∂I(x; θ)}{∂θ} , \frac{∂I(x^′; θ)}{∂θ} \rangle
\end{equation}
$$

当输入仅限于超球面时，MLP 的 NTK 可以表示为点积内核 (形式为 $h_{NTK}(x · x^′)$ 对于标量函数 $h_{NTK} : \mathbb{R} → \mathbb{R}$ )。在作者的方案中，深度模型的输入为 $f_x$，IFM 和 NTK 的组合内核可以表示为：

$$
\begin{equation}
h_{NTK} (f_x · f_{x^′} ) = h_{NTK} (g_c (x − x^′)) = (h_{NTK} \circ g_c)(x − x^′)
\end{equation}
$$

因此，在这些映射的分子特征上训练深度模型对应于核回归固定组合 NTK 函数 $h_{NTK} \circ g_c$。考虑到参数 $c$ 是可调的，IFM 创建了一个组合的 NTK，它不是固定的，而且是可调的。它使我们能够高效地通过操纵参数 $c$ 控制学习的频率范围。

### 2.2 IFM 结合 MLP 模型的训练、推理实验

在我们的实验中，我们为各种深度模型配备了 IFM。具体来说，对于以指纹作为输入的 MLP，我们直接将所提出的特征映射方法应用于指纹（在特征选择和标准化之后）。

## 3. IFM模型实现

接下来开始讲解如何基于 PaddleScience 代码，实现 IFM-MLP 模型的训练与推理。关于该案例中的其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集采用了作者 [IFM](https://github.com/junxia97/IFM) 处理好的 molecues 数据集。

本数据集由IFM作者处理并提供。文章中作者对比了12个数据集，其提供的数据下载包括了至少5个分子数据集， 如bace，bbbp，hiv，sider，tox21等。数据集以csv格式保存。数据集包含了分子的 SMILES strings, labels 以及 fingerprints.

**Fingerprints数据设置**

以MLP使用的Fingerprints为例：遵循常见做法，作者将各种分子指纹的拼接，包括 881 个 PubChem 指纹 (PubchemFP)、307 个子结构指纹 (SubFP) 和 206 个 MOE 1-D 和 2-D 描述符提供给SVM、XGB、RF和MLP模型全面表示分子结构，并通过一些预处理程序去除了一些特征，具体如：（1）缺失值的； (2) 方差极低（方差<0.05）； (3) 与另一个特征有很高的相关性（皮尔逊相关系数> 0.95）。保留的特征被归一化为平均值0和方差1。此外，考虑到传统机器模型（SVM、RF、XGB）不能直接应用于多任务分子数据集中，作者将多任务数据集分为多个单任务数据集并使用每个数据集来训练模型。

**数据协议与测试设置**

首先，作者以 8:1:1 的比例随机分割训练集、验证集和测试集。随后根据验证集的性能调整超参数，并使用之前确定的最佳超参数，使用不同的随机种子进行 50 次独立运行不同数据集分割，以获得更可靠的结果。遵循 MoleculeNet 基准，作者使用受试者操作特征曲线下面积 (AUC-ROC) 评估分类任务，但 MUV 数据集上的精度曲线下面积 (AUC-PRC) 除外，因为其数据分布存在极端偏差。使用均方根误差 (RMSE) 或平均绝对误差 (MAE) 报告回归任务的性能。作者报告了某些数据集上多任务的平均性能，因为它们包含多个任务。此外，为了避免过度拟合问题，如果在连续 50 个 epoch 中没有观察到验证性能改善，则所有深度模型都会采用早停方案进行训练。作者将最大 epoch 设置为 300，批大小设置为 128。更多详细信息，包括每个模型的超参数调整空间等，请参考作者原始论文。

本仓库使用的具体超参数已在yaml配置文件中预设，可根据情况自行调节。

### 3.2 模型预训练

#### 3.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="72" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:72:92
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `IFMMoeDataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 128，`num_works` 为 1。

定义监督约束的代码如下：

``` py linenums="94" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:94:100
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数；作者通过Regularization flag `reg` 参数控制损失函数选择： `MSELoss` 或 `BCEWithLogitsLoss`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 3.2.2 模型构建

在该案例中，分子属性预测模型基于 MLP 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="256" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:256:271
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="32" title="examples/ifm/conf/ifm.yaml"
--8<--
examples/ifm/conf/ifm.yaml:32:35
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称，具体超参数hyper_paras根据实验配置参考ifm.yaml中的`HYPER_OPT`字段。

#### 3.2.3 学习率与优化器构建

本案例中使用的学习率大小设置为 `0.001`。优化器使用 `Adam`，并将参数进行分组，使用不同的`weight_decay`,用 PaddleScience 代码表示如下：

``` py linenums="141" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:141:143
--8<--
```

#### 3.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="145" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:145:182
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `AUC-ROC`、`PRC-AUC`、`RMSE`、`MAE` 和 `R2`,程序会根据`data_label`进行设置，名称为`My_Metric`。

#### 3.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="184" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:184:202
--8<--
```

### 3.3 模型评估

构建模型的代码为：

``` py linenums="256" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:256:271
--8<--
```

构建评估器的代码为：

``` py linenums="273" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:273:310
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py
--8<--
```

## 5. 结果展示

下表展示了MLP模型不嵌入与嵌入作者提出的IFM，在不同数据集上的AUC_ROC表现对比。可下载预训练模型进行评估[IFM-MLP](https://paddle-org.bj.bcebos.com/paddlescience/models/IFM/pretrained.zip)

|  | tox21 | sider | hiv | bace | bbbp |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **MLP-None** | 0.82682 | 0.50039 | 0.71932 | 0.88891 | 0.66834 |
| **MLP-IFM** | 0.84245 | 0.60289 | 0.74007 | 0.89553 | 0.84864 |
| **MLP-IFM Loss** | 0.25697 | 1.36643 | 0.15742 | 0.47294 | 1.39181 |


可以看到增加了IFM模块的模型可以取得更优的预测结果，符合作者的设计目的。

## 6. 参考文献

- [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt)
- [作者原始仓库](https://github.com/junxia97/IFM)
