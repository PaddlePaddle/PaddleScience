# IFM-MLP

!!! note

    1. 开始训练、评估前，请先下载 molecules 数据集 [Google Drive](https://drive.google.com/drive/folders/1ZYdYQ0TtmShJC-z6dr4BU1aPfeQSE9gD?usp=sharing)，并对应修改 yaml 配置文件中的 `data_dir` 为解压后的数据集路径。
    2. 开始训练、评估前，请安装 `rdkit` 和 `scikit-learn`：`pip install requirements.txt`

=== "模型训练命令"

    ``` sh
    # ICAR-ENSO 数据预训练模型: IFM-MLP
    python ifm.py mode=train
    # python extformer_moe_enso_train.py MODEL.checkpoint_level=1 # using recompute to run in device with small GPU memory
    # python extformer_moe_enso_train.py MODEL.checkpoint_level=2 # using recompute to run in device with small GPU memory
    ```

=== "模型评估命令"

    ``` sh
    # ICAR-ENSO 模型评估: IFM-MLP
    python ifm.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/extformer-moe/extformer_moe_pretrained.pdparams
    ```



## 1. 背景简介

分子特性预测（MPP）是计算药物发现中的一项关键任务，旨在识别具有理想药理学和 ADMET（吸收、分布、代谢、排泄和毒性）特性。机器学习模型已被广泛应用在这个快速发展的领域，常用的模型有两种：传统的非深度模型和深度模型。在非深度模型中，分子被输入到传统机器中学习模型，例如计算或计算格式的随机森林和支持向量机手工分子指纹[64]。另一组利用深度模型来提取表达力以数据驱动的方式表示分子。具体来说，多层感知器（MLP）可应用于计算或手工制作的分子指纹；基于序列的神经网络架构包括循环神经网络（RNN）[43]、一维卷积神经网络(1D CNN) [22] 和 Transformers [25, 54] 被用来编码代表的分子简化的分子输入线路输入系统（SMILES）字符串[71]。此外，分子可以自然地表示为以原子为节点、键为边的图，激发了一系列致力于利用这种结构化归纳偏差来获得更好的分子表示[20,76,79,58]。

这些方法的关键进步是图神经网络（GNN），它在学习过程中同时考虑图结构和属性特征[33,68,24]。最近，研究人员将分子的 3D 构象纳入其表示中更好的性能，而实用的考虑因素，例如计算成本、对齐不变性，构象生成的不确定性以及目标分子不可用的构象限制了这些模型的实际适用性[5,17,57,16,38]。我们总结了广泛使用的分子我们的基准测试中的描述符及其相应的模型，如图 1 所示。取得了丰硕的进展，之前的研究[41,29,79,65,30,13,66]观察到深度模型在分子数据集上努力超越非深度模型。然而，这些研究都没有考虑新兴的强大深度模型（例如 Transformer [25]、SphereNet [37]）也没有探索各种分子描述符（例如 3D 分子图）。而且，他们也没有调查深层次的原因。模型经常在分子上失败。

为了缩小这一差距，我们提出了迄今为止最全面的分子特性预测基准研究，以及数据集包含和超参数调整的精确方法。我们的实证结果证实了之前研究的观察结果，即深度模型通常很难超越传统的非深度模型，即使不考虑深度学习算法训练速度较慢的情况。此外，我们观察到一些有趣的现象，这些现象挑战了社区的普遍信念，这可以指导未来研究的最佳方法设计。

我们开发了一种简单而有效的特征映射方法IFM，以帮助深度模型在理论保证的情况下学习非平滑目标函数。

## 2. IFM模型原理

### 2.1 IFM方法

本章节仅对 IFM 的模型原理进行简单地介绍，详细的理论推导请阅读 [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt)。

正如我们之前所解释的，深度模型很难学习分子的非平滑目标函数数据，这种现象在文献中被称为“光谱偏差”[51]。为了克服这种偏见，事先作品 [44, 87] 通过实验发现输入特征的启发式正弦映射允许 MLP 学习非平滑目标函数。然而，这些映射方法将不合需要地混合了原始功能。详细讨论请参见附录有限的空间。作为补救措施，我们引入了一种名为独立特征映射的新方法（IFM），在将分子特征的每个维度输入模型之前分别嵌入它们。将分子特征表示为 $x ∈ \mathbb{R}^d$，我们将 IFM 表示为：

$$
\begin{equation}
f_x = [\sin(v)|| \cos(v)], v = [2πc_1x, . . . , 2πc_kx]
\end{equation}
$$

其中 $||$ 表示两个向量的串联，$c = [c_1, c_2, ···, c_k]$ 是可学习参数从 $N(0, σ)$ 和 $f_x ∈ \mathbb{R}^{2k×d}$ 初始化。我们研究了超参数 $k$ 和 $σ$ 的影响附录。由于 $\cos(a − b) = \cos a \cos b + \sin a \sin b$，我们有：

$$
\begin{equation}
f_x · f_{x^′} =\sum_{i=1}^k cos(2πc_i(x − x^′)) := g_c(x − x^′)
\end{equation}
$$

其中· 是点积，$x^′$是另一个分子特征。因此，IFM 可以映射数据点到向量空间，以便它们的点积达到一定的距离度量，这是预期的特征映射方法的特征[52,4,23]。接下来我们将提供理论根据之前的研究，我们的 IFM 有效性的理由[62]。正如之前透露的有效，深度模型可以用神经正切核（NTK）来近似[28,2,6,36,62]。具体来说，让 $I$ 成为一个全连接的深度网络，其权重 $θ$ 是从高斯初始化的分布 $N$ ，NTK 理论表明，随着 $I$ 中层的宽度变得无穷大，并且随机梯度下降 (SGD) 的学习率接近零，函数 $I(x; θ)$ 收敛使用神经正切核 (NTK) 训练核回归解，即：

$$
\begin{equation}
h_{NTK}(x, x^′) = E_{θ∼N} \langle \frac{∂I(x; θ)}{∂θ} , \frac{∂I(x^′; θ)}{∂θ} \rangle
\end{equation}
$$

当输入仅限于超球面时，MLP 的 NTK 可以表示为点积内核（形式为 $h_{NTK}(x · x^′)$ 对于标量函数 $h_{NTK} : \mathbb{R} → \mathbb{R}$)。在我们的案例中，深度模型的输入为 $f_x$，IFM 和 NTK 的组合内核可以表示为：

$$
\begin{equation}
h_{NTK} (f_x · f_{x^′} ) = h_{NTK} (g_c (x − x^′)) = (h_{NTK} \circ g_c)(x − x^′)
\end{equation}
$$

因此，在这些映射的分子特征上训练深度模型对应于核回归平稳复合 NTK 函数 $h_{NTK} \circ g_c$。考虑到参数 $c$ 是可调的，IFM 创建了一个组合的 NTK，它不仅是固定的，而且是可调的。它使我们能够戏剧性地控制可以通过操纵参数 $c$ 学习的频率范围。


### 2.4 IFM 结合 MLP 模型的训练、推理实验


在我们的实验中，我们为各种深度模型配备了 IFM。具体来说，对于以指纹作为输入的 MLP，我们直接将所提出的特征映射方法应用于指纹（在特征选择和标准化之后）；
## 3. IFM模型实现

接下来开始讲解如何基于 PaddleScience 代码，实现 IFM-MLP 模型的训练与推理。关于该案例中的其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集采用了 [IFM](https://github.com/junxia97/IFM) 处理好的 molecues 数据集。

本数据集由IFM作者处理并提供。文章中作者对比了12个数据集，其提供的数据下载包括了至少5个分子数据集， 如bace，bbbp，hiv，sider，tox21等。数据集以csv格式保存。数据集包含了分子的 SMILES strings, labels 以及 fingerprints.

**Fingerprints数据设置**

以MLP使用的Fingerprints为例：遵循常见做法 [30,63,50]，我们将各种分子指纹的串联，包括 881 个 PubChem 指纹 (PubchemFP)、307 个子结构指纹 (SubFP) 和 206 个 MOE 1-D 和 2-D 描述符 [80] 提供给3SVM、XGB、RF和MLP模型全面表示分子结构，并通过一些预处理程序去除缺失值的特征（1）； (2) 方差极低（方差<0.05）； (3) 与另一个特征有很高的相关性（皮尔逊相关系数> 0.95）。保留的特征被归一化为平均值0和方差1。此外，考虑到传统机器模型（SVM、RF、XGB）不能直接应用于多任务分子数据集中，我们将多任务数据集分为多个单任务数据集并使用每个数据集来训练模型。最后，我们报告这些单个任务的平均性能。

**数据协议与测试设置**

首先，我们以 8:1:1 的比例随机分割训练集、验证集和测试集。然后，我们根据验证集的性能调整超参数。由于计算开销很大，基于 GNN 的 HIV 和 MUV 数据集模型正在进行 30 次评估； QM7和QM8上的所有型号都在10个评估中； QM9 数据集上的所有模型都在一次评估中。然后，我们使用之前确定的最佳超参数，使用不同的随机种子进行 50 次独立运行来进行数据集分割，以获得更可靠的结果。遵循 MoleculeNet 基准 [72]，我们使用接收者操作特征曲线下面积 (AUC-ROC) 评估分类任务，但 MUV 数据集上的精度曲线下面积 (AUC-PRC) 除外，因为其数据分布存在极端偏差。使用均方根误差 (RMSE) 或平均绝对误差 (MAE) 报告回归任务的性能。请注意，我们报告了某些数据集上多任务的平均性能，因为它们包含多个任务。此外，为了避免过度拟合问题，如果在连续 50 个 epoch 中没有观察到验证性能改善，则所有深度模型都会采用早期停止方案进行训练。我们将最大 epoch 设置为 300，批量大小设置为 128。我们在附录中提供了更多详细信息，包括每个模型的超参数调整空间。



### 3.2 模型预训练

#### 3.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="25" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:54:71
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `ExtMoEENSODataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 16，`num_works` 为 8。

定义监督约束的代码如下：

``` py linenums="49" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:73:79
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 3.2.2 模型构建

在该案例中，海面温度模型基于 ExtFormerMoECuboid 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="88" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:106:121
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

``` py linenums="94" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:124:124
--8<--
```

#### 3.2.4 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="59" title="examples/ifm/ifm.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:59:86
--8<--
```

`SupervisedValidator` 评估器与 `SupervisedConstraint` 比较相似，不同的是评估器需要设置评价指标 `metric`，在这里使用了自定义的评价指标分别是 `MAE`、`MSE`、`RMSE`、`corr_nino3.4_epoch` 和 `corr_nino3.4_weighted_epoch`。

#### 3.2.5 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="121" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:129:147
--8<--
```

### 3.3 模型评估

构建模型的代码为：

``` py linenums="138" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:204:219
--8<--
```

构建评估器的代码为：

``` py linenums="142" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:221:250
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py
--8<--
```

## 5. 结果展示

下表展示了MLP不嵌入IFM与嵌入IFM模型在不同数据集上的AUC_ROC表现。[IFM-MLP](https://paddle-org.bj.bcebos.com/paddlescience/models/extformer-moe/extformer_moe_pretrained.pdparams)

|  | tox21 | sider | hiv | bace | bbbp | 
| :-- | :-- | :-- | :-- | :-- | :-- |
| **MLP-None** | 0.82682 | 0.50039 | 0.71932 | 0.88891 | 0.66834 |
| **MLP-IFM** | 0.84245 | 0.60289 | 0.74007 | 0.89553 | 0.84864 |
| **At-Least** | 0.7578 | 0.5814 | 0.7344 | 0.8235 | 0.8433 |


可以看到增加了IFM模块的模型可以取得更优的预测结果。

## 6. 参考文献

- [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt)