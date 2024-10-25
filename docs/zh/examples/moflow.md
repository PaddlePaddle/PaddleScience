# MoFlow(Flow Model for Molecular)

<!--
<a href="https://aistudio.baidu.com/aistudio/projectdetail/6184070?contributionType=1&sUid=438690&shared=1&ts=1684239806160" class="md-button md-button--primary" style>AI Studio快速体验</a>
-->

!!! note "注意事项"

    1. 开始训练、评估前，请先下载 [QM9数据集和ZINC数据集](https://aistudio.baidu.com/datasetdetail/282687)，并对应修改 yaml 配置文件中的 `FILE_PATH` 为解压后的数据集路径，建议放置在例子./datasets/moflow中。
    2. 开始始训练、测试、优化评估前，请安装额外的化学包和数据显示转化工具 `pip install -r requirements.txt` 命令，安装 [rdkit](https://github.com/rdkit/rdkit) 化学工具和 [cairosvg](https://github.com/Kozea/CairoSVG) 数据转换保存工具。
    3. 预训模型需要进行修改，并且放到指定文件夹，修改对应的 yaml 配置文件，执行命令式分子生成不合理的话出现的提示可以不管

=== "模型训练命令"

    ``` sh
    # qm9 数据集模型训练
    python moflow_train.py data_name=qm9

    # zinc250k 数据集模型训练
    python moflow_train.py data_name=zinc250k
    ```

=== "模型推理评估命令"

    ``` sh
    # qm9 数据集预训练模型生成评估， 其中EVAL_mode=Reconstruct为重构生成， EVAL_mode=Random为随机生成，EVAL_mode=Inter2point为分子间插值生成，EVAL_mode=Intergrid为分子网格插值生成，详细说明参考3.7模型的生成评估构建
    python test_generate.py data_name=qm9 EVAL_mode=Reconstruct EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams

    # zinc250k 数据集预训练模型生成评估，其中EVAL_mode=Reconstruct为重构生成， EVAL_mode=Random为随机生成，EVAL_mode=Inter2point为分子间插值生成，EVAL_mode=Intergrid为分子网格插值生成，详细说明参考3.7模型的生成评估构建
    python test_generate.py data_name=zinc250k EVAL_mode=Reconstruct EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams
    ```

=== "模型优化评估命令"

    ``` sh
    # 方式一：不采用预训练模型，第一次运行为模型训练，第二次运行为预测生成结果输出
    # qm9 数据集预训练模型优化，其中OPTIMIZE.property_name=qed为潜空间到QED属性，OPTIMIZE.property_name=plogp从潜空间到plogp属性，详细说明参考3.8模型的优化构建
    python optimize_moflow.py data_name=qm9  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams  OPTIMIZE.property_name=qed

    # zinc250k 数据集预训练模型优化，其中OPTIMIZE.property_name=qed为潜空间到QED属性，OPTIMIZE.property_name=plogp从潜空间到plogp属性，详细说明参考3.8模型的优化构建
    python optimize_moflow.py data_name=zinc250k  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams OPTIMIZE.property_name=qed

    # 方式二：采用提供预训模型，下载优化后的模型进行预测结果生成输出
    # qm9 数据集预训练模型优化
    mkdir -p ./outputs_moflow_optimize/qm9/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qed_opt_pretrained.pdparams -O ./outputs_moflow_optimize/qm9/qed_model.pdparams
    python optimize_moflow.py data_name=qm9  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams  OPTIMIZE.property_name=qed

    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/plogp_opt_pretrained.pdparams -O ./outputs_moflow_optimize/qm9/plogp_model.pdparams
    python optimize_moflow.py data_name=qm9  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams  OPTIMIZE.property_name=plogp

    # zinc250k 数据集预训练模型优化
    mkdir -p ./outputs_moflow_optimize/zinc250k/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/qed_opt_pretrained.pdparams -O ./outputs_moflow_optimize/zinc250k/qed_model.pdparams
    python optimize_moflow.py data_name=zinc250k  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams OPTIMIZE.property_name=qed

    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/plogp_opt_pretrained.pdparams -O ./outputs_moflow_optimize/zinc250k/plogp_model.pdparams
    python optimize_moflow.py data_name=zinc250k  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams OPTIMIZE.property_name=plogp
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [qm9](https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams)|  loss(Residual): <br> 1.09976 |
| [zinc250k](https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams)|  loss(Residual):: <br> 1.12570 |

## 1. 背景简介

MoFlow一种基于流的图生成模型，旨在通过生成具有所需化学特性的分子图来加速药物发现过程。这样的图生成模型通常包括两个步骤：学习潜在表示和生成分子图，从潜在表示中生成新颖且符合化学规则的分子图是非常具有挑战性的，因为分子图具有化学约束和组合复杂性

MoFlow，用于学习分子图与其潜在表示之间的可逆映射，首先通过基于Glow的模型生成键（边），然后通过一种新颖的图条件流模型给定键来生成原子（节点），最后通过后处理的有效性修正将它们组装成一个符合化学规则的分子图，具有准确且可计算的似然训练、高效的一次嵌入和生成、化学有效性保证、对训练数据的100%重构以及良好的泛化能力等优点。其中的Glow模型的一个变种来生成键（多类型边，例如单键、双键和三键），以及一种基于图卷积的新颖图条件流来根据键生成原子（多类型节点，例如C、N等，将原子和键组装成符合键值约束的有效分子图）。

MoFlow是首个能够通过可逆映射一次生成分子图并具有有效性保证的基于流的图生成模型之一，为了捕捉分子图的组合原子和键结构，提出的Glow模型用于生成键（边），以及一种基于图条件流的新颖方法用于根据键生成原子（节点），然后将它们组装成有效的分子图；在分子图生成、重构、优化等方面实现了许多较好的结果，一次推理和生成非常高效，这意味着它在探索化学空间用于药物发现方面具高效性和有效性的潜力。

## 2. 模型原理

本章节仅对MoFlow的模型原理进行简单地介绍，详细的模型结构及推到过程阅读论文[MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://arxiv.org/abs/2006.10137v1)

### 2.1 模型的基础框架

Flow流模型在学习复杂高维数据$X \sim P_\mathcal{X}(X)$与具有相同维数的潜在空间中的$Z \sim P_\mathcal{Z}(Z)$之间的一系列可逆变换$f_Θ = f_L ◦ ... ◦ f_1$，其中潜在分布$P_\mathcal{Z}(Z)$易于建模（例如，在这样的潜在空间中成立强独立性假设）。原始空间中的潜在复杂数据可以通过变量变换公式来建模，其中$Z = f_Θ(X)$以及：

$$
\begin{aligned}
P_\mathcal{X}(X) & = P_\mathcal{Z}(Z)|\det(\frac{\partial Z}{\partial X})
\end{aligned}
$$

采样$\widetilde{X} \sim P_\mathcal{X}(X)$通过采样$\widetilde{Z} \sim P_\mathcal{Z}(Z)$，然后通过$\widetilde{X} = f_Θ^{−1}\widetilde{Z}$进行反向映射来转换$f_Θ$。设$Z = f_Θ(X) = f_L ◦ ... ◦ f_1(X)，H_l = f_l(H_{l−1})$，其中$f_l(l = 1, ...L ∈ \mathbb{N}^+)$是可逆映射，$H_0 = X，H_L = Z$，并且$P_\mathcal{Z}(Z)$遵循具有独立维度的标准各向同性高斯分布。然后，可以通过变量变换公式得到$X$的似然对数：

$$
\begin{aligned}
\log P_\mathcal{X}(X) &=\log P_\mathcal{Z}(Z) + \log \left|\det\left(\frac{\partial Z}{\partial X}\right) \right| \\
&= \sum_{i} \log P_{\mathcal{Z}_i}(Z_i) + \sum_{l=1}^L \log \left|\det\left(\frac{\partial f_l}{\partial H_{l−1}}\right)\right|
\end{aligned}
$$

其中$P_{\mathcal{Z}_i}(Z_i)$是$Z$的第$i^{th}$个维度的概率，$fΘ = f_L ◦ ... ◦ f_1$ 是要学习的可逆深度神经网络。

Coupling可逆放射耦合层，设计一个具有可逆性的函数f的表达性结构，能够计算雅可比行列式的效率通过一个仿射耦合变换$Z = f_Θ(X): \mathbb{R}^n \mapsto \mathbb{R}^n$:

$$
\begin{aligned}
Z_{1:d} & = X_{1:d} \\
Z_{d+1:n} & = X_{d+1:n} ⊙ e^{S_Θ(X_{1:d})} + T_Θ(X_{1:d})
\end{aligned}
$$

通过将$X$分为两个分区$X = (X_{1:d}, X_{d+1:n})$,通过以下方式保证可逆性：

$$
\begin{aligned}
X_{1:d} & = Z_{1:d} \\
X_{d+1:n} & = (Z_{d+1:n} - T_Θ(X_{1:d}) )/ e^{S_Θ(Z_{1:d})}
\end{aligned}
$$

表达能力取决于$X_{d+1:n}$的仿射变换中的任意神经结构的尺度函数 $S_Θ：\mathbb{R}^d \mapsto \mathbb{R}^{n-d}$和变换函数 $T_Θ：\mathbb{R}^d \mapsto \mathbb{R}^{n-d}$，雅可比行列式可以通过以下方式高效计算：$\det(\frac{\partial Z}{\partial X}) = \exp(\sum_j S_Θ(X_{1:d}))$。

### 2.2 MoFlow模型的原理

将$\text{M}$分子图视为由原子作节点，键作边组成的无向图，其数学符号可记为$\mathcal{M} =  \mathcal{A} \times  \mathcal{B} \subset \mathbb{R}^{n \times k} \times \mathbb{R}^{c \times n \times n}$，其中，集合有$n$个原子，$k$种原子类型，$A(i,k)=1$代表节点$i$是$k$型原子，集合代表键（边）,键有$c$种类型，$B(c,i,j)=1$代表原子$i$和$j$之间以$c$类型的键连接,一个分子$\mathcal{M}$可以被看作是一个具有多类型节点和多类型边的无向图。主要的目标是学习一个分子生成模型$P_{\mathcal{M}}(M)$，即从$P_{\mathcal{M}}$中采样任意分子$\text{M}$的概率。为了捕捉分子图的组合原子和键结构，将$P_{\mathcal{M}}(M)$分解为两部分：

$$
\begin{aligned}
P_\mathcal{M}(M) = P_{\mathcal{M}}((A, B)) ≈ P_{\mathcal{A|B}}(A|B; θ_{\mathcal{A|B}})P_\mathcal{B}(B; θ_\mathcal{B})
\end{aligned}
$$

其中$P_{\mathcal{M}}$是分子图的分布，$P_\mathcal{B}$是键（边）的分布，类似于对多通道图像建模，而$P_{\mathcal{A|B}}$是给定键的条件下的原子（节点）的条件分布，通过利用图卷积操作进行建模。$θ_\mathcal{B}$和$θ_{\mathcal{A|B}}$是可学习的建模参数。该模型的目标函数如下：

$$
\begin{aligned}
\mathop{\arg\max}\limits_{\theta_\mathcal{B}, \theta_\mathcal{A|B}}
\mathbb{E}_{\mathcal{M}=(A,B) \sim \mathcal{PM}−data} [ \log P_\mathcal{A|B}(A | B; θ_\mathcal{A|B} + \log P_\mathcal{B}(B; θ_\mathcal{B})]
\end{aligned}
$$

在给定键张量$B \in \mathcal{B} \subset \mathbb{R}^{c×n×n}$，生成正确的原子类型矩阵$A \in \mathcal{A} \subset \mathbb{R}^{n×k}$，以组成有效的分子$M = (A, B) \in \mathcal{M} \subset \mathbb{R}^{n×k+c×n×n}$。首先定义$B$条件流和图条件流$f_\mathcal{A|B}$，将给定$B$的$A$转化为条件潜变量$Z_{A|B} = f_\mathcal{A|B}(A|B)$，其遵循各向同性高斯分布$P_{\mathcal{Z}_\mathcal{A|B}}$。通过条件变量变换公式，可以得到给定键图的原子特征的条件概率$P_\mathcal{A|B}$。$B$条件流$Z_{A|B} = f_\mathcal{A|B}(A|B)$是一个可逆且保持维度的映射，存在逆变换$f^{−1}_\mathcal{A|B}(Z_{A|B} |B) = A|B$，其中$f_\mathcal{A|B}$和$f^{−1}_\mathcal{A|B}：\mathcal{A \times B} \mapsto \mathcal{A \times B}$。在变换过程中，$B \in B$保持不变。在$A$和$B$独立假设的条件下，$f_\mathcal{A|B}$的雅可比矩阵为：

$$
\begin{aligned}
\frac{\partial f_\mathcal{A|B}}{\partial (A, B)}=\bigg[\begin{matrix}
\frac{\partial f_\mathcal{A|B}}{\partial A} & \frac{\partial f_\mathcal{A|B}}{\partial B} \\ 0 & \mathbb{1}_B \end{matrix}\bigg]
\end{aligned}
$$

在得到了的分布，便可以从中抽样，利用逆映射得到$A|B$，并且利用雅克比矩阵给出$A|B$的概率分布，条件变量变换公式的对数似然为：

$$
\begin{aligned}
\log P_\mathcal{A|B}(A|B) = \log P_{\mathcal{Z}_\mathcal{A|B}}(Z_{A|B}) + \log |\det \frac{\partial f_\mathcal{A|B}}{\partial A}|
\end{aligned}
$$

和基于流的RealNVP、Glow模型一样，为了得到可逆映射，Moflow引入了图耦合层，对于每个图耦合层，沿着n行维度将输入$A \in \mathbb{R}^{n×k}$分为两部分$A = (A_1, A_2)$，然后按照以下方式得到输出$Z_{A|B} = (Z_{A_1|B}, Z_{A_2|B}) = f_\mathcal{A|B}(A|B)$,将输入分割成两个部分$A_1$和$A_2$：

$$
\begin{aligned}
Z_{A_1 |B} &= A_1 \\
Z_{A_2 |B} &= A_2 \odot \text{Sigmoid}(S_Θ(A_1 |B)) + T_Θ(A_1 |B)
\end{aligned}
$$

将上述式子求逆，即可得到$A_1$和$A_2$。图卷积层是利用关系图卷积网络（R-GCN）来完成的，具体操作如下：

$$
\begin{aligned}
\text{graphconv}(A_1) & = \sum_{i=1}^c \hat{B}_i (M \odot A)W_i + (M \odot A)W_0
\end{aligned}
$$

同时使用多个堆叠的图卷积->BatchNorm1d->ReLU层和一个多层感知机（MLP）输出层来构建图缩放函数$S_Θ$和图变换函数$T_Θ$。为了数值稳定性，在$S_Θ$中采用了Sigmoid函数，实现级联多个流层时数值稳定。图耦合层的逆映射$f^{-1}_\mathcal{A|B}$为：

$$
\begin{aligned}
A_1 &= Z_{A_1}|B \\
A_2 &= (Z_{A_2}|B - T_Θ(Z_{A_1|B}|B)) / \text{Sigmoid}(S_Θ(Z_{A_1|B}|B))
\end{aligned}
$$

每个图耦合层的雅可比行列式的对数可以通过以下方式进行计算：

$$
\begin{aligned}
\log | \det （\frac{\partial f_\mathcal{A|B}}{\partial A}）|= \sum_j \log \text{Sigmoid}(S_Θ(A_1|B))_j
\end{aligned}
$$

其中$j$迭代每个元素。可以使用任意复杂的图卷积结构来构建$S_Θ$和$T_Θ$，因为上述$f_\mathcal{A|B}$的雅可比行列式计算不涉及$S_Θ$或$T_Θ$的雅可比矩阵计算。

在学习原子表示的时候，为了保证数据稳定性，对于每一行维度使用$\sigma^2 \in \mathbb{R}^{n \times 1}$进行归一化，使得输入经过归一化后的结果为$\hat A = \frac{A - \mu} {\sqrt{\sigma^2 + \epsilon}}$，其中$\epsilon$是一个小的常数。反向变换为$A = \hat A \times \sqrt{\sigma^2 + \epsilon} + \mu$，并且对数雅可比行列式为：

$$
\begin{aligned}
\log | \det \frac{\partial actnorm2D}{\partial X}| = \frac{k}{2}\sum_i^n | \log(\sigma^2_i + \epsilon) |
\end{aligned}
$$

在学习键的数据表示上，采用了基于Glow的思想，和上述学习原子表示的步骤相似，并且为了数据稳定性，同样引入了Glow模型中的$1 \times 1$卷积操作。

最后是进行化学有效性验证，遵循每个原子的价键限制，采用原子和键组合后是否符合化学上键价的约束，定义了价键约束：

$$
\begin{aligned}
\sum_{c,j}c \times B(c, i, j) \le \text{Valency}(\text{Atom}_i) + Ch
\end{aligned}
$$

其中，$c$为键的类型（单键，双键，三键），与其他的模型不同，加入了形式电荷$Ch$的约束，这种效应可能为带电原子引入额外的键。例如，铵[NH4]+的N可能具有4个键，而不是3。类似地，S+和O+的可能具有3个键而不是2。

模型结构如图所示：
<center class ='img'>
<img title="MoFlow的模型结构图" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/moflow.jpg" width="80%"></br>MoFlow的模型结构图
</center>

## 3. 模型的实现

接下来开始讲解如何基于 PaddleScience 代码，来实现药物分子中结构重构的模型复，为实现moflow的模型构建，训练、推理以及评估，接下来仅对模型构建、训练、测试、评估等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据处理

在数据处理中，首先通过读取化学分子结构，采用化学分子库的处理，将数据集中的化学结构部分进行化学键和分子节点的提取，对原子结构和键值进行处理，用 PaddleScience 代码表示如下

``` py linenums="394" title="data/dataset/moflow_dataset.py"
--8<--
ppsci/data/dataset/moflow_dataset.py:394:427
--8<--
```

训练数据采用集合标签进行选择，将数据集分成训练数据和测试数据，其中qm9和zinc250k数据集处理一致，对于其特征和原子结构的处理选择上有些差异。

### 3.2 约束构建

本案例基于数据从中学习化学键约束的方法求解问题，因此按照 PaddleScience 的API结构说明，采用内置的 SupervisedConstraint 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

``` py linenums="140" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:140:161
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `MOlFLOWDataset`，"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，设置的 `batch_size` 为 256，`num_works` 为 8。

定义监督约束的代码如下：

``` py linenums="167" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:167:177
--8<--
```

### 3.3 模型构建

在该案例中，药物分子预测生成模型基于 MoFlowNet 网络模型实现，结合 PaddleScience 代码标准格式，对于模型进行分装，单独对flow，grow等模模型进行调用，其中模型构成的代码表示如下：

``` py linenums="162" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:162:166
--8<--
```

模型网络参数配置如下：

``` py linenums="97" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:97:128
--8<--
```

参数通过配置文件进行设置如下：

``` py linenums="22" title="examples/moflow/conf/moflow_train.yaml"
--8<--
examples/moflow/conf/moflow_train.yaml:22:79
--8<--
```

其中，data_name表示数据集的选择，选择之后对应选择不同数据集对应的网络参数部分，input_keys 和 output_keys 分别代表网络模型输入、输出变量的名称,hyper_params代表不同数据集对应的网络参数，在数据集选择之后模型构建中会进行更新，方便不同数据集下模型的统一构建。使用模型自定义的损失函数模型的训练。

### 3.4 学习率与优化器构建

本案例中使用的学习率大小设置为 `0.001`。优化器使用 `Adam`,用 PaddleScience 代码表示如下：

``` py linenums="181" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:181:183
--8<--
```

### 3.5 评估器构建

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况，需要使用 `SupervisedValidator` 构建评估器。代码如下：

``` py linenums="184" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:184:213
--8<--
```

评价指标 `metric`，使用自定义函数，通过使用分子向量值进行分子生成，对重新生成分子进行单独的评估，在这里使用了自定义的评价指标分别是 `valid`、`unique`、`abs_unique`

### 3.6 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 ppsci.solver.Solver，然后启动训练、评估

``` py linenums="214" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:214:236
--8<--
```

### 3.7 模型的生成评估构建

针对不同数据集构建，提供了不同模型的不同方式评估，通过重建，随机生成，插值生成的方式对于模型生成能力进行全面评估，不同方式参数配置不同，参数配置文件如下所示：

``` py linenums="81" title="examples/moflow/conf/moflow_test.yaml"
--8<--
examples/moflow/conf/moflow_test.yaml:81:126
--8<--
```

其中，EVAL_mode为选择的评估模式，不同的模式评估的方式不同，Reconstruct（重建生成）针对不同的数据集进行药物分子的重建生成，在选定的数据集中的分子进行重建生成；Random（随机生成）针对不同的数据集从潜在空间进行随机生成，参数设置是从10000个样本中随机生成5次；Inter2point（分子间插值生成）在潜在空间进行插值，两个分子之间插值可视化生成分子图；Intergrid（分子网格插值生成）在潜在空间进行插值，分子网格进行可视化生成分子图（插值生成将生成的新分子可视化存储图片）。每种模式下参数根据实际情况进行调整，其中包括结果存储，生成的分子的数量等，其余配置与训练一样，在选择不同数据集训练的模型注意修改数据名称，核对预训练模型的位置。

构建评估器的代码为：

``` py linenums="368" title="examples/moflow/test_generate.py"
--8<--
examples/moflow/test_generate.py:368:529
--8<--
```

### 3.8 模型的优化构建

在模型完成训练后，进行分子优化和约束优化，训练一个额外的MLP模型，从潜空间到QED属性或者plogp属性，得到优化后的分子属性，并进行评估，受到与属性相似性的约束。如果首次运行的时候，会对选择的预训练模型进行优化训练，不同的属性存储的优化模型不同，QED属性保存为前缀为qed，plogp属性保存前缀为plogp，相关优优化模型将保存到指定文件夹下，第二次运行时将对优化后的模型进行评估。代码如下：

``` py linenums="470" title="examples/moflow/optimize_moflow.py"
--8<--
examples/moflow/optimize_moflow.py:470:586
--8<--
```

主要参数与训练相似，需要单独进行评估参数的设置，参数通过配置文件进行设置如下：

``` py linenums="94" title="examples/moflow/conf/moflow_optimize.yaml"
--8<--
examples/moflow/conf/moflow_optimize.yaml:94:107
--8<--
```

## 4. 训练完整代码

``` py linenums="1" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py
--8<--
```

## 5.参考文献
【文章】[MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://arxiv.org/abs/2006.10137v1)

【Code】[Moflow](https://github.com/calvin-zcx/moflow)
