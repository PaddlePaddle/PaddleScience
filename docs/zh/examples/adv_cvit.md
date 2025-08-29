# CVit(Advection)

<a href="https://aistudio.baidu.com/projectdetail/8141430" class="md-button md-button--primary" style>AI Studio快速体验</a>

!!! note

    运行模型前请在 [Zhengyu-Huang/Operator-Learning](https://github.com/Zhengyu-Huang/Operator-Learning/tree/main/data) 中下载 `adv_a0.npy` 和 `adv_aT.npy` 两个文件，并将其放在 `./examples/adv/data/` 文件夹下。

=== "模型训练命令"

    ``` sh
    python adv_cvit.py
    ```

=== "模型评估命令"

    ``` sh
    python adv_cvit.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/adv_cvit_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python adv_cvit.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    python adv_cvit.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [adv_cvit_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/adv_cvit_pretrained.pdparams) | L2 error(mean): 0.028<br>L2 error(median): 0.022<br>L2 error(max): 0.166<br>L2 error(min): 0.0015 |

## 1. 背景简介

sciml 领域所使用的模型现阶段与CV、NLP领域的先进模型有较大差别，并没有很好地利用好这些先进模型所提供的优势。因此论文作者首先提出了一个算子学习的统一视角，按照 Global conditioning 和 Local Conditioning 分别对 DeepONet、FNO、GNO 等模型进行了归纳与总结，然后基于目前广泛应用于CV、NLP领域的 Transformer 结构设计了一种 Global conditioning 的模型 CVit。相比以往的算子学习模型，参数量更小，精度更高。

模型结构如下图所示：

<img src="https://github.com/PredictiveIntelligenceLab/cvit/raw/main/figures/cvit_arch.png" alt="Cvit" width="800">

## 2. 问题定义

CVit 作为一种算子学习模型，以输入函数 $u$、函数 $s$ 的查询点 query coordinate $y$ 为输入，输出经过算子映射后的函数，在查询点 $y$ 处的函数值 $s(y)$。

本问题求解如下方程：

Formulation The 1D advection equation in $\Omega=[0,1)$ is

$$
\begin{aligned}
& \frac{\partial u}{\partial t}+c \frac{\partial u}{\partial x}=0 \quad x \in \Omega, \\
& u(0)=u_0
\end{aligned}
$$

where $c=1$ is the constant advection speed, and periodic boundary conditions are imposed. We are interested in the map from the initial $u_0$ to solution $u(\cdot, T)$ at $T=0.5$. The initial condition $u_0$ is assumed to be

$$
u_0=-1+2 \mathbb{1}\left\{\tilde{u_0} \geq 0\right\}
$$

where $\widetilde{u_0}$ a centered Gaussian

$$
\widetilde{u_0} \sim \mathbb{N}(0, \mathrm{C}) \quad \text { and } \quad \mathrm{C}=\left(-\Delta+\tau^2\right)^{-d} \text {; }
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在本问题中，对于每一个函数 $u$，其经过算子学习模型映射到 $s$ 后，在 $y$ 上都有对应的标签 $s(y)$，因此在这里使用 CVit 来表示 $(u, y)$ 到 $s(y)$ 的映射关系：

$$
s(y) = G(u)(y)
$$

上式中 $G(u)$ 即为 CVit 模型本身，用 PaddleScience 代码表示如下

``` py linenums="55"
--8<--
examples/adv/adv_cvit.py:55:56
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定网络模型的输入变量名是 `("u", "y")`，输出变量名是 `("s")`，这些命名与后续代码保持一致。

接着通过指定 CVit 的输入维度、坐标维度、输出维度、模型层数等超参数，就可实例化出了一个 `model`

``` yaml linenums="34"
--8<--
examples/adv/conf/adv_cvit.yaml:34:54
--8<--
```

### 3.2 数据准备

本问题中的数据储存在 `adv_a0.py` 和 `adv_aT.py` 文件中，将数据随机打乱后，取前 20000 个数据为训练数据，后 10000 个为测试数据。

``` py linenums="27"
--8<--
examples/adv/adv_cvit.py:27:76
--8<--
```

### 3.3 约束构建

#### 3.3.1 监督约束

在训练时，随机选取 `batch_size` 组来自 $u$ 上的数据、并同时随机选取 `query_point` 个 $y$ 坐标，如此构成了训练输入数据，标签数据则从 $s$ 中随机选取同样的 `batch_size` x `query_point` 个标签点。

``` py linenums="83"
--8<--
examples/adv/adv_cvit.py:83:115
--8<--
```

`SupervisedConstraint` 的第一个参数是用于训练的数据配置，我们使用 `ContinuousNamedArrayDataset` 作为数据集类型，并且传入自定义的`gen_input_batch_train`和`gen_label_batch_train`来完成上述的训练输入、标签样本的随机选取过程；

第二个参数是该约束的计算表达式，我们只需要计算 $s$ 即可，因此填入一个不做任何处理，直接取出模型输出结果"s"的匿名表达式；

第三个参数是损失函数，此处选用 `MSELoss` 函数；

第四个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "Sup" 即可。

### 3.4 超参数设定

接下来需要指定训练轮数和学习率，此处按实验经验，使用 200000 轮训练轮数，所以总共为，初始学习率为 0.0001，全局梯度裁剪系数为 1.0，权重衰减为 1e-5，并且每 1 轮训练过程中都进行模型平均 EMA。

``` yaml linenums="56"
--8<--
examples/adv/conf/adv_cvit.yaml:56:79
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较 `AdamW` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="117"
--8<--
examples/adv/adv_cvit.py:117:125
--8<--
```

### 3.6 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="127"
--8<--
examples/adv/adv_cvit.py:127:145
--8<--
```

## 4. 完整代码

``` py linenums="1" title="adv_cvit.py"
--8<--
examples/adv/adv_cvit.py
--8<--
```

## 5. 结果展示

在测试集上的预测结果、参考结果以及绝对值误差如下图所示。

<figure markdown>
  ![adv_cvit.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/adv_cvit.png){ loading=lazy }
  <figcaption> CVit 对信号的拟合结果（此处展示拟合结果误差较差的16.6%的结果，测试集整体的平均误差为 2.8%） </figcaption>
</figure>

## 6. 参考资料

- [Bridging Operator Learning and Conditioned Neural Fields: A Unifying Perspective](https://arxiv.org/abs/2405.13998)
- [PredictiveIntelligenceLab/cvit/adv](https://github.com/PredictiveIntelligenceLab/cvit/blob/main/adv/README.md)
- [The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks](https://arxiv.org/abs/2203.13181)
