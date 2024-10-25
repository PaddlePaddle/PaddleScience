# SPINN(helmholtz3d)

<a href="https://aistudio.baidu.com/projectdetail/8219967" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    python helmholtz3d.py
    ```

=== "模型评估命令"

    ``` sh
    python helmholtz3d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/spinn/spinn_helmholtz3d_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python helmholtz3d.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    python helmholtz3d.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [spinn_helmholtz3d.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/spinn/spinn_helmholtz3d_pretrained.pdparams) | l2_err: 0.0183 <br> rmse: 0.0064 |

## 1. 背景简介

Helmholtz方程是一个重要的偏微分方程，广泛应用于物理学和工程学中，特别是在波动理论和振动问题中。它以德国物理学家赫尔曼·冯·亥姆霍兹（Hermann von Helmholtz）的名字命名。Helmholtz方程的标准形式如下：

$$
\nabla^2 u + k^2 u = q
$$

这里：

- $\nabla^2$ 是拉普拉斯算子（也称为拉普拉斯算符），在三维直角坐标系下，它的形式是：$\nabla^2 = \frac{\partial^2 }{\partial x^2} + \frac{\partial^2 }{\partial y^2} + \frac{\partial^2 }{\partial z^2}$
- $u$ 是待求解的函数，通常表示物理量的幅度，如电磁场、声压或量子波函数等。
- $k$ 是波数，定义为 $k = \frac{2\pi}{\lambda}$，其中 $\lambda$ 是波长。
- $q$ 是源项，通常表示物理量与时间、空间导数之间的相互作用。

本案例解决以下三维 Helmholtz 方程：

$$
\begin{aligned}
  & \nabla^2 u + k^2 u = q, x \in \Omega \\
  & u(x) = 0, x \in \partial \Omega \\
\end{aligned}
$$

$$
\begin{aligned}
  & \text{source term } q = -(a_1 \pi)^2 u -(a_2 \pi)^2 u -(a_3 \pi)^2 u + k^2 u \\
  & \text{where }k=1, a_1=4, a_2=4, a_3=3
\end{aligned}
$$

## 2. 问题定义

本问题的计算域在 $[-1, 1] ^3$ 一个单位正方体内，对于计算域内部点，要求满足上述 Helmholtz 方程，对于计算域边界点，要求满足 $u = 0$。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

SPINN 的模型结构设计如下：

![SPINN_structure](https://paddle-org.bj.bcebos.com/paddlescience/docs/spinn/spinn_structure.png)

在 Helmholtz 问题中，每一个已知的坐标点 $(x, y, z)$ 都有对应的待求解的未知量 $u$（此处我们用 $u$代替），在这里使用 SPINN 来表示 $(x, y, z)$ 到 $(u)$ 的映射函数 $f: \mathbb{R}^3 \to \mathbb{R}^1$ ，即：

$$
u = m(x, y, z)
$$

上式中 $m$ 即为 SPINN 模型本身，用 PaddleScience 代码表示如下

``` py linenums="99"
--8<--
examples/spinn/helmholtz3d.py:99:100
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定网络模型的输入变量名是 `("x", "y", "z")`，输出变量名是 `("u")`，这些命名与后续代码保持一致。

接着通过指定 SPINN 的层数、神经元个数，就实例化出了一个拥有 4 层全连接层，每层全连接层的神经元个数为 64 ，每一个输出变量的隐层特征维度 `r` 为 32 的神经网络模型 `model`， 并且使用 `tanh` 作为激活函数。

``` yaml linenums="38"
--8<--
examples/spinn/conf/helmholtz3d.yaml:38:45
--8<--
```

### 3.2 方程构建

Helmholtz 微分方程可以用如下代码表示：

``` py linenums="102"
--8<--
examples/spinn/helmholtz3d.py:102:104
--8<--
```

注：此处我们需要把 model 手动传递给 `equation["Helmholtz"]`，因为 `Helmholtz` 方程需要用到前向微分功能。

### 3.3 约束构建

#### 3.3.1 内部点约束

以作用在内部点上的 `SupervisedConstraint` 为例，用于生成内部点训练数据的代码如下：

``` py linenums="39"
--8<--
examples/spinn/helmholtz3d.py:39:83
--8<--
```

用于构建内部点约束的代码如下：

``` py linenums="106"
--8<--
examples/spinn/helmholtz3d.py:106:156
--8<--
```

`SupervisedConstraint` 的第一个参数是用于训练的数据配置，由于我们使用实时随机生成的数据，而不是固定数据点，因此填入自定义的输入数据/标签生成函数；

第二个参数是方程表达式，因此传入 Helmholtz 的方程对象；

第三个参数是损失函数，此处选用 `MSELoss` 即可；

第四个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "PDE" 即可。

#### 3.3.2 边值约束

第三个约束条件是边值约束，代码如下：

``` py linenums="158"
--8<--
examples/spinn/helmholtz3d.py:158:190
--8<--
```

### 3.4 超参数设定

接下来需要指定训练轮数和学习率，此处按实验经验，使用 50 轮训练轮数，每轮迭代 1000 步，0.001 的初始学习率。

``` yaml linenums="47"
--8<--
examples/spinn/conf/helmholtz3d.yaml:47:63
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="192"
--8<--
examples/spinn/helmholtz3d.py:192:196
--8<--
```

### 3.6 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="198"
--8<--
examples/spinn/helmholtz3d.py:198:227
--8<--
```

## 4. 完整代码

``` py linenums="1" title="helmholtz3d.py"
--8<--
examples/spinn/helmholtz3d.py
--8<--
```

## 5. 结果展示

在计算域上均匀采样出 $100^3$ 个点，其预测结果和解析解如下图所示。

<figure markdown>
  ![spinn_helmholtz3d.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/spinn/spinn_helmholtz3d.png){ loading=lazy }
  <figcaption> 左侧为 PaddleScience 预测结果，右侧为解析解结果</figcaption>
</figure>

本问题使用模型预测的误差为 l2_err = 0.0183，rmse = 0.0064，与解析解误差较小，基本一致。

## 6. 参考资料

- [Separable Physics-Informed Neural Networks](https://arxiv.org/pdf/2306.15969)
- [SPINN](https://github.com/stnamjef/SPINN?tab=readme-ov-file)
