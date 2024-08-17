# Allen-Cahn

<a href="https://aistudio.baidu.com/projectdetail/7927786" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat --create-dirs -o ./dataset/allen_cahn.mat
    python allen_cahn_piratenet.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat --create-dirs -o ./dataset/allen_cahn.mat
    python allen_cahn_piratenet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python allen_cahn_piratenet.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat --create-dirs -o ./dataset/allen_cahn.mat
    python allen_cahn_piratenet.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [allen_cahn_piratenet_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_pretrained.pdparams) | L2Rel.u: 1.2e-05 |

## 1. 背景简介

Allen-Cahn 方程（有时也叫作模型方程或相场方程）是一种数学模型，通常用于描述两种不同相之间的界面演化。这个方程最早由Samuel Allen和John Cahn在1970年代提出，用以描述合金中相分离的过程。Allen-Cahn 方程是一种非线性偏微分方程，其一般形式可以写为：

$$ \frac{\partial u}{\partial t} = \varepsilon^2 \Delta u - F'(u) $$

这里：

- $u(\mathbf{x},t)$ 是一个场变量，代表某个物理量，例如合金的组分浓度或者晶体中的有序参数。
- $t$ 表示时间。
- $\mathbf{x}$ 表示空间位置。
- $\Delta$ 是Laplace算子，对应于空间变量的二阶偏导数（即 $\Delta u = \nabla^2 u$ ），用来描述空间扩散过程。
- $\varepsilon$ 是一个正的小参数，它与相界面的宽度相关。
- $F(u)$ 是一个双稳态势能函数，通常取为$F(u) = \frac{1}{4}(u^2-1)^2$，这使得 $F'(u) = u^3 - u$ 是其导数，这代表了非线性的反应项，负责驱动系统向稳定状态演化。

这个方程中的 $F'(u)$ 项使得在 $u=1$ 和 $u=-1$ 附近有两个稳定的平衡态，这对应于不同的物理相。而 $\varepsilon^2 \Delta u$ 项则描述了相界面的曲率引起的扩散效应，这导致界面趋向于减小曲率。因此，Allen-Cahn 方程描述了由于相界面曲率和势能影响而发生的相变。

在实际应用中，该方程还可能包含边界条件和初始条件，以便对特定问题进行数值模拟和分析。例如，在特定的物理问题中，可能会有 Neumann 边界条件（导数为零，表示无通量穿过边界）或 Dirichlet 边界条件（固定的边界值）。

本案例解决以下 Allen-Cahn 方程：

$$
\begin{aligned}
    & u_t - 0.0001 u_{xx} + 5 u^3 - 5 u  = 0,\quad t \in [0, 1],\ x\in[-1, 1],\\
    &u(x,0) = x^2 \cos(\pi x),\\
    &u(t, -1) = u(t, 1),\\
    &u_x(t, -1) = u_x(t, 1).
\end{aligned}
$$

## 2. 问题定义

根据上述方程，可知计算域为$[0, 1]\times [-1, 1]$，含有一个初始条件： $u(x,0) = x^2 \cos(\pi x)$，两个周期边界条件：$u(t, -1) = u(t, 1)$、$u_x(t, -1) = u_x(t, 1)$。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 Allen-Cahn 问题中，每一个已知的坐标点 $(t, x)$ 都有对应的待求解的未知量 $(u)$，
，在这里使用 PirateNet 来表示 $(t, x)$ 到 $(u)$ 的映射函数 $f: \mathbb{R}^2 \to \mathbb{R}^1$ ，即：

$$
u = f(t, x)
$$

上式中 $f$ 即为 PirateNet 模型本身，用 PaddleScience 代码表示如下

``` py linenums="63"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:63:64
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定网络模型的输入变量名是 `("t", "x")`，输出变量名是 `("u")`，这些命名与后续代码保持一致。

接着通过指定 PirateNet 的层数、神经元个数，就实例化出了一个拥有 3 个 PiraBlock，每个 PiraBlock 的隐层神经元个数为 256 的神经网络模型 `model`， 并且使用 `tanh` 作为激活函数。

``` yaml linenums="34"
--8<--
examples/allen_cahn/conf/allen_cahn_piratenet.yaml:34:40
--8<--
```

### 3.2 方程构建

Allen-Cahn 微分方程可以用如下代码表示：

``` py linenums="66"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:66:67
--8<--
```

### 3.3 计算域构建

本问题的计算域为 $[0, 1]\times [-1, 1]$，其中用于训练的数据已提前生成，保存在 `./dataset/allen_cahn.mat` 中，读取并生成计算域内的离散点。

``` py linenums="69"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:69:81
--8<--
```

### 3.4 约束构建

#### 3.4.1 内部点约束

以作用在内部点上的 `SupervisedConstraint` 为例，代码如下：

``` py linenums="94"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:94:110
--8<--
```

`SupervisedConstraint` 的第一个参数是用于训练的数据配置，由于我们使用实时随机生成的数据，而不是固定数据点，因此填入自定义的输入数据/标签生成函数；

第二个参数是方程表达式，因此传入 Allen-Cahn 的方程对象；

第三个参数是损失函数，此处选用 `CausalMSELoss` 函数，其会根据 `causal` 和 `tol` 参数，对不同的时间窗口进行重新加权， 能更好地优化瞬态问题；

第四个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "PDE" 即可。

#### 3.4.2 周期边界约束

此处我们采用 hard-constraint 的方式，在神经网络模型中，对输入数据使用cos、sin等周期函数进行周期化，从而让$u_{\theta}$在数学上直接满足方程的周期性质。
根据方程可得函数$u(t, x)$在$x$轴上的周期为 2，因此将该周期设置到模型配置里即可。

``` yaml linenums="41"
--8<--
examples/allen_cahn/conf/allen_cahn_piratenet.yaml:41:42
--8<--
```

#### 3.4.3 初值约束

第三个约束条件是初值约束，代码如下：

``` py linenums="112"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:112:125
--8<--
```

在微分方程约束、初值约束构建完毕之后，以刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="126"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:126:130
--8<--
```

### 3.5 超参数设定

接下来需要指定训练轮数和学习率，此处按实验经验，使用 300 轮训练轮数，0.001 的初始学习率。

``` yaml linenums="50"
--8<--
examples/allen_cahn/conf/allen_cahn_piratenet.yaml:50:63
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="132"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:132:136
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="138"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:138:156
--8<--
```

### 3.8 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="158"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:158:184
--8<--
```

## 4. 完整代码

``` py linenums="1" title="allen_cahn_piratenet.py"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py
--8<--
```

## 5. 结果展示

在计算域上均匀采样出 $201\times501$ 个点，其预测结果和解析解如下图所示。

<figure markdown>
  ![allen_cahn_piratenet.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/AllenCahn/allen_cahn_piratenet_ac.png){ loading=lazy }
  <figcaption> 左侧为 PaddleScience 预测结果，中间为解析解结果，右侧为两者的差值</figcaption>
</figure>

可以看到对于函数$u(t, x)$，模型的预测结果和解析解的结果基本一致。

## 6. 参考资料

- [PIRATENETS: PHYSICS-INFORMED DEEP LEARNING WITHRESIDUAL ADAPTIVE NETWORKS](https://arxiv.org/pdf/2402.00326.pdf)
- [Allen-Cahn equation](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/examples/allen_cahn/README.md)
