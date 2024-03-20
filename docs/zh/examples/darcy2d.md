# 2D-Darcy

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6184070?contributionType=1&sUid=438690&shared=1&ts=1684239806160" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    python darcy2d.py
    ```

=== "模型评估命令"

    ``` sh
    python darcy2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/darcy2d/darcy2d_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [darcy2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/darcy2d/darcy2d_pretrained.pdparams) | loss(Residual): 0.36500<br>MSE.poisson(Residual): 0.00006 |

## 1. 背景简介

Darcy Flow是一个基于达西定律的工具，用于计算液体的流动。在地下水模拟、水文学、水文地质学和石油工程等领域中，Darcy Flow被广泛应用。

例如，在石油工程中，Darcy Flow被用来预测和模拟石油在多孔介质中的流动。多孔介质是一种由小颗粒组成的物质，颗粒之间存在空隙。石油会填充这些空隙并在其中流动。通过Darcy Flow，工程师可以预测和控制石油的流动，从而优化石油开采和生产过程。

此外，Darcy Flow也被用于研究和预测地下水的流动。例如，在农业领域，通过模拟地下水流动可以预测灌溉对土壤水分的影响，从而优化作物灌溉计划。在城市规划和环境保护中，Darcy Flow也被用来预测和防止地下水污染。

2D-Darcy 是达西渗流（Darcy flow）的一种，流体在多孔介质中流动时，渗流速度小，流动服从达西定律，渗流速度和压力梯度之间呈线性关系，这种流动称为线性渗流。

## 2. 问题定义

假设达西流模型中，每个位置 $(x,y)$ 上的流速 $\mathbf{u}$ 和压力 $p$ 之间满足以下关系式：

$$
\begin{cases}
\begin{aligned}
  \mathbf{u}+\nabla p =& 0,(x,y) \in \Omega \\
  \nabla \cdot \mathbf{u} =& f,(x,y) \in \Omega \\
  p(x,y) =& \sin(2 \pi x )\cos(2 \pi y), (x,y) \in \partial \Omega
\end{aligned}
\end{cases}
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 darcy-2d 问题中，每一个已知的坐标点 $(x, y)$ 都有对应的待求解的未知量 $p$
，我们在这里使用比较简单的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(x, y)$ 到 $p$ 的映射函数 $f: \mathbb{R}^2 \to \mathbb{R}^1$ ，即：

$$
p = f(x, y)
$$

上式中 $f$ 即为 MLP 模型本身，用 PaddleScience 代码表示如下

``` py linenums="33"
--8<--
examples/darcy/darcy2d.py:33:34
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("x", "y")`，输出变量名是 `"p"`，这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数，我们就实例化出了一个拥有 5 层隐藏神经元，每层神经元数为 20 的神经网络模型 `model`。

### 3.2 方程构建

由于 2D-Poisson 使用的是 Poisson 方程的2维形式，因此可以直接使用 PaddleScience 内置的 `Poisson`，指定该类的参数 `dim` 为2。

``` py linenums="36"
--8<--
examples/darcy/darcy2d.py:36:37
--8<--
```

### 3.3 计算域构建

本文中 2D darcy 问题作用在以 (0.0, 0.0),  (1.0, 1.0) 为对角线的二维矩形区域，
因此可以直接使用 PaddleScience 内置的空间几何 `Rectangle` 作为计算域。

``` py linenums="39"
--8<--
examples/darcy/darcy2d.py:39:40
--8<--
```

### 3.4 约束构建

在本案例中，我们使用了两个约束条件在计算域中指导模型的训练分别是作用于采样点上的 darcy 方程约束和作用于边界点上的约束。

在定义约束之前，需要给每一种约束指定采样点个数，表示每一种约束在其对应计算域内采样数据的数量，以及通用的采样配置。

``` py linenums="42"
--8<--
examples/darcy/darcy2d.py:42:46
--8<--
```

#### 3.4.1 内部点约束

以作用在内部点上的 `InteriorConstraint` 为例，代码如下：

``` py linenums="48"
--8<--
examples/darcy/darcy2d.py:48:65
--8<--
```

`InteriorConstraint` 的第一个参数是方程表达式，用于描述如何计算约束目标，此处填入在 [3.2 方程构建](#32) 章节中实例化好的 `equation["Poisson"].equations`；

第二个参数是约束变量的目标值，在本问题中我们希望 Poisson 方程产生的结果被优化至与其标准解一致，因此将它的目标值全设为 `poisson_ref_compute_func` 产生的结果；

第三个参数是约束方程作用的计算域，此处填入在 [3.3 计算域构建](#33) 章节实例化好的 `geom["rect"]` 即可；

第四个参数是在计算域上的采样配置，此处我们使用全量数据点训练，因此 `dataset` 字段设置为 "IterableNamedArrayDataset" 且 `iters_per_epoch` 也设置为 1，采样点数 `batch_size` 设为 9801(表示99x99的采样网格)；

第五个参数是损失函数，此处我们选用常用的MSE函数，且 `reduction` 设置为 `"sum"`，即我们会将参与计算的所有数据点产生的损失项求和；

第六个参数是选择是否在计算域上进行等间隔采样，此处我们选择开启等间隔采样，这样能让训练点均匀分布在计算域上，有利于训练收敛；

第七个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "EQ" 即可。

#### 3.4.2 边界约束

同理，我们还需要构建矩形的四个边界的约束。但与构建 `InteriorConstraint` 约束不同的是，由于作用区域是边界，因此我们使用 `BoundaryConstraint` 类，代码如下：

``` py linenums="67"
--8<--
examples/darcy/darcy2d.py:67:77
--8<--
```

`BoundaryConstraint` 类第一个参数表示我们直接对网络模型的输出结果 `out["p"]` 作为程序运行时的约束对象；

第二个参数是指我们约束对象的真值如何获得，这里我们直接通过其解析解进行计算，定义解析解的代码如下：

``` py
lambda _in: np.sin(2.0 * np.pi * _in["x"]) * np.cos(2.0 * np.pi * _in["y"])
```

`BoundaryConstraint` 类其他参数的含义与 `InteriorConstraint` 基本一致，这里不再介绍。

在微分方程约束、边界约束、初值约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="78"
--8<--
examples/darcy/darcy2d.py:78:82
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数和学习率，此处我们按实验经验，使用一万轮训练轮数。

``` yaml linenums="39"
--8<--
examples/darcy/conf/darcy2d.yaml:39:47
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 OneCycle 学习率调整策略。

``` py linenums="84"
--8<--
examples/darcy/darcy2d.py:84:86
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.GeometryValidator` 构建评估器。

``` py linenums="88"
--8<--
examples/darcy/darcy2d.py:88:105
--8<--
```

### 3.8 可视化器构建

在模型评估时，如果评估结果是可以可视化的数据，我们可以选择合适的可视化器来对输出结果进行可视化。

本文中的输出数据是一个区域内的二维点集，因此我们只需要将评估的输出数据保存成 **vtu格式** 文件，最后用可视化软件打开查看即可。代码如下：

``` py linenums="106"
--8<--
examples/darcy/darcy2d.py:106:147
--8<--
```

### 3.9 模型训练、评估与可视化

#### 3.9.1 使用 Adam 训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="148"
--8<--
examples/darcy/darcy2d.py:148:169
--8<--
```

#### 3.9.2 使用 L-BFGS 微调[可选]

在使用 `Adam` 优化器训练完毕之后，我们可以将优化器更换成二阶优化器 `L-BFGS` 继续训练少量轮数（此处我们使用 `Adam` 优化轮数的 10% 即可），从而进一步提高模型精度。

``` py linenums="172"
--8<--
examples/darcy/darcy2d.py:172:198
--8<--

```

???+ tip "提示"

    在常规优化器训练完毕之后，使用 `L-BFGS` 微调少量轮数的方法，在大多数场景中都可以进一步有效提高模型精度。

## 4. 完整代码

``` py linenums="1" title="darcy2d.py"
--8<--
examples/darcy/darcy2d.py
--8<--
```

## 5. 结果展示

下方展示了模型对正方形计算域中每个点的压力$p(x,y)$、x(水平)方向流速$u(x,y)$、y(垂直)方向流速$v(x,y)$的预测结果、参考结果以及两者之差。

<figure markdown>
  ![darcy 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/Darcy2D/darcy2d_p.png){ loading=lazy }
  <figcaption>左：预测压力 p，中：参考压力 p，右：压力差</figcaption>
  ![darcy 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/Darcy2D/darcy2d_u_x.png){ loading=lazy }
  <figcaption>左：预测x方向流速 p，中：参考x方向流速 p，右：x方向流速差</figcaption>
  ![darcy 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/Darcy2D/darcy2d_u_y.png){ loading=lazy }
  <figcaption>左：预测y方向流速 p，中：参考y方向流速 p，右：y方向流速差</figcaption>
</figure>
