# LabelFree-DNN-Surrogate (Aneurysm flow & Pipe flow)

=== "模型训练命令"

    案例一：Pipe Flow

    ``` sh
    python poiseuille_flow.py
    ```

    案例二：Aneurysm Flow

    ``` sh
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/LabelFree-DNN-Surrogate/LabelFree-DNN-Surrogate_data.zip
    unzip LabelFree-DNN-Surrogate_data.zip

    python aneurysm_flow.py
    ```

=== "模型评估命令"

    案例一：Pipe Flow

    ``` sh
    python poiseuille_flow.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LabelFree-DNN-Surrogate/poiseuille_flow_pretrained.pdparams
    ```

    案例二：Aneurysm Flow

    ``` sh
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/LabelFree-DNN-Surrogate/LabelFree-DNN-Surrogate_data.zip
    unzip LabelFree-DNN-Surrogate_data.zip

    python aneurysm_flow.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LabelFree-DNN-Surrogate/aneurysm_flow.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python poiseuille_flow.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    python poiseuille_flow.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
|[aneurysm_flow.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/LabelFree-DNN-Surrogate/aneurysm_flow.pdparams)| L-2 error u : 2.548e-4 <br> L-2 error v : 7.169e-5 |

## 1. 背景简介

流体动力学问题的数值模拟主要依赖于使用多项式将控制方程在空间或/和时间上离散化为有限维代数系统。由于物理的多尺度特性和对复杂几何体进行网格划分的敏感性，这样的过程对于大多数实时应用程序（例如，临床诊断和手术计划）和多查询分析（例如，优化设计和不确定性量化）。在本文中，我们提供了一种物理约束的 DL 方法，用于在不依赖任何模拟数据的情况下对流体流动进行代理建模。 具体来说，设计了一种结构化深度神经网络 (DNN) 架构来强制执行初始条件和边界条件，并将控制偏微分方程（即 Navier-Stokes 方程）纳入 DNN的损失中以驱动训练。 对与血液动力学应用相关的许多内部流动进行了数值实验，并研究了流体特性和域几何中不确定性的前向传播。结果表明，DL 代理近似与第一原理数值模拟之间的流场和前向传播不确定性非常吻合。

## 2. 案例一：PipeFlow

### 2.1 问题定义

管道流体是一类非常常见和常用的流体系统，例如动脉中的血液或气管中的气流，一般管道流受到管道两端的压力差驱动，或者重力体积力驱动。
在心血管系统中，前者更占主导地位，因为血流主要受心脏泵送引起的压降控制。 一般来说，模拟管中的流体动力学需要用数值方法求解完整的 Navier-Stokes 方程，但如果管是直的并且具有恒定的圆形横截面，则可以获得完全发展的稳态流动的解析解，即 一个理想的基准来验证所提出方法的性能。 因此，我们首先研究二维圆管中的流动（也称为泊肃叶流）。

质量守恒：

$$
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
$$

$x$ 动量守恒：

$$
u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} + \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2})
$$

$y$ 动量守恒：

$$
u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} + \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2})
$$

我们只关注这种完全发展的流动并且在边界施加了无滑移边界条件。与传统PINNs方法不同的是，我们将无滑动边界条件通过速度函数假设的方式强制施加在边界上：
对于流体域边界和流体域内部圆周边界，则需施加 Dirichlet 边界条件：

<figure markdown>
  ![pipe](https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/pipe.png){ loading=lazy }
  <figcaption>流场示意图</figcaption>
</figure>

流体域入口边界：

$$
p=0.1
$$

流体域出口边界：

$$
p=0
$$

流体域上下边界：

$$
u=0, v=0
$$

### 2.2 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

#### 2.2.1 模型构建

在本案例中，每一个已知的坐标点和该点的动力粘性系数三元组 $(x, y, \nu)$ 都有自身的横向速度 $u$、纵向速度 $v$、压力 $p$
三个待求解的未知量，我们在这里使用比较简单的三个 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(x, y, \nu)$ 到 $(u, v, p)$ 的映射函数 $f_1, f_2, f_3: \mathbb{R}^3 \to \mathbb{R}^3$ ，即：

$$
u= transform_{output}(f_1(transform_{input}(x, y, \nu)))
$$

$$
v= transform_{output}(f_2(transform_{input}(x, y, \nu)))
$$

$$
p= transform_{output}(f_3(transform_{input}(x, y, \nu)))
$$

上式中 $f_1, f_2, f_3$ 即为 MLP 模型本身，$transform_{input}, transform_{output}$, 表示施加额外的结构化自定义层，用于施加约束和丰富输入，用 PaddleScience 代码表示如下:

``` py linenums="71"
--8<--
examples/pipe/poiseuille_flow.py:71:105
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `["x"、 "y"、 "nu"]`，输出变量名是 `["u"、 "v"、 "p"]`，这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数以及激活函数，我们就实例化出了三个拥有 3 层隐藏神经元和 1 层输出层神经元的神经网络，每层神经元数为 50，使用 "swish" 作为激活函数的神经网络模型 `model_u` `model_v` `model_p`。

#### 2.2.2 方程构建

由于本案例使用的是 Navier-Stokes 方程的2维稳态形式，因此可以直接使用 PaddleScience 内置的 `NavierStokes`。

``` py linenums="110"
--8<--
examples/pipe/poiseuille_flow.py:110:115
--8<--
```

在实例化 `NavierStokes` 类时需指定必要的参数：动力粘度 $\nu$ 为网络输出, 流体密度 $\rho=1.0$。

#### 2.2.3 计算域构建

本文中本案例的计算域和参数自变量 $\nu$ 由`numpy`随机数生成的点云构成，因此可以直接使用 PaddleScience 内置的点云几何 `PointCloud` 组合成空间的 `Geometry` 计算域。

``` py linenums="45"
--8<--
examples/pipe/poiseuille_flow.py:45:69
--8<--
```

#### 2.2.4 约束构建

根据 [2.1 问题定义](#21) 得到的公式和和边界条件，对应了在计算域中指导模型训练的几个约束条件，即：

- 施加在流体域内部点上的Navier-Stokes 方程约束

    质量守恒：

    $$
    \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
    $$

    $x$ 动量守恒：

    $$
    u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} - \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2}) = 0
    $$

    $y$ 动量守恒：

    $$
    u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} - \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2}) = 0
    $$

    为了方便获取中间变量，`NavierStokes` 类内部将上式左侧的结果分别命名为 `continuity`, `momentum_x`, `momentum_y`。

- 施加在流体域入出口、流体域上下血管壁边界的的 Dirichlet 边界条件约束。作为本文创新点之一，此案例创新性的使用了结构化边界条件，即通过网络的输出层后面，增加一层公式层，来施加边界条件（公式在边界处值为零）。避免了数据点作为边界条件无法有效约束的不足。统一使用用类函数`Transform()`进行初始化和管理。具体的推理过程为：

    流体域上下边界(血管壁)修正函数的公式形式为:

    $$
    \hat{u}(t,x,\theta;W,b) = u_{par}(t,x,\theta) + D(t,x,\theta)\tilde{u}(t,x,\theta;W,b)
    $$

    $$
    \hat{p}(t,x,\theta;W,b) = p_{par}(t,x,\theta) + D(t,x,\theta)\tilde{p}(t,x,\theta;W,b)
    $$

    其中$u_{par}$和$p_{par}$是满足边界条件和初始条件的特解，具体的修正函数带入后得到：

    $$
    \hat{u} = (\dfrac{d^2}{4} - y^2) \tilde{u}
    $$

    $$
    \hat{v} = (\dfrac{d^2}{4} - y^2) \tilde{v}
    $$

    $$
    \hat{p} = \dfrac{x - x_{in}}{x_{out} - x_{in}}p_{out} + \dfrac{x_{out} - x}{x_{out} - x_{in}}p_{in} + (x - x_{in})(x_{out} - x) \tilde{p}
    $$

接下来使用 PaddleScience 内置的 `InteriorConstraint` 和模型`Transform`自定义层，构建上述两种约束条件。

- 内部点约束

    以作用在流体域内部点上的 `InteriorConstraint` 为例，代码如下：

    ``` py linenums="122"
    --8<--
    examples/pipe/poiseuille_flow.py:122:142
    --8<--
    ```

    `InteriorConstraint` 的第一个参数是方程表达式，用于描述如何计算约束目标，此处填入在 [2.2.2 方程构建](#222) 章节中实例化好的 `equation["NavierStokes"].equations`；

    第二个参数是约束变量的目标值，在本问题中我们希望 Navier-Stokes 方程产生的三个中间结果 `continuity`, `momentum_x`, `momentum_y` 被优化至 0，因此将它们的目标值全部设为 0；

    第三个参数是约束方程作用的计算域，此处填入在 [2.2.3 计算域构建](#223) 章节实例化好的 `interior_geom` 即可；

    第四个参数是在计算域上的采样配置，此处我们使用分批次数据点训练，因此 `dataset` 字段设置为 `NamedArrayDataset` 且 `iters_per_epoch` 也设置为 1，采样点数 `batch_size` 设为 128；

    第五个参数是损失函数，此处我们选用常用的MSE函数，且 `reduction` 设置为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项求和取平均；

    第六个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "EQ" 即可。

#### 2.2.5 超参数设定

接下来我们需要指定训练轮数和学习率，使用3000轮训练轮数，学习率设为 0.005。

#### 2.2.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="107"
--8<--
examples/pipe/poiseuille_flow.py:107:108
--8<--
```

#### 2.2.7 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="144"
--8<--
examples/pipe/poiseuille_flow.py:144:156
--8<--
```

另一方面，此案例的可视化和定量评估主要依赖于：

1. 在 $x=0$ 截面速度 $u(y)$ 随 $y$ 在四种不同的动力粘性系数 ${\nu}$ 采样下的曲线和解析解的对比

2. 当我们选取截断高斯分布的动力粘性系数 ${\nu}$ 采样(均值为 $\hat{\nu} = 10^{−3}$， 方差 $\sigma_{\nu}​=2.67 \times 10^{−4}$)，中心处速度的概率密度函数和解析解对比

``` py linenums="159"
--8<--
examples/pipe/poiseuille_flow.py:159:261
--8<--
```

### 2.3 完整代码

``` py linenums="1" title="poiseuille_flow.py"
--8<--
examples/pipe/poiseuille_flow.py
--8<--
```

### 2.4 结果展示

<figure markdown>
  ![laplace 2d]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/pipe_result.png){ loading=lazy }
  <figcaption>(左)在 x=0 截面速度 u(y) 随 y 在四种不同的动力粘性系数采样下的曲线和解析解的对比 (右)当我们选取截断高斯分布的动力粘性系数 nu 采样(均值为 nu=0.001， 方差 sigma​=2.67 x 10e−4)，中心处速度的概率密度函数和解析解对比</figcaption>
</figure>

DNN代理模型的结果如左图所示，和泊肃叶流动的精确解(论文公式13)进行比较：

$$
u_a = \dfrac{\delta p}{2 \nu \rho L} + (\dfrac{d^2}{4} - y^2)
$$

公式和图片中的 $y$ 表示展向坐标，$\delta p$，从图片中我们可以观察到DNN预测的，4种不同粘度采样下的速度曲线（红色虚线），几乎完美符合解析解的速度曲线（蓝色实线），其中，4个case的雷诺数（$Re$）分别为283，121，33，3。实际上，只要雷诺数适中，DNN能精确预测任意给定动力学粘性系数的管道流。

右图展示了中心线(x方向管道中心)速度，在给定动力学粘性系数（高斯分布）下的不确定性。动力学粘性系数的高斯分布，平均值为$1e^{-3}$，方差为$2.67e^{-4}$，这样保证了动力学粘性系数是一个正随机变量。此外，这个高斯分布的区间为$(0,+\infty)$，概率密度函数为：

$$
f(\nu ; \bar{\nu}, \sigma_{\nu}) = \dfrac{\dfrac{1}{\sigma_{\nu}} N(\dfrac{(\nu - \bar{\nu})}{\sigma_{\nu}})}{1 - \phi(-\dfrac{\bar{\nu}}{\sigma_{\nu}})}
$$

更多细节请参考论文第九页

## 3. 案例二: Aneurysm Flow

### 3.1 问题定义

本文主要研究了两种类型的典型血管流（具有标准化的血管几何形状），狭窄流和动脉瘤流。
狭窄血流是指流过血管的血流，其中血管壁变窄和再扩张。 血管的这种局部限制与许多心血管疾病有关，例如动脉硬化、中风和心脏病发作 。
动脉瘤内的血管血流，即由于血管壁薄弱导致的动脉扩张，称为动脉瘤血流。 动脉瘤破裂可能导致危及生命的情况，例如，由于脑动脉瘤破裂引起的蛛网膜下腔出血 (SAH)，而血液动力学的研究可以提高诊断和对动脉瘤进展和破裂的基本了解 。

虽然现实的血管几何形状通常是不规则和复杂的，包括曲率、分叉和连接点，但这里研究理想化的狭窄和动脉瘤模型以进行概念验证。 即，狭窄血管和动脉瘤血管都被理想化为具有不同横截面半径的轴对称管，其由以下函数参数化，

质量守恒：

$$
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
$$

$x$ 动量守恒：

$$
u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} + \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2})
$$

$y$ 动量守恒：

$$
u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} + \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2})
$$

我们只关注这种完全发展的流动并且在边界施加了无滑移边界条件。与传统PINNs方法不同的是，我们将无滑动边界条件通过速度函数假设的方式强制施加在边界上：
对于流体域边界和流体域内部圆周边界，则需施加 Dirichlet 边界条件：

<figure markdown>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm.png){ loading=lazy }
  <figcaption>流场示意图</figcaption>
</figure>

流体域入口边界：

$$
p=0.1
$$

流体域出口边界：

$$
p=0
$$

流体域上下边界：

$$
u=0, v=0
$$

### 3.2 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

#### 3.2.1 模型构建

在本案例中，每一个已知的坐标点和几何放大系数 $(x, y, scale)$ 都有自身的横向速度 $u$、纵向速度 $v$、压力 $p$
三个待求解的未知量，我们在这里使用比较简单的三个 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(x, y, scale)$ 到 $(u, v, p)$ 的映射函数 $f_1, f_2, f_3: \mathbb{R}^3 \to \mathbb{R}^3$ ，即：

$$
u= transform_{output}(f_1(transform_{input}(x, y, scale)))
$$

$$
v= transform_{output}(f_2(transform_{input}(x, y, scale)))
$$

$$
p= transform_{output}(f_3(transform_{input}(x, y, scale)))
$$

上式中 $f_1, f_2, f_3$ 即为 MLP 模型本身，$transform_{input}, transform_{output}$, 表示施加额外的结构化自定义层，用于施加约束和链接输入，用 PaddleScience 代码表示如下:

``` py linenums="117"
--8<--
examples/aneurysm/aneurysm_flow.py:117:151
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `["x"、 "y"、 "scale"]`，输出变量名是 `["u"、 "v"、 "p"]`，这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数以及激活函数，我们就实例化出了三个拥有 3 层隐藏神经元和 1 层输出层神经元的神经网络，每层神经元数为 20，使用 "silu" 作为激活函数的神经网络模型 `model_1` `model_2` `model_3`。

此外，使用`kaiming normal`方法对权重和偏置初始化。

``` py linenums="106"
--8<--
examples/aneurysm/aneurysm_flow.py:106:115
--8<--
```

#### 3.2.2 方程构建

由于本案例使用的是 Navier-Stokes 方程的2维稳态形式，因此可以直接使用 PaddleScience 内置的 `NavierStokes`。

``` py linenums="172"
--8<--
examples/aneurysm/aneurysm_flow.py:172:172
--8<--
```

在实例化 `NavierStokes` 类时需指定必要的参数：动力粘度 $\nu = 0.001$, 流体密度 $\rho = 1.0$。

``` py linenums="37"
--8<--
examples/aneurysm/aneurysm_flow.py:37:41
--8<--
```

#### 3.2.3 计算域构建

本文中本案例的计算域和参数自变量$scale$由`numpy`随机数生成的点云构成，因此可以直接使用 PaddleScience 内置的点云几何 `PointCloud` 组合成空间的 `Geometry` 计算域。

``` py linenums="43"
--8<--
examples/aneurysm/aneurysm_flow.py:43:104
--8<--
```

#### 3.2.4 约束构建

根据 [3.1 问题定义](#31) 得到的公式和和边界条件，对应了在计算域中指导模型训练的几个约束条件，即：

- 施加在流体域内部点上的Navier-Stokes 方程约束

    质量守恒：

    $$
    \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
    $$

    $x$ 动量守恒：

    $$
    u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} - \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2}) = 0
    $$

    $y$ 动量守恒：

    $$
    u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} - \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2}) = 0
    $$

    为了方便获取中间变量，`NavierStokes` 类内部将上式左侧的结果分别命名为 `continuity`, `momentum_x`, `momentum_y`。

- 施加在流体域入出口、流体域上下血管壁边界的的 Dirichlet 边界条件约束。作为本文创新点之一，此案例创新性的使用了结构化边界条件，即通过网络的输出层后面，增加一层公式层，来施加边界条件（公式在边界处值为零）。避免了数据点作为边界条件无法有效约束。统一使用用类函数`Transform()`进行初始化和管理。具体的推理过程为：

    设狭窄缩放系数为$A$:

    $$
    R(x) = R_{0} - A\dfrac{1}{\sqrt{2\pi\sigma^2}}exp(-\dfrac{(x-\mu)^2}{2\sigma^2})
    $$

    $$
    d = R(x)
    $$

    具体的修正函数带入后得到：

    $$
    \hat{u} = (\dfrac{d^2}{4} - y^2) \tilde{u}
    $$

    $$
    \hat{v} = (\dfrac{d^2}{4} - y^2) \tilde{v}
    $$

    $$
    \hat{p} = \dfrac{x - x_{in}}{x_{out} - x_{in}}p_{out} + \dfrac{x_{out} - x}{x_{out} - x_{in}}p_{in} + (x - x_{in})(x_{out} - x) \tilde{p}
    $$

接下来使用 PaddleScience 内置的 `InteriorConstraint` 和模型`Transform`自定义层，构建上述两种约束条件。

- 内部点约束

    以作用在流体域内部点上的 `InteriorConstraint` 为例，代码如下：

    ``` py linenums="174"
    --8<--
    examples/aneurysm/aneurysm_flow.py:174:193
    --8<--
    ```

    `InteriorConstraint` 的第一个参数是方程表达式，用于描述如何计算约束目标，此处填入在 [3.2.2 方程构建](#322) 章节中实例化好的 `equation["NavierStokes"].equations`；

    第二个参数是约束变量的目标值，在本问题中我们希望 Navier-Stokes 方程产生的三个中间结果 `continuity`, `momentum_x`, `momentum_y` 被优化至 0，因此将它们的目标值全部设为 0；

    第三个参数是约束方程作用的计算域，此处填入在 [3.2.3 计算域构建](#323) 章节实例化好的 `interior_geom` 即可；

    第四个参数是在计算域上的采样配置，此处我们使用分批次数据点训练，因此 `dataset` 字段设置为 `NamedArrayDataset` 且 `iters_per_epoch` 也设置为 1，采样点数 `batch_size` 设为 128；

    第五个参数是损失函数，此处我们选用常用的MSE函数，且 `reduction` 设置为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项求和取平均；

    第六个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "EQ" 即可。

#### 3.2.5 超参数设定

接下来我们需要指定训练轮数和学习率，使用400轮训练轮数，学习率设为 0.005。

``` yaml linenums="33"
--8<--
examples/aneurysm/conf/aneurysm_flow.yaml:33:42
--8<--
```

#### 3.2.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="152"
--8<--
examples/aneurysm/aneurysm_flow.py:152:170
--8<--
```

#### 3.2.7 模型训练、评估与可视化(需要下载数据)

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后推理。

``` py linenums="212"
--8<--
examples/aneurysm/aneurysm_flow.py:212:212
--8<--
```

### 3.3 完整代码

``` py linenums="1" title="aneurysm_flow.py"
--8<--
examples/aneurysm/aneurysm_flow.py
--8<--
```

### 3.4 结果展示

<figure markdown>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm_result_1.png)<br>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm_result_2.png)<br>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm_result_3.png)
  <figcaption>第一行为x方向速度，第二行为y方向速度，第三行为壁面剪切应力曲线</figcaption>
</figure>

图片展示了对于几何变化的动脉瘤流动的求解能力，其中训练是通过，对几何缩放系数$A$从$0$到$-2e^{-2}$区间采样进行的。三种不同几何的流场预测如图所示，动脉瘤的大小从左到右增加，流动速度在血管扩张区域减小，在动脉瘤中心处衰减最多。从前两行图片可以看出CFD结果和模型预测结果符合较好。对于WSS壁面剪切应力，曲线随着几何的变化也被模型精确捕获。

更多细节参考论文13页。

## 4. 参考文献

参考文献： [Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data](https://arxiv.org/abs/1906.02382)

参考代码： [LabelFree-DNN-Surrogate](https://github.com/Jianxun-Wang/LabelFree-DNN-Surrogate)
