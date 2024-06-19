# 2D-LDC(2D Lid Driven Cavity Flow)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6137973" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "Re=1000"

    === "模型训练命令"

        ``` sh
        # linux
        wget -nc -P ./data/ \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat --create-dirs -o ./data/ldc_Re100.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat --create-dirs -o ./data/ldc_Re400.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_sota.py
        ```

    === "模型评估命令"

        ``` sh
        # linux
        wget -nc -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        python ldc_2d_Re3200_sota.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re1000_sota_pretrained.pdparams
        ```

    === "模型导出命令"

        ``` sh
        python ldc_2d_Re3200_sota.py mode=export
        ```

    === "模型推理命令"

        ``` sh
        # linux
        wget -nc -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        python ldc_2d_Re3200_sota.py mode=infer
        ```

    | 预训练模型  | $Re$  | 指标 |
    | :-- | :-- | :-- |
    | - | 100 | U_validator/loss: 0.00017<br>U_validator/L2Rel.U: 0.04875 |
    | - | 400 | U_validator/loss: 0.00047<br>U_validator/L2Rel.U: 0.07554 |
    | [**ldc_re1000_sota_pretrained.pdparams**](https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re1000_sota_pretrained.pdparams) | 1000 | **U_validator/loss: 0.00053<br>U_validator/L2Rel.U: 0.07777** |
    | - | 3200 | U_validator/loss: 0.00227<br>U_validator/L2Rel.U: 0.15440 |

=== "Re=3200"

    === "模型训练命令"

        ``` sh
        # linux
        wget -nc -P ./data/ \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1600.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat --create-dirs -o ./data/ldc_Re100.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat --create-dirs -o ./data/ldc_Re400.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1600.mat --create-dirs -o ./data/ldc_Re1600.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_piratenet.py
        ```

    === "模型评估命令"

        ``` sh
        # linux
        wget -nc -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_piratenet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re3200_piratenet_pretrained.pdparams
        ```

    === "模型导出命令"

        ``` sh
        python ldc_2d_Re3200_piratenet.py mode=export
        ```

    === "模型推理命令"

        ``` sh
        # linux
        wget -nc -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_piratenet.py mode=infer
        ```

    | 预训练模型  | $Re$  | 指标 |
    | :-- | :-- | :-- |
    | - | 100 | U_validator/loss: 0.00016<br>U_validator/L2Rel.U: 0.04741 |
    | - | 400 | U_validator/loss: 0.00071<br>U_validator/L2Rel.U: 0.09288 |
    | - | 1000 | U_validator/loss: 0.00191<br>U_validator/L2Rel.U: 0.14797 |
    | - | 1600 | U_validator/loss: 0.00276<br>U_validator/L2Rel.U: 0.17360 |
    | [**ldc_re3200_piratenet_pretrained.pdparams**](https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re3200_piratenet_pretrained.pdparams) | 3200 | **U_validator/loss: 0.00016<br>U_validator/L2Rel.U: 0.04166** |

!!! 说明

    本案例仅提供 $Re=1000/3200$ 两种情况下的预训练模型，若需要其他雷诺数下的预训练模型，请执行训练命令手动训练即可得到各雷诺数下的模型权重。

## 1. 背景简介

顶盖方腔驱动流LDC问题在许多领域中都有应用。例如，这个问题可以用于计算流体力学（CFD）领域中验证计算方法的有效性。虽然这个问题的边界条件相对简单，但是其流动特性却非常复杂。在顶盖驱动流LDC中，顶壁朝x方向以U=1的速度移动，而其他三个壁则被定义为无滑移边界条件，即速度为零。

此外，顶盖方腔驱动流LDC问题也被用于研究和预测空气动力学中的流动现象。例如，在汽车工业中，通过模拟和分析车体内部的空气流动，可以帮助优化车辆的设计和性能。

总的来说，顶盖方腔驱动流LDC问题在计算流体力学、空气动力学以及相关领域中都有广泛的应用，对于研究和预测流动现象、优化产品设计等方面都起到了重要的作用。

## 2. 问题定义

本案例假设 $Re=3200$，计算域为一个长宽均为 1 的方腔，应用以下公式进行顶盖驱动方腔流研究**稳态**流场问题：

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

**令：**

$t^* = \dfrac{L}{U_0}$

$x^*=y^* = L$

$u^*=v^* = U_0$

$p^* = \rho {U_0}^2$

**定义：**

无量纲坐标 $x：X = \dfrac{x}{x^*}$；无量纲坐标 $y：Y = \dfrac{y}{y^*}$

无量纲速度 $x：U = \dfrac{u}{u^*}$；无量纲速度 $y：V = \dfrac{v}{u^*}$

无量纲压力 $P = \dfrac{p}{p^*}$

雷诺数 $Re = \dfrac{L U_0}{\nu}$

则可获得如下无量纲Navier-Stokes方程，施加于方腔内部：

质量守恒：

$$
\dfrac{\partial U}{\partial X} + \dfrac{\partial U}{\partial Y} = 0
$$

$x$ 动量守恒：

$$
U\dfrac{\partial U}{\partial X} + V\dfrac{\partial U}{\partial Y} = -\dfrac{\partial P}{\partial X} + \dfrac{1}{Re}(\dfrac{\partial ^2 U}{\partial X^2} + \dfrac{\partial ^2 U}{\partial Y^2})
$$

$y$ 动量守恒：

$$
U\dfrac{\partial V}{\partial X} + V\dfrac{\partial V}{\partial Y} = -\dfrac{\partial P}{\partial Y} + \dfrac{1}{Re}(\dfrac{\partial ^2 V}{\partial X^2} + \dfrac{\partial ^2 V}{\partial Y^2})
$$

对于方腔边界，则需施加 Dirichlet 边界条件：

上边界：

$$
u(x, y) = 1 − \dfrac{\cosh (C_0(x − 0.5))} {\cosh (0.5C_0)} ,
$$

左边界、下边界、右边界：

$$
u=0, v=0
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 2D-LDC 问题中，每一个已知的坐标点 $(x, y)$ 都有自身的横向速度 $u$、纵向速度 $v$、压力 $p$
三个待求解的未知量，我们在这里使用适合于 PINN 任务的 PirateNet 来表示 $(x, y)$ 到 $(u, v, p)$ 的映射函数 $f: \mathbb{R}^2 \to \mathbb{R}^3$ ，即：

$$
u, v, p = f(x, y)
$$

上式中 $f$ 即为 `PirateNet` 模型本身，用 PaddleScience 代码表示如下

``` py linenums="41"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:41:42
--8<--
```

其中 `cfg.MODEL` 配置如下所示：

``` yaml linenums="38"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:38:41
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `["x", "y"]`，输出变量名是 `["u", "v", "p"]`，这些命名与后续代码保持一致。

如上所示，通过指定 `PirateNet` 的层数、神经元个数以及激活函数，我们就实例化出了一个拥有 12 层隐藏神经元，每层神经元数为 256，使用 "tanh" 作为激活函数的神经网络模型 `model`。

### 3.2 Curriculum Learning

为了加快收敛速度，我们使用 Curriculum learning 的方法来训练模型，即先训练模型在低雷诺数下，然后逐步增加雷诺数，最终达到高雷诺数下的收敛。

``` py linenums="210"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:210:211
--8<--
```

### 3.3 方程构建

由于 2D-LDC 使用的是 Navier-Stokes 方程的2维稳态形式，因此可以直接使用 PaddleScience 内置的 `NavierStokes`。

``` py linenums="88"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:88:91
--8<--
```

在课程学习的函数中，我们在实例化 `NavierStokes` 类时需指定必要的参数：动力粘度 $\nu=\frac{1}{Re}$, 流体密度 $\rho=1.0$，其中 $Re$ 是一个随着训练过程中会逐步增大的变量。

### 3.4 计算域构建

本文中 2D-LDC 问题训练、评估所需的数据，通过读取对应雷诺数的文件得到。

``` py linenums="93"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:93:103
--8<--
```

### 3.5 约束构建

根据 [2. 问题定义](#2) 得到的无量纲公式和和边界条件，对应了在计算域中指导模型训练的两个约束条件，即：

1. 施加在矩形内部点上的无量纲 Navier-Stokes 方程约束（经过简单移项）

    $$
    \dfrac{\partial U}{\partial X} + \dfrac{\partial U}{\partial Y} = 0
    $$

    $$
    U\dfrac{\partial U}{\partial X} + V\dfrac{\partial U}{\partial Y} + \dfrac{\partial P}{\partial X} - \dfrac{1}{Re}(\dfrac{\partial ^2 U}{\partial X^2} + \dfrac{\partial ^2 U}{\partial Y^2}) = 0
    $$

    $$
    U\dfrac{\partial V}{\partial X} + V\dfrac{\partial V}{\partial Y} + \dfrac{\partial P}{\partial Y} - \dfrac{1}{Re}(\dfrac{\partial ^2 V}{\partial X^2} + \dfrac{\partial ^2 V}{\partial Y^2}) = 0
    $$

    为了方便获取中间变量，`NavierStokes` 类内部将上式左侧的结果分别命名为 `continuity`, `momentum_x`, `momentum_y`。

2. 施加在矩形上、下、左、右边界上的 Dirichlet 边界条件约束

    上边界：

    $$
    u(x, y) = 1 − \dfrac{\cosh (C_0(x − 0.5))} {\cosh (0.5C_0)} ,
    $$

    左边界、下边界、右边界：

    $$
    u=0, v=0
    $$

接下来使用 PaddleScience 内置的 `SupervisedConstraint` 构建上述两种约束条件。

#### 3.5.1 内部点约束

以作用在矩形内部点上的 `SupervisedConstraint` 为例，代码如下：

``` py linenums="105"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:105:132
--8<--
```

`SupervisedConstraint` 的第一个参数是数据集配置，用于描述如何构建输入数据，此处填入输入数据和标签数据的构造函数函数 `gen_input_batch` 和 `gen_label_batch`，以及数据集的名称 `ContinuousNamedArrayDataset`；

第二个参数是约束变量的目标值，此处填入在 [3.3 方程构建](#33) 章节中实例化好的 `equation["NavierStokes"].equations`；

第三个参数是损失函数，此处我们选用常用的 `MSE` 函数，且 `reduction` 设置为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项平均；

第四个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "PDE" 即可。

#### 3.5.2 边界约束

将上边界的标签数据按照上述对应公式进行处理，其余点的标签数据设置为 0。然后继续构建方腔边界的 Dirichlet 约束，我们仍然使用 `SupervisedConstraint` 类。

``` py linenums="134"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:134:160
--8<--
```

在微分方程约束、边界约束、初值约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="161"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:161:165
--8<--
```

### 3.6 超参数设定

接下来需要在配置文件中指定训练轮数，分别在 Re=100, 400, 1000, 1600, 3200 上训练 10, 20, 50, 50, 500 轮，每轮迭代次数为 1000，

``` yaml linenums="33"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:33:35
--8<--
```

其次，设置合适的学习率衰减策略，

``` yaml linenums="52"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:52:66
--8<--
```

最后，设置训练过程中损失自动平衡策略为 `GradNorm`，

``` yaml linenums="72"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:72:75
--8<--
```

### 3.7 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="44"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:44:48
--8<--
```

### 3.8 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="167"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:167:185
--8<--
```

此处计算 $U=\sqrt{u^2+v^2}$ 的预测误差；

评价指标 `metric` 选择 `ppsci.metric.L2Rel` 即可；

其余配置与 [约束构建](#35) 的设置类似。

### 3.9 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估

``` py linenums="187"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:187:208
--8<--
```

## 4. 完整代码

``` py linenums="1" title="ldc_2d_Re3200_piratenet.py"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py
--8<--
```

## 5. 结果展示

下方展示了模型对于边长为 1 的正方形计算域的内部点进行预测的结果 $U=\sqrt{u^2+v^2}$。

=== "Re=1000"

    <figure markdown>
    ![ldc_re1000_sota_ac](https://paddle-org.bj.bcebos.com/paddlescience/docs/ldc/ldc_re1000_sota_ac.png){ loading=lazy }
    <figcaption> </figcaption>
    </figure>

    可以看到在 $Re=1000$ 下，预测结果与求解器的结果基本相同（L2 相对误差为 7.7%）。

=== "Re=3200"

    <figure markdown>
    ![ldc_re3200_piratenet_ac](https://paddle-org.bj.bcebos.com/paddlescience/docs/ldc/ldc_re3200_piratenet_ac.png){ loading=lazy }
    <figcaption></figcaption>
    </figure>

    可以看到在 $Re=3200$ 下，预测结果与求解器的结果基本相同（L2 相对误差为 4.1%）。

## 6. 参考资料

- [PIRATENETS: PHYSICS-INFORMED DEEP LEARNING WITHRESIDUAL ADAPTIVE NETWORKS](https://arxiv.org/pdf/2402.00326.pdf)
- [jaxpi LDC example](https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/ldc#readme)
