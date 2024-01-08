# Heat_Exchanger

=== "模型训练命令"

    ``` sh
    python heat_exchanger.py
    ```

=== "模型评估命令"

    ``` sh
    python heat_exchanger.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/HEDeepONet/HEDeepONet_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [heat_exchanger_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/HEDeepONet/HEDeepONet_pretrained.pdparams) | The L2 norm error between the actual heat exchanger efficiency and the predicted heat exchanger efficiency: 0.02087<br>MSE.heat_boundary(interior_mse): 0.52005<br>MSE.cold_boundary(interior_mse): 0.16590<br>MSE.wall(interior_mse): 0.01203 |

## 1. 背景简介

### 1.1 换热器

换热器（亦称为热交换器或热交换设备）是用来使热量从热流体传递到冷流体，以满足规定的工艺要求的装置，是对流传热及热传导的一种工业应用。

在一般空调设备中都有换热器，即空调室内机和室外机的冷热排；换热器作放热用时称为“冷凝器”，作吸热用时称为“蒸发器”，冷媒在此二者的物理反应相反。所以家用空调机作为冷气机时，室内机的换热器称作蒸发器，室外机的则称为冷凝器；换做暖气机的角色时，则相反称之，如图所示为蒸发循环制冷系统。研究换热器热仿真可以为优化设计、提高性能和可靠性、节能减排以及新技术研发提供重要的参考和指导。

<figure markdown>
  ![heat_exchanger.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/heat_exchanger.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 蒸发循环制冷系统</figcaption>
</figure>

换热器在工程和科学领域具有多方面的重要性，其作用和价值主要体现在以下几个方面：

- 能源转换效率：换热器在能源转换中扮演着重要角色。通过优化热能的传递和利用，能够提高发电厂、工业生产和其他能源转换过程的效率。它们有助于将燃料中的热能转化为电能或机械能，最大限度地利用能源资源。
- 工业生产优化：在化工、石油、制药等行业中，换热器用于加热、冷却、蒸馏和蒸发等工艺。通过有效的换热器设计和运用，可以改善生产效率、控制温度和压力，提高产品质量，并且减少能源消耗。
- 温度控制与调节：换热器可以用于控制温度。在工业生产中，保持适当的温度对于反应速率、产品质量和设备寿命至关重要。换热器能够帮助调节和维持系统的温度在理想的操作范围内。
- 环境保护与可持续发展：通过提高能源转换效率和工业生产过程中的能源利用率，换热器有助于减少对自然资源的依赖，并降低对环境的负面影响。能源效率的提高也可以减少温室气体排放，有利于环境保护和可持续发展。
- 工程设计与创新：在工程设计领域，换热器的优化设计和创新推动了工程技术的发展。不断改进的换热器设计能够提高性能、减少空间占用并适应多种复杂工艺需求。

综上所述，换热器在工程和科学领域中的重要性体现在其对能源利用效率、工业生产过程优化、温度控制、环境保护和工程技术创新等方面的重要贡献。这些方面的不断改进和创新推动着工程技术的发展，有助于解决能源和环境方面的重要挑战。

## 2. 问题定义

### 2.1 问题描述

假设换热器内部流体流动是一维的，如图所示。

<figure markdown>
  ![1DHE.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/1DHE.png){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

忽略壁面的传热热阻和轴向热传导；与外界无热量交换，如图所示。则冷热流体和传热壁面三个节点的能量守恒方程分别为：

$$
\begin{aligned}
& L\left(\frac{q_m c_p}{v}\right)_{\mathrm{c}} \frac{\partial T_{\mathrm{c}}}{\partial \tau}-L\left(q_m c_p\right)_{\mathrm{c}} \frac{\partial T_{\mathrm{c}}}{\partial x}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{c}}\left(T_{\mathrm{w}}-T_{\mathrm{c}}\right), \\
& L\left(\frac{q_m c_p}{v}\right)_{\mathrm{h}} \frac{\partial T_{\mathrm{h}}}{\partial \tau}+L\left(q_m c_p\right)_{\mathrm{h}} \frac{\partial T_{\mathrm{h}}}{\partial x}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{h}}\left(T_{\mathrm{w}}-T_{\mathrm{h}}\right), \\
& \left(M c_p\right)_{\mathrm{w}} \frac{\partial T_{\mathrm{w}}}{\partial \tau}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{h}}\left(T_{\mathrm{h}}-T_{\mathrm{w}}\right)+\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{c}}\left(T_{\mathrm{c}}-T_{\mathrm{w}}\right).
\end{aligned}
$$

其中:

- $T$ 代表温度，
- $q_m$ 代表质量流量，
- $c_p$ 代表比热容，
- $v$ 代表流速，
- $L$ 代表流动长度，
- $\eta_{\mathrm{o}}$ 代表翅片表面效率，
- $\alpha$ 代表传热系数，
- $A$ 代表传热面积，
- $M$ 代表传热结构的质量，
- $\tau$ 代表对应时间，
- $x$ 代表流动方向，
- 下标 $\mathrm{h}$、$\mathrm{c}$ 和 $\mathrm{w}$ 分别表示热边流体、冷边流体和换热壁面。

换热器冷、热流体进出口参数满足能量守恒, 即:

$$
\left(q_m c_p\right)_{\mathrm{h}}\left(T_{\mathrm{h}, \text { in }}-T_{\mathrm{h}, \text { out }}\right)=\left(q_m c_p\right)_c\left(T_{\mathrm{c}, \text {out }}-T_{\mathrm{c}, \text {in }}\right).
$$

换热器效率 $\eta$ 为实际传热量与理论最大的传热量之比，即：

$$
\eta=\frac{\left(q_m c_p\right)_{\mathrm{h}}\left(T_{\mathrm{h}, \text { in }}-T_{\mathrm{h}, \text { out }}\right)}{\left(q_m c_p\right)_{\text {min }}\left(T_{\mathrm{h}, \text { in }}-T_{\mathrm{c}, \text { in }}\right)},
$$

式中，下标 $min$ 表示冷热流体热容较小值。

### 2.2 PI-DeepONet模型

PI-DeepONet模型，将 DeepONet 和 PINN 方法相结合，是一种结合了物理信息和算子学习的深度神经网络模型。这种模型可以通过控制方程的物理信息来增强 DeepONet 模型，同时可以将不同的 PDE 配置分别作为不同的分支网络的输入数据，从而可以有效地用于在各种（参数和非参数）PDE 配置下进行超快速的模型预测。

对于换热器问题，PI-DeepONet 模型可以表示为如图所示的模型结构：

<figure markdown>
  ![PI-DeepONet.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/PI-DeepONet.png){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

如图所示，我们一共使用了 2 个分支网络和一个主干网络，分支网络分别输入热边的质量流量和冷边的质量流量，主干网络输入一维坐标点坐标和时间信息。每个分支网和主干网均输出 $q$ 维特征向量，通过Hadamard（逐元素）乘积组合所有这些输出特征，然后将所得向量相加为预测温度场的标量输出。

## 3. 问题求解

接下来开始讲解如何将该问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该换热器热仿真问题。为了快速理解 PaddleScience，接下来仅对模型构建、约束构建等关键步骤进行阐述，而其余细节请参考[API文档](../api/arch.md)。

### 3.1 模型构建

在换热器热仿真问题中，每一个已知的坐标点 $(t, x)$ 和每一组热边的质量流量和冷边的质量流量 $(q_{mh}, q_{mc})$ 都对应一组热边流体的温度 $T_h$ 、冷边流体的温度 $T_c$ 和换热壁面的温度 $T_h$ 三个待求解的未知量。我们在这里使用 2 个分支网络和一个主干网络，3 个网络均为 MLP(Multilayer Perceptron, 多层感知机) 。 2 个分支网络分别表示 $(q_{mh}, q_{mc})$ 到输出函数 $(b_1，b_2)$ 的映射函数 $f_1，f_2: \mathbb{R}^2 \to \mathbb{R}^{3q}$，即：

$$
\begin{aligned}
b_1 &= f_1(q_{mh}),\\
b_2 &= f_2(q_{mc}).
\end{aligned}
$$

上式中 $f_1,f_2$ 均为 MLP 模型，$(b_1，b_2)$ 分别为两个分支网络的输出函数，$3q$ 为输出函数的维数。主干网络表示 $(t, x)$ 到输出函数 $t_0$ 的映射函数 $f_3: \mathbb{R}^2 \to \mathbb{R}^{3q}$，即：

$$
\begin{aligned}
t_0 &= f_3(t,x).
\end{aligned}
$$

上式中 $f_3$ 为 MLP 模型，$(t_0)$ 为主支网络的输出函数，$3q$ 为输出函数的维数。我们可以将两个分支网络和主干网络的输出函数 $(b_1,b_2, t_0)$ 分成3组，然后对每一组的输出函数分别进行Hadamard（逐元素）乘积再相加得到标量温度场，即：

$$
\begin{aligned}
T_h &= \sum_{i=1}^q b_1^ib_2^i t_0^i,\\
T_c &= \sum_{i=q+1}^{2q} b_1^ib_2^i t_0^i,\\
T_w &= \sum_{i=2q+1}^{3q} b_1^ib_2^i t_0^i.
\end{aligned}
$$

我们定义 PaddleScience 内置的 HEDeepONets 模型类，并调用，PaddleScience 代码表示如下

``` py linenums="33"
--8<--
examples/heat_exchanger/heat_exchanger.py:33:34
--8<--
```

这样我们就实例化出了一个拥有 3 个 MLP 模型的 HEDeepONets 模型，每个分支网络包含 9 层隐藏神经元，每层神经元数为 256，主干网络包含 6 层隐藏神经元，每层神经元数为 128，使用 "swish" 作为激活函数，并包含三个输出函数 $T_h,T_c,T_w$ 的神经网络模型 `model`。

### 3.2 计算域构建

对本文中换热器问题构造训练区域，即以 [0, 1] 的一维区域，且时间域为 21 个时刻 [0,1,2,...,21]，该区域可以直接使用 PaddleScience 内置的空间几何 `Interval` 和时间域 `TimeDomain`，组合成时间-空间的 `TimeXGeometry` 计算域。代码如下

``` py linenums="36"
--8<--
examples/heat_exchanger/heat_exchanger.py:36:43
--8<--
```

???+ tip "提示"

    `Rectangle` 和 `TimeDomain` 是两种可以单独使用的 `Geometry` 派生类。

    如输入数据只来自于二维矩形几何域，则可以直接使用 `ppsci.geometry.Rectangle(...)` 创建空间几何域对象；

    如输入数据只来自一维时间域，则可以直接使用 `ppsci.geometry.TimeDomain(...)` 构建时间域对象。

### 3.3 输入数据构建

- 通过 `TimeXGeometry` 计算域来构建输入的时间和空间均匀数据，
- 通过 `np.random.rand` 来生成 (0,2) 之间的随机数，这些随机数用于构建热边和冷边的质量流量的训练和测试数据。

对时间、空间均匀数据和热边、冷边的质量流量数据进行组合，得到最终的训练和测试输入数据。代码如下

``` py linenums="45"
--8<--
examples/heat_exchanger/heat_exchanger.py:45:63
--8<--
```

然后对训练数据按照空间坐标和时间进行分类，将训练数据和测试数据分类成左边界数据、内部数据、右边界数据以及初值数据。代码如下

``` py linenums="65"
--8<--
examples/heat_exchanger/heat_exchanger.py:65:124
--8<--
```

### 3.4 方程构建

换热器热仿真问题由 [2.1 问题描述](#21) 中描述的方程组成，这里我们定义 PaddleScience 内置的 `HeatEquation` 方程类来构建该方程。指定该类的参数均为1，代码如下

``` py linenums="126"
--8<--
examples/heat_exchanger/heat_exchanger.py:126:136
--8<--
```

### 3.5 约束构建

换热器热仿真问题由 [2.1 问题描述](#21) 中描述的方程组成，我们设置以下边界条件：

$$
\begin{aligned}
T_h(t,0) &= 10,\\
T_c(t,1) &= 1.
\end{aligned}
$$

同时，我们设置初值条件：

$$
\begin{aligned}
T_h(0,x) &= 10,\\
T_c(0,x) &= 1,\\
T_w(0,x) &= 5.5.
\end{aligned}
$$

此时我们对左边界数据、内部数据、右边界数据以及初值数据设置四个约束条件，接下来使用 PaddleScience 内置的 `SupervisedConstraint` 构建上述四种约束条件，代码如下

``` py linenums="138"
--8<--
examples/heat_exchanger/heat_exchanger.py:138:263
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `“dataset”` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `"NamedArrayDataset"` 表示分 batch 顺序读取数据；
2. `input`： 输入变量名；
3. `label`： 标签变量名；
4. `weight`： 权重大小。

"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，另外还指定了该类初始化时参数 `drop_last` 为 `False`、`shuffle` 为 `True`。

第二个参数是损失函数，此处我们选用常用的 MSE 函数，且 `reduction` 为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项求和取平均；

第三个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

在微分方程约束和监督约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="264"
--8<--
examples/heat_exchanger/heat_exchanger.py:264:270
--8<--
```

### 3.6 优化器构建

接下来我们需要指定学习率，学习率设为 0.001，训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="272"
--8<--
examples/heat_exchanger/heat_exchanger.py:272:273
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，我们使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="275"
--8<--
examples/heat_exchanger/heat_exchanger.py:275:349
--8<--
```

配置与 [3.5 约束构建](#35) 的设置类似。

### 3.8 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="351"
--8<--
examples/heat_exchanger/heat_exchanger.py:351:371
--8<--
```

### 3.9 结果可视化

最后在给定的可视化区域上进行预测并可视化，设冷边和热边的质量流量均为1，可视化数据是区域内的一维点集，每个时刻 $t$ 对应的坐标是 $x^i$，对应值是 $(T_h^{i}, T_c^i, T_w^i)$，在此我们画出 $T_h,T_c,T_w$ 随时间的变化图像。同时根据换热器效率的公式计算出换热器效率 $\eta$ ，画出换热器效率 $\eta$ 随时间的变化图像，代码如下：

``` py linenums="373"
--8<--
examples/heat_exchanger/heat_exchanger.py:373:430
--8<--
```

## 4. 完整代码

``` py linenums="1" title="heat_exchanger.py"
--8<--
examples/heat_exchanger/heat_exchanger.py
--8<--
```

## 5. 结果展示

如图所示为不同时刻热边温度、冷边温度、壁面温度 $T_h, T_c, T_w$ 随传热面积 $A$ 的变化图像以及换热器效率 $\eta$ 随时间的变化图像。

???+ info "说明"

    本案例只作为demo展示，尚未进行充分调优，下方部分展示结果可能与 OpenFOAM 存在一定差别。

<figure markdown>
  ![T_h.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/T_h.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 不同时刻热边温度 T_h 随传热面积 A 的变化图像</figcaption>
</figure>

<figure markdown>
  ![T_c.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/T_c.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 不同时刻冷边温度 T_c 随传热面积 A 的变化图像</figcaption>
</figure>

<figure markdown>
  ![T_w.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/T_w.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 不同时刻壁面温度 T_w 随传热面积 A 的变化图像</figcaption>
</figure>

<figure markdown>
  ![eta.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/eta.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 换热器效率随时间的变化图像</figcaption>
</figure>

从图中可以看出：

- 热边温度在 $A=1$ 处随时间的变化逐渐递减，冷边温度在 $A=0$ 处随时间的变化逐渐递增；
- 壁面温度在 $A=1$ 处随时间的变化逐渐递减，在 $A=0$ 处随时间的变化逐渐递增；
- 换热器效率随时间的变化逐渐递增，在 $t=21$ 时达到最大值。

同时我们可以假设热边质量流量和冷边质量流量相等，即 $q_h=q_c$，定义传热单元数：

$$
NTU = \dfrac{Ak}{(q_mc)_{min}}.
$$

对不同的传热单元数，我们可以分别计算对应的换热器效率，并画出换热器效率随传热单元数的变化图像，如图所示。

<figure markdown>
  ![eta-1.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/eta-1.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 换热器效率随传热单元数的变化图像</figcaption>
</figure>

从图中可以看出：换热器效率随传热单元数的变化逐渐递增，这也符合实际的换热器效率随传热单元数的变化规律。
