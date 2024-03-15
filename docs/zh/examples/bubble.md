# Bubble_flow

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat --output bubble.mat
    python bubble.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat --output bubble.mat
    python bubble.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/bubble/bubble_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [bubble_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/bubble/bubble_pretrained.pdparams) | loss(bubble_mse): 0.00558<br>MSE.u(bubble_mse): 0.00090<br>MSE.v(bubble_mse): 0.00322<br>MSE.p(bubble_mse): 0.00066<br>MSE.phil(bubble_mse): 0.00079 |

## 1. 背景简介

### 1.1 气泡流

气泡流是多相流的典型代表。而多相流是研究两种或以上不同相态或不同组分的物质共存并有明确分界面的多相流体流动力学、热力学、传热传质学、燃烧学、化学和生物反应以及相关工业过程中的共性科学问题。这是一门从传统能源转化与利用领域逐渐发展起来的新兴交叉科学，涉及到能源、动力、核反应堆、化工、石油、制冷、低温、可再生能源开发利用、航空航天、环境保护、生命科学等许多领域，在国民经济的基础与支柱产业及国防科学技术发展中有不可替代的巨大作用。在多相流中，各相之间有明显的界面，并且各自保持相对独立的物质特性，如气-液、气-固、液-液、液-固等。本文我们主要研究气泡流，当然本文介绍的部分物理信息神经网络方法（Semi-physics-informed neural networks，Semi-PINNs）同样适用于多相流问题。

气泡流是一种流体力学现象，发生在气液两相混合物在管中受到力的作用而流动且混合物中含气量较低的情况下。此时，气相以分散的小气泡分布于液相中，在管子中央的气泡较多，靠近管壁的气泡较少。这些小气泡近似球形，并且其运动速度大于液体流速，这种流态被称为气泡流。气泡流已广泛应用于各种领域，例如

- 化工过程：在化工过程中，气泡流常常发生在液体中存在大量气体的场合，例如在气体吸收、解吸、萃取、乳化等过程中。气泡流的特性对于工艺流程的优化和设备的选择具有重要的影响。
- 生物医学：在生物医学领域，气泡流常常被用于药物的传递、细胞的分离、生物反应器的设计等。例如，通过控制气泡的大小和流速，可以将药物精确地传递到目标部位。
- 环境工程：在环境工程中，气泡流可用于水处理、废水处理、气体净化等过程中。例如，通过气泡流可以将氧气引入污水中，促进微生物的生长，从而加速有机废物的分解。
- 食品工业：在食品工业中，气泡流也具有广泛的应用。例如，在制作面包、蛋糕等食品时，通过气泡流的特性可以控制面团的发酵过程，从而获得更好的口感和质地。
- 航空航天：在航空航天领域，气泡流的研究可以帮助设计师更好地理解飞机、火箭等复杂流体动力学系统的流动特性，从而优化设计，提高性能。
- 石油工业：在石油工业中，气泡流常常出现在采油、输油、炼油等过程中。通过气泡流可以增加油水界面的张力，提高采油效率。

气泡流的研究和应用对于许多领域都具有重要的意义，不仅有助于我们深入理解流体动力学的基本原理，还可以为实际生产和工程应用提供有益的指导和帮助。同时由于气泡流是一种具有高密度梯度的经典流体力学问题，因此经常被用于测试算法的有效性。

### 1.2 Semi-PINNs方法

物理信息神经网络（Physics-informed Neural Networks，PINNs）是一种基于神经网络的物理模型，旨在解决有监督学习任务，同时尊重由非线性偏微分方程描述的物理规律。这种网络不仅学习到训练数据样本的分布规律，还能学习到数学方程描述的物理定律。

PINNs 的背景源于对数据驱动方法和物理模型的结合。在许多科学和工程应用中，由于训练数据的采集难度高和复杂性，纯数据驱动方法往往难以取得理想的效果。同时，传统的物理模型虽然能够准确地描述自然现象，但在某些情况下可能无法充分利用所有可用的数据。因此，PINN作为一种结合了数据驱动和物理模型的方法，旨在利用两者的优势，提高预测的准确性和泛化能力。

PINNs 的原理是将物理方程作为正则器，以神经网络作为求解器，将神经网络预测的结果与实际观测数据进行比较，并通过反向传播算法更新神经网络的权重，以减小预测误差。这种方法在训练过程中考虑了物理约束，从而能够更准确地捕捉系统的动态行为。

尽管PINN具有许多优点，如能够处理高维数据和解决反问题等，但它仍然存在一些局限性，例如损失函数中物理方程的考虑通常需要物理量的高阶微分。特别是在两相流中，不同流体界面处的相位值呈现出从 0 到 1 的剧烈变化，使得梯度的计算变得非常困难。 因此，对于具有高梯度的变量，高分辨率训练数据将是算法成功的先决条件，然而这将大大增加深度学习的计算量，同时许多实验也很难获得高分辨率的数据。

为此，我们不采用完整的流体动力学方程来监督训练气泡流动的过程，而是基于部分物理信息以获得令人满意的结果。具体来说，我们仅将流体连续性条件（ $\nabla \mathbf{u} =0$ ）和压力泊松方程（表示为 $\mathcal{P}$⁠ ) 代入损失函数，可以将其描述为具有部分物理信息的神经网络—— Semi-PINNs。

## 2. 问题定义

### 2.1 气泡流控制方程

气泡流模型一般由 Navier–Stokes 方程进行描述，

$$
\begin{cases}
\begin{aligned}
  &\nabla \cdot \mathbf{u}=0, \\
  &\rho\left(\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u} \cdot \nabla) \mathbf{u}\right)=-\nabla p+\mu \nabla^2 \mathbf{u},
\end{aligned}
\end{cases}
$$

其中 $\rho$ 是流体密度，$\mathbf{u} = ( u , v )$ 是二维速度矢量，$p$ 是流体压力，$\mu$ 是动力粘度。空气和水之间的界面由某个水平集或全局定义函数的等值线表示，即定义二维空间中的水平集函数 $\phi = \phi ( x , y , t )$。对于水来说 $\phi=0$，对于空气来说 $\phi=1$。在两者界面附近，有从 0 到 1 的光滑过渡，同时设界面的水平集 $\phi= 0.5$。

设 $\rho_l$ 为水（液体）密度，$\rho_g$ 为空气（气体）密度，$\mu_l$ 为水粘度，$\mu_g$ 为空气粘度，则流体中的密度和粘度可以通过水平集函数表示为：

$$\begin{aligned}
\rho & =\rho_l+\phi\left(\rho_g-\rho_l\right), \\
\mu & =\mu_l+\phi\left(\mu_g-\mu_l\right) .
\end{aligned}$$

为了模拟液体和气体之间的界面，水平集函数 $\phi = \phi ( x , y , t )$ 定义为

$$\frac{\partial \phi}{\partial t}+\mathbf{u} \cdot \nabla \phi=\gamma \nabla \cdot\left(\epsilon_{l s} \nabla \phi-\phi(1-\phi) \frac{\nabla \phi}{|\nabla \phi|}\right),$$

其中等号左端描述了界面的运动，而右端旨在保持界面紧凑，使数值稳定。$\gamma$ 是初始化参数，决定了水平集函数的重新初始化或稳定量，$\epsilon_{l s}$ 是控制界面厚度的参数，它等于网格最大尺寸。

### 2.2 BubbleNet（Semi-PINNs方法）

气泡流问题的研究通常可以分为单气泡流(图 A )和多气泡流(图 B )。

<figure markdown>
  ![bubble.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/BubbleNet/bubble.jpeg){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

本文我们主要考虑单气泡流(图 A )，当然对于多气泡流问题同样适用。对于单气泡情况，气泡初始直径设置为 $d =  4~μm$，微通道长度为 $15~μm$，宽度为 $5~μm$。沿轴向施加压力差 $\Delta p = 10~Pa$ 来驱动气泡流动，通道末端的压力保持为恒定压力 $p_0 = 799.932~Pa(6~mmHg)$，对应于人脑和淋巴液流动中间质液的压力。初始条件 (IC) 即设置为 $p=p_0$，室温为 $293.15~K$，如图 A 所示。该数值设置旨在模拟脑血管中的气泡传输，以研究血脑屏障。同时我们设 $\gamma=1$ 和 $\epsilon_{l s}=0.430$。

本文的算法 BubbleNet 的主要内容如下：

- 采用时间离散归一化（TDN），即

$$\mathcal{W}=\frac{\mathcal{U}-\mathcal{U}_{\min }}{\mathcal{U}_{\max }-\mathcal{U}_{\min }},$$

其中 $\mathcal{U}=(u, v, p, \phi)$ 是从气泡流模拟中获得的粗化数据，$\mathcal{U}_{\min },~\mathcal{U}_{\max }$ 分别为每个时间步粗化CFD数据的最大值和最小值。如此处理的原因在于流场中的物理量变化较大，TDN 将有助于消除物理量变化造成的不准确性。

- 引入流函数 $\psi$ 用于预测速度场 $u$、$v$，即

$$u=\frac{\partial \psi}{\partial y},\quad  v=-\frac{\partial \psi}{\partial x},$$

流函数的引入避免了损失函数中速度向量的梯度计算，提高了神经网络的效率。

- 引入压力泊松方程，以提高预测的准确性，即对动量方程等号两端同时求散度：

$$\nabla^2 p=\rho \frac{\nabla \cdot \mathbf{u}}{\Delta t}-\rho \nabla \cdot(\mathbf{u} \cdot \nabla \mathbf{u})+\mu \nabla^2(\nabla \cdot \mathbf{u}) = -\rho \nabla \cdot(\mathbf{u} \cdot \nabla \mathbf{u}).$$

- 使用均方误差（MSE）来计算损失函数中预测和训练数据的偏差，损失函数表示为

$$\mathcal{L}=\frac{1}{m} \sum_{i=1}^m\left(\mathcal{W}_{\text {pred }(i)}-\mathcal{W}_{\text {train }(i)}\right)^2+\frac{1}{m} \sum_{i=1}^m\left(\nabla^2 p_{(i)}\right)^2,$$

其中 $\mathcal{W}=(u, v, p, \phi)$ 表示归一化后的数据集。

对于单气泡流问题，本文提出的 BubbleNet（Semi-PINNs方法）的模型结构图为：

<figure markdown>
  ![pinns.jpeg](https://paddle-org.bj.bcebos.com/paddlescience/docs/BubbleNet/pinns.jpeg){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

其中我们使用三个子网：$Net_{\psi},~Net_p$ 和 $Net_{\phi}$ 来分别预测 $\psi$、$p$ 和 $\phi$，并通过对 $\psi$ 自动微分来计算速度 $u,~v$。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该单气泡流问题。为了快速理解 PaddleScience，接下来仅对模型构建、约束构建等关键步骤进行阐述，而其余细节请参考[API文档](../api/arch.md)。

### 3.1 数据处理

数据集是通过在细网格下的 CFD 结果获得的，包含未归一化的 $x,~y,~t,~u,~v,~p,~\phi$，以字典的形式存储在 `.mat` 文件中。运行本问题代码前请下载[数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat)。

下载后，我们需要首先对数据集进行时间离散归一化（TDN）处理，同时构造训练集和验证集。

``` py linenums="40"
--8<--
examples/bubble/bubble.py:40:84
--8<--
```

### 3.2 模型构建

在气泡流问题中，每一个已知的坐标点 $(t, x, y)$ 都有自身的流函数 $\psi$、压力 $p$ 和 水平集函数 $\phi$ 三个待求解的未知量，我们在这里使用 3 个并行的的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(t, x, y)$ 分别到 $(\psi, p, \phi)$ 的映射函数 $f_i: \mathbb{R}^3 \to \mathbb{R}$，即：

$$
\begin{aligned}
\psi &= f_1(t, x, y),\\
p &= f_2(t, x, y),\\
\phi &= f_3(t, x, y).
\end{aligned}
$$

上式中 $f_1,f_2,f_3$ 均为 MLP 模型，用 PaddleScience 代码表示如下

``` py linenums="86"
--8<--
examples/bubble/bubble.py:86:89
--8<--
```

使用  `transform_out` 函数实现流函数 $\psi$ 到速度 $u,~v$ 的变换，代码如下

``` py linenums="91"
--8<--
examples/bubble/bubble.py:91:98
--8<--
```

同时需要对模型 `model_psi` 注册相应的 transform ，然后将 3 个 MLP 模型组成 `Model_List`

``` py linenums="100"
--8<--
examples/bubble/bubble.py:100:102
--8<--
```

这样我们就实例化出了一个拥有 3 个 MLP 模型，每个 MLP 包含 9 层隐藏神经元，每层神经元数为 30，使用 "tanh" 作为激活函数，并包含输出 transform 的神经网络模型 `model_list`。

### 3.3 计算域构建

本文中单气泡流的训练区域由字典 `train_input` 储存的点云构成，因此可以直接使用 PaddleScience 内置的点云几何 `PointCloud` 读入数据，组合成时间-空间的计算域。

同时构造可视化区域，即以 [0, 0], [15, 5] 为对角线的二维矩形区域，且时间域为 126 个时刻 [1, 2,..., 125, 126]，该区域可以直接使用 PaddleScience 内置的空间几何 `Rectangle` 和时间域 `TimeDomain`，组合成时间-空间的 `TimeXGeometry` 计算域。代码如下

``` py linenums="104"
--8<--
examples/bubble/bubble.py:104:116
--8<--
```

???+ tip "提示"

    `Rectangle` 和 `TimeDomain` 是两种可以单独使用的 `Geometry` 派生类。

    如输入数据只来自于二维矩形几何域，则可以直接使用 `ppsci.geometry.Rectangle(...)` 创建空间几何域对象；

    如输入数据只来自一维时间域，则可以直接使用 `ppsci.geometry.TimeDomain(...)` 构建时间域对象。

### 3.4 约束构建

根据 [2.2 BubbleNet（Semi-PINNs方法）](#22-bubblenetsemi-pinns) 中定义的损失函数表达式，对应了在计算域中指导模型训练的两个约束条件，接下来使用 PaddleScience 内置的 `InteriorConstraint` 和 `SupervisedConstraint` 构建上述两种约束条件。

#### 3.4.1 内部点约束

我们以内部点约束 `InteriorConstraint` 来实现在损失函数中加入压力泊松方程的约束，代码如下：

``` py linenums="121"
--8<--
examples/bubble/bubble.py:121:136
--8<--
```

`InteriorConstraint` 的第一个参数是方程表达式，用于计算方程结果，此处计算损失函数表达式第二项中 $\nabla^2 p_{(i)}$；

第二个参数是约束变量的目标值，在本问题中我们希望 $\nabla^2 p_{(i)}$ 的结果被优化至 0，因此将目标值设为 0；

第三个参数是约束方程作用的计算域，此处填入在 [3.3 计算域构建](#33) 章节中实例化好的 `geom["time_rect"]` 即可；

第四个参数是在计算域上的采样配置，此处我们使用全量数据点训练，因此 `dataset` 字段设置为 "IterableNamedArrayDataset" 且 `iters_per_epoch` 也设置为 1，采样点数 `batch_size` 设为 228595；

第五个参数是损失函数，此处我们选用常用的MSE函数，且 `reduction` 设置为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项的平均值；

第六个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "EQ" 即可。

#### 3.4.2 监督约束

同时在训练数据上以监督学习方式进行训练，此处采用监督约束 `SupervisedConstraint`：

``` py linenums="138"
--8<--
examples/bubble/bubble.py:138:154
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `“dataset”` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `"NamedArrayDataset"` 表示分 batch 顺序读取的 `.mat` 类型的数据集；
2. `input_keys`： 输入变量名；
3. `label_keys`： 标签变量名。

"sampler" 字段定义了使用的 `Sampler` 类名为 `BatchSampler`，另外还指定了该类初始化时参数 `drop_last` 为 `False`、`shuffle` 为 `True`。

第二个参数是损失函数，此处我们选用常用的 MSE 函数，且 `reduction` 为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项求和取平均；

第三个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "Sup" 即可。

在微分方程约束和监督约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="156"
--8<--
examples/bubble/bubble.py:156:160
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数和学习率，此处我们按实验经验，使用一万轮训练轮数，评估间隔为一千轮，学习率设为 0.001。

``` yaml linenums="52"
--8<--
examples/bubble/conf/bubble.yaml:52:56
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="162"
--8<--
examples/bubble/bubble.py:162:163
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="165"
--8<--
examples/bubble/bubble.py:165:186
--8<--
```

配置与 [3.4 约束构建](#34) 的设置类似。

### 3.8 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="189"
--8<--
examples/bubble/bubble.py:189:206
--8<--
```

### 3.9 结果可视化

最后在给定的可视化区域上进行预测并可视化，可视化数据是区域内的二维点集，每个时刻 $t$ 的坐标是 $(x^t_i, y^t_i)$，对应值是 $(u^t_i, v^t_i, p^t_i,\phi^t_i)$，在此我们对预测得到的 $(u^t_i, v^t_i, p^t_i,\phi^t_i)$ 进行反归一化，我们将反归一化后的数据按时刻保存成 126 个 **vtu格式** 文件，最后用可视化软件打开查看即可。代码如下：

``` py linenums="222"
--8<--
examples/bubble/bubble.py:222:248
--8<--
```

## 4. 完整代码

``` py linenums="1" title="bubble.py"
--8<--
examples/bubble/bubble.py
--8<--
```

## 5. 结果展示

我们使用 paraview 打开保存的 126 个 **vtu格式** 文件，可以得到以下关于速度 $u,~v$，压力 $p$，以及水平集函数（气泡的形状） $\phi$ 随时间的动态变化图像

![type:video](https://www.youtube.com/embed/Ub88rh-qACI?si=hEzFu0kE4KhsWWTe)
<center>速度 u 随时间的动态变化图像</center>
![type:video](https://www.youtube.com/embed/KYcXho0DQxc?si=YwexUy4RTIZrUjOG)
<center>速度 v 随时间的动态变化图像</center>
![type:video](https://www.youtube.com/embed/VHMus9CJWfI?si=klzDydp5gbOV3ZdZ)
<center>压力 p 随时间的动态变化图像</center>
![type:video](https://www.youtube.com/embed/wtCKbsVnBGs?si=wGJlUdEU3xKe4iLr)
<center>水平集函数（气泡的形状） phi 随时间的动态变化图像</center>

从动态变化图像可以得出以下结论：

- 从水平集函数（气泡的形状） $\phi$ 随时间的动态变化图像，可以看出该模型可以很好的预测气泡在液体管中的变化过程，具有良好的精度；
- 从速度 $u,~v$  随时间的动态变化图像，可以看出该模型可以很好的预测气泡在液体管中变化时的速度大小，同时对速度的预测优于传统 DNN 方法，具体比较可以[参考文章](https://doi.org/10.1063/5.0079602)；
- 然而观察压力 $p$  随时间的动态变化图像，几乎不怎么变化， 并没有成功捕获压力场中的气泡形状特征细节，这是因为，与大压力范围(动态变化图像中压力 $p$ 的范围[800, 810] )相比，描绘气泡形状的压力大小的细微差别太小。

综上所述，物理信息与传统神经网络相结合的 Semi-PINNs 方法可以灵活地构建网络框架，获得满足工程需求的满意结果，尤其是在不易获取大量训练数据时是非常有效的。虽然编码完整的流体动力学方程的深度神经网络在预测上可能更准确，但目前的 BubbleNet 本质上是面向工程的 Semi-PINNs 方法，具有简单、计算效率和灵活性的优势。这就提出了一个值得未来继续研究的有趣问题，即我们可以通过选择性地将物理信息引入神经网络来优化网络性能，更多相关内容及结论请[参考文章](https://doi.org/10.1063/5.0079602)。

## 6. 参考资料

参考文献： [Predicting micro-bubble dynamics with semi-physics-informed deep learning](https://doi.org/10.1063/5.0079602)

参考代码： [BubbleNet(Semi-PINNs)](https://github.com/hanfengzhai/BubbleNet)
