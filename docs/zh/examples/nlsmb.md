# NLS-MB

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

=== "模型训练命令"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py
    # rogue wave
    python NLS-MB_optical_rogue_wave.py
    ```

=== "模型评估命令"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_soliton_pretrained.pdparams
    # rogue wave
    python NLS-MB_optical_rogue_wave.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_rogue_wave_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py mode=export
    # rogue wave
    python NLS-MB_optical_rogue_wave.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py mode=infer
    # rogue wave
    python NLS-MB_optical_rogue_wave.py mode=infer

    ```

| 预训练模型  | 指标 |
|:--| :--|
| [NLS-MB_soliton_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_soliton_pretrained.pdparams) | Residual/loss: 0.00000<br>Residual/MSE.Schrodinger_1: 0.00000<br>Residual/MSE.Schrodinger_2: 0.00000<br>Residual/MSE.Maxwell_1: 0.00000<br>Residual/MSE.Maxwell_2: 0.00000<br>Residual/MSE.Bloch: 0.00000 |
| [NLS-MB_optical_rogue_wave.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_optical_rogue_wave.pdparams) | Residual/loss: 0.00001<br>Residual/MSE.Schrodinger_1: 0.00000<br>Residual/MSE.Schrodinger_2: 0.00000<br>Residual/MSE.Maxwell_1: 0.00000<br>Residual/MSE.Maxwell_2: 0.00000<br>Residual/MSE.Bloch: 0.00000 |

## 1. 背景简介

非线性局域波动力学，作为非线性科学的重要分支，涵盖了孤子、呼吸子和怪波等基本形式的非线性局域波。激光锁模技术为这些理论预言的非线性局域波提供了实验验证的平台，人们通过此技术观察到了孤子分子和怪波等丰富的非线性现象，进一步推动了非线性局域波的研究。目前，该领域的研究已深入流体力学、非线性光学、玻色-爱因斯坦凝聚(BEC)、等离子体物理等多个物理领域。在光纤领域，非线性动力学的研究基于光纤的光学器件、信息处理、材料设计以及信号传输的原理，对光纤激光器、放大器、波导和通信技术的发展起到了关键作用。光脉冲在光纤中的传播动力学受非线性偏微分方程（如非线性薛定谔方程NLSE）的调控。当色散与非线性效应共存时，这些方程往往难以解析求解。因此，分步傅立叶方法及其改进版本被广泛应用于研究光纤中的非线性效应，其优势在于实现简单且具有较高的相对精度。然而，对于长距离且高度非线性的场景，为满足精度需求，必须大幅减少分步傅立叶方法的步长，这无疑增加了计算复杂性，导致时域中网格点集数量庞大，计算过程耗时较长。PINN比数据驱动的方法在数据少得多的情况下表现出更好的性能，并且计算复杂性（以倍数表示）通常比SFM低两个数量级。

## 2. 问题定义

在掺铒光纤中，光脉冲的传播性质可以由耦合NLS-MB方程来描述，其形式为

$$
\begin{cases}
   \dfrac{\partial E}{\partial x} = i \alpha_1 \dfrac{\partial^2 E}{\partial t ^2} - i \alpha_2 |E|^2 E+2 p \\
   \dfrac{\partial p}{\partial t} = 2 i \omega_0 p+2 E \eta \\
   \dfrac{\partial \eta}{\partial t} = -(E p^* + E^* p)
\end{cases}
$$

其中，*x*, *t*分别表示归一化的传播距离和时间，复包络*E*是慢变的电场，*p*是共振介质偏振的量度，$\eta$表示粒子数反转的程度，符号*表示复共轭。$\alpha_1$是群速度色散参数，$\alpha_2$​​是Kerr非线性参数，是测量共振频率的偏移。NLS-MB系统是由Maimistov和Manykin首次提出来的,用来描述极短的脉冲在Kerr非线性介质中的传播.该系统在解决光纤损耗使得其传输距离受限这一问题上,也扮演着重要的作用。在这个方程中，它描述的是自感应透明孤子和NLS孤子的混合状态，称作SIT-NLS孤子，这两种孤子可以共存，并且已经有很多关于其在光纤通信中的研究.

### 2.1 Optical soliton

在光纤的反常色散区，由于色散和非线性效应的相互作用，可产生一种非常引人注目的现象——光孤子。“孤子”(soliton)是一种特殊的波包，它可以传输很长距离而不变形孤子在物理学的许多分支已得到广的研究，本案例讨论的光纤中的孤子不仅具有基础理论研究价值，而且在光纤通信方面也有实际应用。

$$
\begin{gathered}
  E(x,t) = \frac{{2\exp ( - 2it)}}{{\cosh (2t + 6x)}},  \\
  p(x,t) = \frac{{\exp ( - 2it)\left\{ {\exp ( - 2t - 6x) - \exp (2t + 6x)} \right\}}}{{\cosh {{(2t + 6x)}^2}}},  \\
  \eta (x,t) = \frac{{\cosh {{(2t + 6x)}^2} - 2}}{{\cosh {{(2t + 6x)}^2}}}.
\end{gathered}
$$

我们考虑计算域为 $[−1, 1] × [−1, 1]$。 我们首先确定优化策略。 每个边界上有 $200$ 个点,即 $N_b = 2 × 200$。为了计算 NLS-MB 的方程损失,在域内随机选择 $20,000$ 个点。

### 2.2 Optical rogue wave

光学怪波（Optical rogue waves）是光学中的一种现象，类似于海洋中的孤立海浪，但在光学系统中。它们是突然出现并且幅度异常高的光波，光学孤立子波有一些潜在的应用，尤其是在光通信和激光技术领域。一些研究表明，它们可以用于增强光信号的传输和处理，或者用于产生超短脉冲激光。
我们考虑计算域为 $[−0.5, 0.5] × [−2.5, 2.5]$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

本文使用PINN经典的MLP模型进行训练。

``` py linenums="94"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:94:95
--8<--
```

### 3.2 方程构建

由于 Optical soliton 使用的是 NLS-MB 方程，因此可以直接使用 PaddleScience 内置的 `NLSMB`。

``` py linenums="97"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:97:100
--8<--
```

### 3.3 计算域构建

本文中 Optical soliton 问题作用在以空间(-1.0, 1.0),  时间(-1.0, 1.0) 的时空区域，
因此可以直接使用 PaddleScience 内置的时空几何 `time_interval` 作为计算域。

``` py linenums="108"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:108:114
--8<--
```

### 3.4 约束构建

因数据集为解析解,我们先构造解析解函数

``` py linenums="26"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:26:44
--8<--
```

#### 3.4.1 内部点约束

以作用在内部点上的 `InteriorConstraint` 为例，代码如下：

``` py linenums="150"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:150:169
--8<--
```

`InteriorConstraint` 的第一个参数是方程（组）表达式，用于描述如何计算约束目标，此处填入在 [3.2 方程构建](#32) 章节中实例化好的 `equation["NLS-MB"].equations`；

第二个参数是约束变量的目标值，在本问题中希望 NLS-MB 每个方程均被优化至 0；

第三个参数是约束方程作用的计算域，此处填入在 [3.3 计算域构建](#33) 章节实例化好的 `geom["time_interval"]` 即可；

第四个参数是在计算域上的采样配置，此处设置 `batch_size` 为 `20000`。

第五个参数是损失函数，此处选用常用的 MSE 函数，且 `reduction` 设置为 `"mean"`，即会将参与计算的所有数据点的均方误差；

第六个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "EQ" 即可。

#### 3.4.2 边界约束

由于我们边界点和初值点具有解析解,因此我们使用监督约束

``` py linenums="171"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:171:176
--8<--
```

### 3.5 超参数设定

接下来需要指定训练轮数和学习率，此处按实验经验，使用 50000 轮训练轮数，0.001 的初始学习率。

``` yaml linenums="41"
--8<--
examples/NLS-MB/conf/NLS-MB_soliton.yaml:41:54
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="184"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:184:185
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.GeometryValidator` 构建评估器。

``` py linenums="187"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:187:208
--8<--
```

### 3.8 可视化器构建

在模型训练完毕之后，我们可以在计算域取点进行预测，并手动计算出振幅，并可视化结果。

``` py linenums="255"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:255:269
--8<--
```

### 3.9 模型训练、评估与可视化

#### 3.9.1 使用 Adam 训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="210"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:210:227
--8<--
```

#### 3.9.2 使用 L-BFGS 微调[可选]

在使用 `Adam` 优化器训练完毕之后，我们可以将优化器更换成二阶优化器 `L-BFGS` 继续训练少量轮数（此处我们使用 `Adam` 优化轮数的 10% 即可），从而进一步提高模型精度。

``` py linenums="229"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:229:253
--8<--
```

???+ tip "提示"

    在常规优化器训练完毕之后，使用 `L-BFGS` 微调少量轮数的方法，在大多数场景中都可以进一步有效提高模型精度。

## 4. 完整代码

``` py linenums="1" title="NLS-MB_optical_soliton.py"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py
--8<--
```

## 5. 结果展示

### 5.1 optical_soliton

<figure markdown>
  ![optical_soliton](https://paddle-org.bj.bcebos.com/paddlescience/docs/NLS-MB/pred_optical_soliton.png){ loading=lazy}
  <figcaption>解析解结果与 PINN 预测结果对比，从上到下分别为：慢变电场（E），共振偏量（p）以及粒子数反转程度（eta）</figcaption>
</figure>

### 5.2 optical_rogue_wave

<figure markdown>
  ![optical_rogue_wave](https://paddle-org.bj.bcebos.com/paddlescience/docs/NLS-MB/pred_optical_rogue_wave.png){ loading=lazy}
  <figcaption>解析解结果与 PINN 预测结果对比，从上到下分别为：慢变电场（E），共振偏量（p）以及粒子数反转程度（eta）</figcaption>
</figure>

可以看到PINN预测与解析解的结果基本一致。

## 6. 参考资料

1. [S.-Y. Xu, Q. Zhou, and W. Liu, Prediction of Soliton Evolution and Equation Parameters for NLS–MB Equation Based on the phPINN Algorithm, Nonlinear Dyn (2023)](https://doi.org/10.1007/s11071-023-08824-w).
