# Shock Wave

<a href="https://aistudio.baidu.com/projectdetail/6755993?contributionType=1&sUid=438690&shared=1&ts=1694949960479" class="md-button md-button--primary" style>AI Studio快速体验</a>

## 1. 背景简介

激波是自然界以及工程应用中经常发现的现象。它们不仅广泛地存在于航空航天领域的可压缩流动中，而且也表现在理论与应用物理以及工程应用等其它领域。在超声速与高超声速流动中，激波的出现对流体流动的整体特征会产生重要影响。激波捕捉问题已在CFD领域发展了数十年，以弱解的数学理论为基础的激波捕捉方法以其简单易实现的特点发展迅速，并在复杂超声速、高超声速流动数值模拟中得到了广泛应用。

本案例针对 PINN-WE 模型进行优化，使得该模型可适用于超音速、高超音速等具有强激波的流场模拟中。

PINN-WE 模型通过损失函数加权，在 PINN 优化过程中减弱强梯度区域的拟合，避免了因激波区域强梯度引起的激波过拟合问题，其在一维 Euler 问题、弱激波情况下的二维问题中取得了不错的结果。但是在超音速二维流场中，该模型并没有取得很好的效果，在实验中还发现该模型经常出现激波位置偏移，激波形状不对称等非物理解的预测结果。因此本案例针对上述 PINN-WE 模型的这一问题，提出渐进加权的思想，抛弃优化过程中强调梯度思想，而是创新性地通过逐步强化梯度权重对模型优化的影响，使得模型在优化过程中能够得到较好的、符合物理的激波位置。

## 2. 问题定义

本问题针对二维超声速流场圆柱弓形激波进行模拟，涉及二维Euler方程，如下所示：

$$
\begin{array}{cc}
  \dfrac{\partial \hat{U}}{\partial t}+\dfrac{\partial \hat{F}}{\partial \xi}+\dfrac{\partial \hat{G}}{\partial \eta}=0 \\
  \text { 其中, } \quad
  \begin{cases}
    \hat{U}=J U \\
    \hat{F}=J\left(F \xi_x+G \xi_y\right) \\
    \hat{G}=J\left(F \eta_x+G \eta_y\right)
  \end{cases} \\
  U=\left(\begin{array}{l}
  \rho \\
  \rho u \\
  \rho v \\
  E
  \end{array}\right), \quad F=\left(\begin{array}{l}
  \rho u \\
  \rho u^2+p \\
  \rho u v \\
  (E+p) u
  \end{array}\right), \quad G=\left(\begin{array}{l}
  \rho v \\
  \rho v u \\
  \rho v^2+p \\
  (E+p) v
  \end{array}\right)
\end{array}
$$

自由来流条件 $\rho_{\infty}=1.225 \mathrm{~kg} / \mathrm{m}^3$ ; $P_{\infty}=1 \mathrm{~atm}$

整体流程如下所示：

![computation_progress](https://paddle-org.bj.bcebos.com/paddlescience/docs/ShockWave/computation_progress.png)

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

!!! note "说明"

    本案例默认使用 `MA=2.0` 作为参数，如需使用其他参数，请在 `examples/shock_wave/shock_wave.py` 对 `MA` 进行修改。

### 3.1 模型构建

在 ShockWave 问题中，给定时间 $t$ 和位置坐标 $(x,y)$，模型负责预测出对应的 $x$ 方向速度、 $y$ 防线速度、压力、密度四个物理量 $(u,v,p,\rho)$，因此我们在这里使用比较简单的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(t,x,y)$ 到 $(u,v,p,\rho)$ 的映射函数 $g: \mathbb{R}^3 \to \mathbb{R}^4$ ，即：

$$
u,v,p,\rho = g(t,x,y)
$$

上式中 $g$ 即为 MLP 模型本身，用 PaddleScience 代码表示如下

``` py linenums="255"
--8<--
examples/shock_wave/shock_wave.py:255:256
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("t", "x", "y")`，输出变量名是 `("u", "v", "p", "rho")`，这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数以及激活函数，我们就实例化出了一个拥有 9 层隐藏神经元，每层神经元数为 90，使用 "tanh" 作为激活函数的神经网络模型 `model`。

### 3.2 方程构建

本案例涉及二维欧拉方程和边界上的方程，如下所示

``` py linenums="32"
--8<--
examples/shock_wave/shock_wave.py:32:211
--8<--
```

``` py linenums="258"
--8<--
examples/shock_wave/shock_wave.py:258:259
--8<--
```

### 3.3 计算域构建

本案例的计算域为 0 ~ 0.4 单位时间，长为 1.5，宽为 2.0 的长方形区域，其内含有一个圆心坐标为 [1, 1]，半径为 0.25 的圆，代码如下所示

``` py linenums="261"
--8<--
examples/shock_wave/shock_wave.py:261:273
--8<--
```

### 3.4 约束构建

#### 3.4.1 内部点约束

我们将欧拉方程施加在计算域的内部点上，并且使用拉丁超立方(Latin HyperCube Sampling, LHS)方法采样共 `N_INTERIOR` 个训练点，代码如下所示：

``` py linenums="268"
--8<--
examples/shock_wave/shock_wave.py:268:268
--8<--
```

``` py linenums="275"
--8<--
examples/shock_wave/shock_wave.py:275:289
--8<--
```

``` py linenums="349"
--8<--
examples/shock_wave/shock_wave.py:349:363
--8<--
```

#### 3.4.2 边界约束

我们将边界条件施加在计算域的边界点上，同样使用拉丁超立方(Latin HyperCube Sampling, LHS)方法在边界上采样共 `N_BOUNDARY` 个训练点，代码如下所示：

``` py linenums="269"
--8<--
examples/shock_wave/shock_wave.py:269:269
--8<--
```

``` py linenums="291"
--8<--
examples/shock_wave/shock_wave.py:291:324
--8<--
```

``` py linenums="376"
--8<--
examples/shock_wave/shock_wave.py:376:400
--8<--
```

#### 3.4.3 初值约束

我们将边界条件施加在计算域的初始时刻的点上，同样使用拉丁超立方(Latin HyperCube Sampling, LHS)方法在初始时刻的计算域内采样共 `N_BOUNDARY` 个训练点，代码如下所示：

``` py linenums="326"
--8<--
examples/shock_wave/shock_wave.py:326:347
--8<--
```

``` py linenums="364"
--8<--
examples/shock_wave/shock_wave.py:364:375
--8<--
```

在以上三个约束构建完毕之后，需要将他们包装成一个字典，方便后续作为参数传递

``` py linenums="401"
--8<--
examples/shock_wave/shock_wave.py:401:406
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数和学习率，此处我们按实验经验，使用 100 轮训练轮数。

``` py linenums="412"
--8<--
examples/shock_wave/shock_wave.py:412:412
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `L-BFGS` 优化器并设定 `max_iter` 为 100。

``` py linenums="408"
--8<--
examples/shock_wave/shock_wave.py:408:409
--8<--
```

### 3.7 模型训练与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="412"
--8<--
examples/shock_wave/shock_wave.py:412:426
--8<--
```

本案例需要根据每一轮训练的 epoch 值，计算PDE、BC方程内的权重系数 `relu`。因此在 solver 实例化完毕之后，需额外将其传递给方程本身，代码如下：

``` py linenums="427"
--8<--
examples/shock_wave/shock_wave.py:427:430
--8<--
```

最后启动训练即可：

``` py linenums="432"
--8<--
examples/shock_wave/shock_wave.py:432:433
--8<--
```

训练完毕后，我们可视化最后一个时刻的计算域内辨率为 600x600 的激波，共 360000 个点，代码如下：

``` py linenums="435"
--8<--
examples/shock_wave/shock_wave.py:435:506
--8<--
```

## 4. 完整代码

=== "Ma=2.0"

    ``` py linenums="1" title="shock_wave.py"
    --8<--
    examples/shock_wave/shock_wave.py
    --8<--
    ```

=== "Ma=0.728"

    ``` py linenums="1" title="shock_wave.py"
    --8<--
    examples/shock_wave/shock_wave.py::245
    --8<--
        MA=0.728
        --8<--
        examples/shock_wave/shock_wave.py:248:
        --8<--
    ```

## 5. 结果展示

本案例针对 $Ma=2.0$ 和 $Ma=0.728$ 两种不同的参数配置进行了实验，结果如下所示

=== "Ma=2.0"

    <figure markdown>
      ![Ma_2.0](https://paddle-org.bj.bcebos.com/paddlescience/docs/ShockWave/shock_wave(Ma_2.000).png){ loading=lazy }
      <figcaption> Ma=2.0时，x方向速度u、y方向速度v、压力p、密度rho的预测结果</figcaption>
    </figure>

=== "Ma=0.728"

    <figure markdown>
      ![Ma_0.728](https://paddle-org.bj.bcebos.com/paddlescience/docs/ShockWave/shock_wave(Ma_0.728).png){ loading=lazy }
      <figcaption> Ma=0.728时，x方向速度u、y方向速度v、压力p、密度rho的预测结果</figcaption>
    </figure>

## 6. 参考资料

- [Compressible PINN - AIStudio](https://aistudio.baidu.com/projectdetail/5528154)
- [Discontinuity computing with physics-informed neural network](https://arxiv.org/abs/2206.03864)
