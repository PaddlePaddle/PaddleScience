# PIRBN

## 1. 背景简介

我们最近发现经过训练，物理信息神经网络（PINN）往往会成为局部近似函数。这一观察结果促使我们开发了一种新型的物理-信息径向基网络（PIRBN），该网络在整个训练过程中都能够维持局部近似性质。与深度神经网络不同，PIRBN仅包含一个隐藏层和一个径向基“激活”函数。在适当的条件下，我们证明了使用梯度下降方法训练PIRBN可以收敛到高斯过程。此外，我们还通过神经邻近核（NTK）理论研究了PIRBN的训练动态。此外，我们还对PIRBN的初始化策略进行了全面调查。基于数值示例，我们发现PIRBN在解决具有高频特征和病态计算域的非线性偏微分方程方面比PINN更有效。此外，现有的PINN数值技术，如自适应学习、分解和不同类型的损失函数，也适用于PIRBN。

PIRBN网络的结构
![介绍](pirbn_images/intro.png)

不同阶数的高斯激活函数
![gaussian](pirbn_images/gaussian.png)
(a) 0, 1, 2阶高斯激活函数
(b) 设置不同b值
(c) 设置不同c值

## 2. 问题定义

在NTK和基于NTK的适应性训练方法的帮助下，PINN在处理具有高频特征的问题时的性能可以得到显著提升。例如，考虑一个偏微分方程及其边界条件：

$$
\begin{aligned}
& \frac{\mathrm{d}^2}{\mathrm{~d} x^2} u(x)-4 \mu^2 \pi^2 \sin (2 \mu \pi x)=0, \text { for } x \in[0,1] \\
& u(0)=u(1)=0
\end{aligned}
$$

其中μ是一个控制PDE解的频率特征的常数。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddlePaddle 代码，用深度学习的方法求解该问题。
为了快速理解 PaddlePaddle，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 PIRBN 问题中，建立网络，用 PaddlePaddle 代码表示如下

``` py linenums="44"
--8<--
jointContribution/PIRBN/main.py:44:46
--8<--
```

### 3.2 数据构建

本案例涉及读取数据构建，如下所示

``` py linenums="18"
--8<--
jointContribution/PIRBN/main.py:18:41
--8<--
```

### 3.3 训练和评估构建

训练和评估构建，设置损失计算函数，返回字段，代码如下所示：

``` py linenums="59"
--8<--
jointContribution/PIRBN/train.py:59:97
--8<--
```

### 3.4 超参数设定

接下来我们需要指定训练轮数，此处我们按实验经验，使用 20001 轮训练轮数。

``` py linenums="47"
--8<--
jointContribution/PIRBN/main.py:47:47
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `Adam` 优化器并设定 `learning_rate` 为 1e-3。

``` py linenums="40"
--8<--
jointContribution/PIRBN/train.py:40:42
--8<--
```

### 3.6 模型训练与评估

模型训练与评估

``` py linenums="99"
--8<--
jointContribution/PIRBN/train.py:99:106
--8<--
```

## 4. 完整代码

``` py linenums="1" title="main.py"
--8<--
jointContribution/PIRBN/main.py
--8<--
```

## 5. 结果展示

PINN 案例针对 epoch=20001 和 learning\_rate=1e-3 的参数配置进行了实验，结果返回Loss为 0.13567。
PIRBN 案例针对 epoch=20001 和 learning\_rate=1e-3 的参数配置进行了实验，结果返回Loss为 0.59471。

![PINN](pirbn_images/pinn_result.png)
![PIRBN](pirbn_images/pirbn_result.png)

![PINN](pirbn_images/sine_function_4_10.0_0_tanh.png)
![PIRBN](pirbn_images/sine_function_8_10.0_100_gaussian.png)

## 6. 参考资料

- [Physics-informed radial basis network (PIRBN): A local approximating neural network for solving nonlinear PDEs](https://arxiv.org/abs/2304.06234)
- <https://github.com/JinshuaiBai/PIRBN >
