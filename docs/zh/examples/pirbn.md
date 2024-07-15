# PIRBN

=== "模型训练和评估命令"

    ``` sh
    cd PaddleScience/jointContribution/PIRBN
    python main.py
    ```

## 1. 背景简介

我们最近发现经过训练，物理信息神经网络（PINN）往往会成为局部近似函数。这一观察结果促使我们开发了一种新型的物理-信息径向基网络（PIRBN），该网络在整个训练过程中都能够维持局部近似性质。与深度神经网络不同，PIRBN 仅包含一个隐藏层和一个径向基“激活”函数。在适当的条件下，我们证明了使用梯度下降方法训练 PIRBN 可以收敛到高斯过程。此外，我们还通过神经邻近核（NTK）理论研究了 PIRBN 的训练动态。此外，我们还对 PIRBN 的初始化策略进行了全面调查。基于数值示例，我们发现 PIRBN 在解决具有高频特征和病态计算域的非线性偏微分方程方面比PINN更有效。此外，现有的 PINN 数值技术，如自适应学习、分解和不同类型的损失函数，也适用于 PIRBN。

<figure markdown>
  ![介绍](https://paddle-org.bj.bcebos.com/paddlescience/docs/PIRBN/PIRBN_1.png){ loading=lazy }
  <figcaption>网络的结构</figcaption>
</figure>
图片左侧为常见神经网络结构的输入层，隐藏层，输出层，隐藏层包含激活层，a 中为单层隐藏层，b 中为多层隐藏层，图片右侧为 PIRBN 网络的激活函数，计算网络的损失 Loss 并反向传递。图片说明当使用 PIRBN 时，每个 RBF 神经元仅在输入接近神经元中心时被激活。直观地说，PIRBN 具有局部逼近特性。通过梯度下降算法训练一个 PIRBN 也可以通过 NTK 理论进行分析。

<figure markdown>
  ![gaussian](https://paddle-org.bj.bcebos.com/paddlescience/docs/PIRBN/PIRBN_2.png){ loading=lazy }
  <figcaption>不同阶数的高斯激活函数</figcaption>
</figure>
(a) 0, 1, 2 阶高斯激活函数
(b) 设置不同 b 值
(c) 设置不同 c 值

当使用高斯函数作为激活函数时，输入与输出之间的映射关系可以数学上表示为高斯函数的某种形式。RBF 网络是一种常用于模式识别、数据插值和函数逼近的神经网络，其关键特征是使用径向基函数作为激活函数，使得网络具有更好的全局逼近能力和灵活性。

## 2. 问题定义

在 NTK 和基于 NTK 的适应性训练方法的帮助下，PINN 在处理具有高频特征的问题时的性能可以得到显著提升。例如，考虑一个偏微分方程及其边界条件：

$$
\begin{aligned}
& \frac{\mathrm{d}^2}{\mathrm{~d} x^2} u(x)-4 \mu^2 \pi^2 \sin (2 \mu \pi x)=0, \text { for } x \in[0,1] \\
& u(0)=u(1)=0
\end{aligned}
$$

其中$\mu$是一个控制PDE解的频率特征的常数。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddlePaddle 代码，用深度学习的方法求解该问题。
为了快速理解 PaddlePaddle，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 PIRBN 问题中，建立网络，用 PaddlePaddle 代码表示如下

``` py linenums="40"
--8<--
jointContribution/PIRBN/main.py:40:42
--8<--
```

### 3.2 数据构建

本案例涉及读取数据构建，如下所示

``` py linenums="18"
--8<--
jointContribution/PIRBN/main.py:18:38
--8<--
```

### 3.3 训练和评估构建

训练和评估构建，设置损失计算函数，返回字段，代码如下所示：

``` py linenums="52"
--8<--
jointContribution/PIRBN/train.py:52:90
--8<--
```

### 3.4 超参数设定

接下来我们需要指定训练轮数，此处我们按实验经验，使用 20001 轮训练轮数。

``` py linenums="43"
--8<--
jointContribution/PIRBN/main.py:43:43
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `Adam` 优化器并设定 `learning_rate` 为 1e-3。

``` py linenums="33"
--8<--
jointContribution/PIRBN/train.py:33:35
--8<--
```

### 3.6 模型训练与评估

模型训练与评估

``` py linenums="92"
--8<--
jointContribution/PIRBN/train.py:92:99
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

<figure markdown>
  ![PINN](https://paddle-org.bj.bcebos.com/paddlescience/docs/PIRBN/PIRBN_3.png){ loading=lazy }
  <figcaption>PINN 结果图</figcaption>
</figure>
图为使用双曲正切函数（tanh）作为激活函数（activation function），并且使用 LuCun 初始化方法来初始化神经网络中的所有参数。

- 图中子图 1 为预测值和真实值的曲线比较
- 图中子图 2 为误差值
- 图中子图 3 为损失值
- 图中子图 4 为训练 1 次的 Kg 图
- 图中子图 5 为训练 2000 次的 Kg 图
- 图中子图 6 为训练 20000 次的 Kg 图

可以看到预测值和真实值可以匹配，误差值逐渐升高然后逐渐减少，Loss 历史降低后波动，Kg 图随训练次数增加而逐渐收敛。

<figure markdown>
  ![PIRBN](https://paddle-org.bj.bcebos.com/paddlescience/docs/PIRBN/PIRBN_4.png){ loading=lazy }
  <figcaption>PIRBN 结果图</figcaption>
</figure>
图为使用高斯函数（gaussian function）作为激活函数（activation function）生成的数据，并且使用 LuCun 初始化方法来初始化神经网络中的所有参数。

- 图中子图 1 为预测值和真实值的曲线比较
- 图中子图 2 为误差值
- 图中子图 3 为损失值
- 图中子图 4 为训练 1 次的 Kg 图
- 图中子图 5 为训练 2000 次的 Kg 图
- 图中子图 6 为训练 20000 次的 Kg 图

可以看到预测值和真实值可以匹配，误差值逐渐升高然后逐渐减少再升高，Loss 历史降低后波动，Kg 图随训练次数增加而逐渐收敛。

## 6. 参考资料

- [Physics-informed radial basis network (PIRBN): A local approximating neural network for solving nonlinear PDEs](https://arxiv.org/abs/2304.06234)
- <https://github.com/JinshuaiBai/PIRBN>
