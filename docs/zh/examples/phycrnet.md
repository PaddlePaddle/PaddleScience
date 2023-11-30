# PhyCRNet

=== "模型训练命令"

    ``` sh
    # linux
    python Burgers_2d_solver_HighOrder.py
    python main.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    python Burgers_2d_solver_HighOrder.py
    python main.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/phycrnet/phycrnet_pretrained.pdparams
    ```

## 1. 背景简介

复杂时空系统通常可以通过偏微分方程（PDE）来建模，它们在许多领域都十分常见，如应用数学、物理学、生物学、化学和工程学。求解PDE系统一直是科学计算领域的一个关键组成部分。
本文的具体目标是为了提出一种新颖的、考虑物理信息的卷积-递归学习架构（PhyCRNet）及其轻量级变体（PhyCRNet-s），用于解决没有任何标签数据的多元时间空间PDEs。我们不试图将我们提出的方法与经典的数值求解器进行比较，而是为复杂PDEs的代理建模提供一种时空深度学习视角。

## 2. 问题定义

在此，我们考虑一组多维(n)、非线性、耦合的参数设置下的偏微分方程(PDE)系统的通用形式：

$$
\mathbf{u}_t+\mathcal{F}\left[\mathbf{u}, \mathbf{u}^2, \cdots, \nabla_{\mathbf{x}} \mathbf{u}, \nabla_{\mathbf{x}}^2 \mathbf{u}, \nabla_{\mathbf{x}} \mathbf{u} \cdot \mathbf{u}, \cdots ; \boldsymbol{\lambda}\right]=\mathbf{0}
$$

我们的目标是开发基于深度神经网络（DNN）的方法，用于解决给定式中的时空PDE系统的正向分析问题。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 PhyCRNet 问题中，建立网络，用 PaddleScience 代码表示如下

``` py linenums="105"
--8<--
examples/phycrnet/main.py:163:174
--8<--
```

PhyCRNet 参数 input_channels 是输入通道数，hidden_channels 是隐藏层通道数，input_kernel_size 是内核层大小。

### 3.2 数据构建

运行本问题代码前请按照下方命令生成数据集

``` shell
python Burgers_2d_solver_HighOrder.py
python FN_2d_solver_HighOrder.py
```

本案例涉及读取数据构建，如下所示

``` py linenums="182"
--8<--
examples/phycrnet/main.py:182:191
--8<--
```

### 3.3 约束构建

设置训练数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="200"
--8<--
examples/phycrnet/main.py:200:213
--8<--
```

### 3.4 评估器构建

设置评估数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="216"
--8<--
examples/phycrnet/main.py:216:230
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数，此处我们按实验经验，使用 2000 轮训练轮数。

``` py linenums="143"
--8<--
examples/phycrnet/main.py:143:143
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `Adam` 优化器并设定 `learning_rate` 为 1e-4。

``` py linenums="242"
--8<--
examples/phycrnet/main.py:242:242
--8<--
```

### 3.7 模型训练与评估

为了评估所有基于神经网络的求解器产生的解决方案精度，我们分两个阶段评估了全场误差传播：训练和外推。在时刻 τ 的全场误差 $\epsilon_\tau$ 的定义为给定 b 的累积均方根误差 (a-RMSE)。

$$
\epsilon_\tau=\sqrt{\frac{1}{N_\tau} \sum_{k=1}^{N_\tau} \frac{\left\|\mathbf{u}^*\left(\mathbf{x}, t_k\right)-\mathbf{u}^\theta\left(\mathbf{x}, t_k\right)\right\|_2^2}{m n}}
$$

其中，$N_\tau$ 是 $[0, \tau]$ 内的总时间步数，$\mathbf{u}^*\left(\mathbf{x}, t_k\right)$ 为参考解决方案。

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="243"
--8<--
examples/phycrnet/main.py:243:254
--8<--
```

最后启动训练、评估即可：

``` py linenums="256"
--8<--
examples/phycrnet/main.py:256:260
--8<--
```

## 4. 完整代码

``` py linenums="1" title="phycrnet"
--8<--
examples/phycrnet/main.py
--8<--
```

## 5. 结果展示

PhyCRNet 案例针对 epoch=200 和 learning\_rate=1e-4 的参数配置进行了实验，结果返回Loss为 17.86。

## 6. 参考资料

- [PhyCRNet: Physics-informed Convolutional-Recurrent Network for Solving Spatiotemporal PDEs](https://arxiv.org/abs/2106.14103)
- <https://github.com/isds-neu/PhyCRNet>
