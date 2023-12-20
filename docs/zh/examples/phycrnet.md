# PhyCRNet
<a href="https://aistudio.baidu.com/projectdetail/7296776" class="md-button md-button--primary" style>AI Studio快速体验</a>


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
本文的具体目标是为了提出一种新颖的、考虑物理信息的卷积-递归学习架构（PhyCRNet）及其轻量级变体（PhyCRNet-s），用于解决没有任何标签数据的多元时间空间PDEs。

## 2. 问题定义

在本模型中，我们考虑的是含有时间和空间的PDE模型，此类模型在推理过程中会存在时间上的误差累积的问题，因此，本文作者通过设计循环卷积神经网络试图减轻每一步时间迭代的误差累积。

## 3. 问题求解

### 3.1 模型构建

在 PhyCRNet 问题中，建立网络：

``` py linenums="43"
--8<--
examples/phycrnet/main.py:43:45
--8<--
```

### 3.2 数据载入
我们使用RK4或者谱方法生成的数据（初值为使用正态分布生成），需要从.mat文件中将其读入，并将其整合：
``` py linenums="54"
--8<--
examples/phycrnet/main.py:54:72
--8<--
```

### 3.3 约束构建

设置约束以及相关损失函数：

``` py linenums="74"
--8<--
examples/phycrnet/main.py:74:90
--8<--
```

### 3.4 评估器构建

设置评估数据集和相关损失函数：

``` py linenums="92"
--8<--
examples/phycrnet/main.py:92:109
--8<--
```


### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `Adam` 优化器并设定 `learning_rate`

``` py linenums="112"
--8<--
examples/phycrnet/main.py:112:116
--8<--
```

### 3.7 模型训练与评估

为了评估所有基于神经网络的求解器产生的解决方案精度，我们分两个阶段评估了全场误差传播：训练和外推。在时刻 τ 的全场误差 $\epsilon_\tau$ 的定义为给定 b 的累积均方根误差 (a-RMSE)。

$$
\epsilon_\tau=\sqrt{\frac{1}{N_\tau} \sum_{k=1}^{N_\tau} \frac{\left\|\mathbf{u}^*\left(\mathbf{x}, t_k\right)-\mathbf{u}^\theta\left(\mathbf{x}, t_k\right)\right\|_2^2}{m n}}
$$

这一步需要通过设置外界函数来进行，因此在训练过程中，我们使用`function.transform_out`来进行训练
``` py linenums="47"
--8<--
examples/phycrnet/main.py:47:51
--8<--
```
而在评估过程中，我们使用`function.tranform_output_val`来进行评估，并生成累计均方根误差。
``` py linenums="134"
--8<--
examples/phycrnet/main.py:134:134
--8<--
```
完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="117"
--8<--
examples/phycrnet/main.py:117:129
--8<--
```

最后启动训练、评估即可：

``` py linenums="132"
--8<--
examples/phycrnet/main.py:132:140
--8<--
```

## 4. 完整代码

``` py linenums="1" title="phycrnet"
--8<--
examples/phycrnet/main.py
--8<--
```

## 5. 结果展示

本文通过对Burgers' Equation 以及两种reaction-diffusion systems进行训练，所得结果如下：
## 6. 结论
本论文通过提出一个新的神经网络PhyCRNet,通过将传统有限差分的思路嵌入物理信息神经网络中，针对性地解决原神经网络缺少对长时间数据的推理能力、误差累积以及缺少泛化能力的问题。与此同时，本文通过类似于有限差分的边界处理方式，将原本边界条件的软限制转为硬限制，大大提高了神经网络的准确性。

## 7. 参考资料

- [PhyCRNet: Physics-informed Convolutional-Recurrent Network for Solving Spatiotemporal PDEs](https://arxiv.org/abs/2106.14103)
- <https://github.com/isds-neu/PhyCRNet>
