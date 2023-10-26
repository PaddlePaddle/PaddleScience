# EPNN

=== "模型训练命令"

    ``` sh
    # linux
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -O datasets/epnn/dstate-16-plas.dat
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -O datasets/epnn/dstress-16-plas.dat
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat --output datasets/epnn/dstate-16-plas.dat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat --output datasets/epnn/dstress-16-plas.dat
    python epnn.py
    ```

=== "模型评估命令"

    ``` sh
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -O datasets/epnn/dstate-16-plas.dat
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -O datasets/epnn/dstress-16-plas.dat
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat --output datasets/epnn/dstate-16-plas.dat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat --output datasets/epnn/dstress-16-plas.dat
    python epnn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/epnn/epnn_pretrained.pdparams
    ```

## 1. 背景简介

在这项工作中，我们提出了一种能够高效逼近经典弹塑性本构关系的深度神经网络架构。该网络富含经典弹塑性的关键物理方面，包括应变添加剂分解为弹性和塑性部分，以及非线性增量弹性。这导致了一个名为Elasto-Plastic Neural Network (EPNN)的Physics-Informed Neural Network (PINN)代理模型。详细的分析表明，将这些物理嵌入神经网络的架构中，可以更有效地训练网络，同时使用更少的数据进行训练，同时增强对训练数据外加载制度的推断能力。EPNN的架构是模型和材料无关的，即它可以适应各种弹塑性材料类型，包括地质材料和金属；并且实验数据可以直接用于训练网络。为了证明所提出架构的稳健性，我们将其一般框架应用于砂土的弹塑性行为。我们使用基于相对先进的基于流变性的颗粒材料本构模型的材料点模拟生成的合成数据来训练神经网络。EPNN在预测不同初始密度砂土的未观测应变控制加载路径方面优于常规神经网络架构。

## 2. 问题定义

在神经网络中，信息通过由连接的神经元流动。神经网络中每个链接的“强度”是由一个可变的权重决定的：

$$
z_l^{\mathrm{i}}=W_{k l}^{\mathrm{i}-1, \mathrm{i}} a_k^{\mathrm{i}-1}+b^{\mathrm{i}-1}, \quad k=1: N^{\mathrm{i}-1} \quad \text { or } \quad \mathbf{z}^{\mathrm{i}}=\mathbf{a}^{\mathrm{i}-1} \mathbf{W}^{\mathrm{i}-1, \mathrm{i}}+b^{\mathrm{i}-1} \mathbf{I}
$$

其中b是偏置项；N为不同层中神经元数量；I指的是所有元素都为1的单位向量。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 EPNN 问题中，建立网络，用 PaddleScience 代码表示如下

``` py linenums="341"
--8<--
examples/epnn/functions.py:341:361
--8<--
```

Epnn 参数 input_keys 是输入字段名，output_keys 是输出字段名，node_sizes 是节点大小列表，activations 是激活函数字符串列表，drop_p 是 nn.Dropout 中的 p 参数。

### 3.2 数据构建

本案例涉及读取数据构建，如下所示

``` py linenums="36"
--8<--
examples/epnn/epnn.py:36:41
--8<--
```

### 3.3 约束构建

设置训练数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="83"
--8<--
examples/epnn/epnn.py:83:103
--8<--
```

### 3.4 评估器构建

设置评估数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="106"
--8<--
examples/epnn/epnn.py:106:128
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数，此处我们按实验经验，使用 10000 轮训练轮数。

``` yaml linenums="39"
--8<--
examples/epnn/conf/epnn.yaml:39:39
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `Adam` 优化器并设定 `learning_rate`。

``` py linenums="366"
--8<--
examples/epnn/functions.py:366:412
--8<--
```

### 3.7 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="131"
--8<--
examples/epnn/epnn.py:131:144
--8<--
```

最后启动训练即可：

``` py linenums="147"
--8<--
examples/epnn/epnn.py:147:147
--8<--
```

## 4. 完整代码

``` py linenums="1" title="epnn.py"
--8<--
examples/epnn/epnn.py
--8<--
```

## 5. 结果展示

EPNN 案例针对 epoch=10000 的参数配置进行了实验，结果返回Loss为 0.00471。

## 6. 参考资料

- [A physics-informed deep neural network for surrogate
modeling in classical elasto-plasticity](https://arxiv.org/abs/2204.12088)
- <https://github.com/meghbali/ANNElastoplasticity>
