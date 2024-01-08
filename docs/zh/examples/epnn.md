# EPNN

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat --output datasets/dstate-16-plas.dat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat --output datasets/dstress-16-plas.dat
    python epnn.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat --output datasets/dstate-16-plas.dat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat --output datasets/dstress-16-plas.dat
    python epnn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/epnn/epnn_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [epnn_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/epnn/epnn_pretrained.pdparams) | error(total): 3.96903<br> error(error_elasto): 0.65328<br> error(error_plastic): 3.04176<br> error(error_stress): 0.27399 |

## 1. 背景简介

这里主要为复现 Elasto-Plastic Neural Network (EPNN) 的 Physics-Informed Neural Network (PINN) 代理模型。将这些物理嵌入神经网络的架构中，可以更有效地训练网络，同时使用更少的数据进行训练，同时增强对训练数据外加载制度的推断能力。EPNN 的架构是模型和材料无关的，即它可以适应各种弹塑性材料类型，包括地质材料和金属；并且实验数据可以直接用于训练网络。为了证明所提出架构的稳健性，我们将其一般框架应用于砂土的弹塑性行为。EPNN 在预测不同初始密度砂土的未观测应变控制加载路径方面优于常规神经网络架构。

## 2. 问题定义

在神经网络中，信息通过连接的神经元流动。神经网络中每个链接的“强度”是由一个可变的权重决定的：

$$
z_l^{\mathrm{i}}=W_{k l}^{\mathrm{i}-1, \mathrm{i}} a_k^{\mathrm{i}-1}+b^{\mathrm{i}-1}, \quad k=1: N^{\mathrm{i}-1} \quad \text { or } \quad \mathbf{z}^{\mathrm{i}}=\mathbf{a}^{\mathrm{i}-1} \mathbf{W}^{\mathrm{i}-1, \mathrm{i}}+b^{\mathrm{i}-1} \mathbf{I}
$$

其中 $b$ 是偏置项；$N$ 为不同层中神经元数量；$I$ 指的是所有元素都为 1 的单位向量。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 EPNN 问题中，建立网络，用 PaddleScience 代码表示如下

``` py linenums="371"
--8<--
examples/epnn/functions.py:371:391
--8<--
```

EPNN 参数 `input_keys` 是输入字段名，`output_keys` 是输出字段名，`node_sizes` 是节点大小列表，`activations` 是激活函数字符串列表，`drop_p` 是节点丢弃概率。

### 3.2 数据生成

本案例涉及读取数据生成，如下所示

``` py linenums="36"
--8<--
examples/epnn/epnn.py:36:41
--8<--
```

``` py linenums="306"
--8<--
examples/epnn/functions.py:306:321
--8<--
```

这里使用 Data 读取文件构造数据类，然后使用 get_shuffled_data 混淆数据，然后计算需要获取的混淆数据数量 itrain，最后使用 get 获取每组 itrain 数量的 10 组数据。

### 3.3 约束构建

设置训练数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="63"
--8<--
examples/epnn/epnn.py:63:86
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，配置中 `“dataset”` 字段表示使用的训练数据集信息，其各个字段分别表示：

1. `name`： 数据集类型，此处 `"NamedArrayDataset"` 表示顺序读取的数据集；
2. `input`： 输入数据集；
3. `label`： 标签数据集；

第二个参数是损失函数，此处使用自定义函数 `train_loss_func`。

第三个参数是方程表达式，用于描述如何计算约束目标，计算后的值将会按照指定名称存入输出列表中，从而保证 loss 计算时可以使用这些值。

第四个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

在约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

### 3.4 评估器构建

与约束同理，本问题使用 `ppsci.validate.SupervisedValidator` 构建评估器，参数含义也与[约束构建](#33)类似，唯一的区别是评价指标 `metric`。代码如下所示：

``` py linenums="88"
--8<--
examples/epnn/epnn.py:88:103
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数，此处我们按实验经验，使用 10000 轮训练轮数。iters_per_epoch 为 1。

``` yaml linenums="40"
--8<--
examples/epnn/conf/epnn.yaml:40:41
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

由于使用多个模型，需要设置多个优化器，对 EPNN 网络部分，需要设置 `Adam` 优化器。

``` py linenums="395"
--8<--
examples/epnn/functions.py:395:404
--8<--
```

然后对增加的 gkratio 参数，需要再设置优化器。

``` py linenums="406"
--8<--
examples/epnn/functions.py:406:413
--8<--
```

优化器按顺序优化，代码汇总为：

``` py linenums="395"
--8<--
examples/epnn/functions.py:395:413
--8<--
```

### 3.7 自定义 loss

由于本问题包含无监督学习，数据中不存在标签数据，loss 根据模型返回数据计算得到，因此需要自定义 loss。方法为先定义相关函数，再将函数名作为参数传给 `FunctionalLoss` 和 `FunctionalMetric`。

需要注意自定义 loss 函数的输入输出参数需要与 PaddleScience 中如 `MSE` 等其他函数保持一致，即输入为模型输出 `output_dict` 等字典变量，loss 函数输出为 loss 值 `paddle.Tensor`。

相关的自定义 loss 函数使用 `MAELoss` 计算，代码为

``` py linenums="114"
--8<--
examples/epnn/functions.py:114:126
--8<--
```

### 3.8 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="106"
--8<--
examples/epnn/epnn.py:106:118
--8<--
```

模型训练时设置 eval_during_train 为 True，将在每次训练后评估。

``` yaml linenums="43"
--8<--
examples/epnn/conf/epnn.yaml:43:43
--8<--
```

最后启动训练即可：

``` py linenums="121"
--8<--
examples/epnn/epnn.py:121:121
--8<--
```

## 4. 完整代码

``` py linenums="1" title="epnn.py"
--8<--
examples/epnn/epnn.py
--8<--
```

## 5. 结果展示

EPNN 案例针对 epoch=10000 的参数配置进行了实验，结果返回 Loss 为 0.00471。

下图分别为不同 epoch 的 Loss, Training error, Cross validation error 图形：

<figure markdown>
  ![loss_trend](https://paddle-org.bj.bcebos.com/paddlescience/docs/EPNN/loss_trend.png){ loading=lazy }
  <figcaption> 训练 loss 图形 </figcaption>
</figure>

## 6. 参考资料

- [A physics-informed deep neural network for surrogate modeling in classical elasto-plasticity](https://arxiv.org/abs/2204.12088)

- [参考代码](https://github.com/meghbali/ANNElastoplasticity)
