# 3D-Brusselator

<a href="https://aistudio.baidu.com/projectdetail/8347444" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # linux
    wget -P data -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz --create-dirs -o data/brusselator3d_dataset.npz
    python brusselator3d.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -P data -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz --create-dirs -o data/brusselator3d_dataset.npz
    python brusselator3d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/Brusselator3D/brusselator3d_pretrained.pdparams
    ```


| 预训练模型  | 指标 |
|:--| :--|
| [brusselator3d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/Brusselator3D/brusselator3d_pretrained.pdparams) | loss(sup_validator): 14.51938<br>L2Rel.output(sup_validator): 0.07354 |

## 1. 背景简介

该案例引入拉普拉斯神经算子（LNO）来构建深度学习网络，它利用拉普拉斯变换来分解输入空间。与傅里叶神经算子 (FNO) 不同，LNO 可以处理非周期信号、考虑瞬态响应并表现出指数收敛，它结合了输入和输出空间之间的极点-残差关系，从而实现了更大的可解释性和改进的泛化能力。LNO 中单个拉普拉斯层与 FNO 中的四个傅里叶模块上精度近似，对于非线性反应扩散系统，LNO的误差小于FNO。

该案例研究 LNO 网络在布鲁塞尔反应扩散系统上的应用。

## 2. 问题定义

反应扩散系统描述了化学物质或粒子的浓度随时间和空间的变化，常应用于化学、生物学、地质学和物理学。扩散反应方程可以表示为：

$$D\frac{\partial^2 y}{\partial x^2}+ky^2-\frac{\partial y}{\partial t}=f(x,t)$$

其中 $y(x,t)$ 表示化学物质或颗粒在位置x和时间t的浓度，$f(x,t)$ 是源项，$D$ 是扩散系数，$k$ 是反应速率。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集为使用 LNO 论文原代码提供的数据集，数据集中包含训练集输入、标签数据，验证集输入、标签数据，数据存储在 `.npz` 文件中，在训练前需要读入数据。

运行本问题代码前请下载 [数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz)，并存放在相应路径：

``` yaml linenums="39"
--8<--
examples/brusselator3d/conf/brusselator3d.yaml:39:40
--8<--
```

### 3.2 模型构建

<figure markdown>
  ![LNO](https://paddle-org.bj.bcebos.com/paddlescience/docs/Brusselator3D/lno.png){ loading=lazy style="margin:0 auto"}
  <figcaption> (a) LNO 整体架构 (b) Laplace 层</figcaption>
</figure>

上图为 LNO 整体架构和 Laplace 层示意图。输入数据进入网络后，先通过浅神经网络 $P$ 提升到更高的维度，之后一方面进行局部线性变换 $W$，另一方面应用拉普拉斯层，之后再将这两条路径的结果进行加和，最后再通过浅神经网络 $Q$ 返回目标维度。

拉普拉斯层中的，上面一行代表应用极残差法来计算基于系统极 $\mu_{n}$ 和残差 $\beta_{n}$ 的瞬态响应残差 $\gamma_{n}$ 表示拉普拉斯域中的瞬态响应，下面一行代表应用极残差方法，根据输入极 $i\omega_{l}$ 和残差 $i\alpha_{l}$ 计算稳态响应残差 $i\lambda_{l}$ 表示拉普拉斯域中的稳态响应。

具体代码请参考 [完整代码](#4) 中 lno.py 文件。

在构建网络之前，需要根据参数设定，使用 `linespace` 明确各个维度长度，以便 LNO 网络进行 $\lambda$ 的初始化。用 PaddleScience 代码表示如下：

``` py linenums="120"
--8<--
examples/brusselator3d/brusselator3d.py:120:128
--8<--
```

另外，如果设置模型参数中 `use_grid` 为 `True`，不需要提前处理，模型会自动生成并添加网格，如果为 `False`，则需要在处理数据时，手动为数据添加网格，然后再输入模型：

``` py linenums="114"
--8<--
examples/brusselator3d/brusselator3d.py:114:118
--8<--
```

### 3.3 参数和超参数设定

我们需要指定问题相关的参数，如数据集路径、各个维度长度等。

``` yaml linenums="32"
--8<--
examples/brusselator3d/conf/brusselator3d.yaml:32:40
--8<--
```

另外需要在配置文件中指定训练轮数、`batch_size` 等其他训练所需参数。

``` yaml linenums="54"
--8<--
examples/brusselator3d/conf/brusselator3d.yaml:54:58
--8<--
```

### 3.4 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `AdamW` 优化器，并配合使用机器学习中常用的 StepDecay 学习率调整策略。

`AdamW` 优化器基于 `Adam` 优化器进行了改进，用来解决 `Adam` 优化器中 L2 正则化失效的问题。

``` py linenums="130"
--8<--
examples/brusselator3d/brusselator3d.py:130:134
--8<--
```

### 3.5 约束构建

本问题采用监督学习的方式进行训练，仅存在监督约束 `SupervisedConstraint`，代码如下：

``` py linenums="136"
--8<--
examples/brusselator3d/brusselator3d.py:136:161
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `dataset` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `NamedArrayDataset` 表示从 Array 中读取的数据集；
2. `input`： Array 类型的输入数据；
3. `label`： Array 类型的标签数据；

`batch_size` 字段表示 batch 的大小；

`sampler` 字段表示采样方法，其中各个字段表示：

1. `name`： 采样器类型，此处 `BatchSampler` 表示批采样器；
2. `drop_last`： 是否需要丢弃最后无法凑整一个 mini-batch 的样本，设为 False；
3. `shuffle`： 是否需要在生成样本下标时打乱顺序，设为 True；

`num_workers` 字段表示 输入加载时的线程数；

第二个参数是损失函数，这里选用常用的 L2Rel 损失函数，且 reduction 设置为 "sum" ，即将参与计算的所有数据点产生的损失项求和；

第三个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

``` py linenums="157"
--8<--
examples/brusselator3d/brusselator3d.py:157:157
--8<--
```

### 3.6 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此需要构建评估器：

``` py linenums="163"
--8<--
examples/brusselator3d/brusselator3d.py:163:187
--8<--
```

其中大部分参数含义与约束器中类似，不同的参数有：

第三个参数是输出的转写公式 `output_expr`，规定了最终输入数据的 key 和 value;

第四个参数是误差评估函数，这里选用的 L2Rel Error 函数，reduction 未设置，即为默认值 "mean" ，将参与计算的所有数据点产生的 Error 求平均。


### 3.7 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="189"
--8<--
examples/brusselator3d/brusselator3d.py:189:202
--8<--
```

## 4. 完整代码

``` py linenums="1" title="brusselator3d.py"
--8<--
examples/brusselator3d/brusselator3d.py
--8<--
```

``` py linenums="1" title="lno.py"
--8<--
ppsci/arch/lno.py
--8<--
```

## 5. 结果展示

下面展示了在验证集上的预测结果和标签。

<figure markdown>
  ![brusselator3d_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Brusselator3D/pretrained_result.png){ loading=lazy }
  <figcaption>蓝线为预测结果，黄线为标签</figcaption>
</figure>

可以看到模型预测的结果与标签基本一致。

## 6. 参考文献

- [LNO: Laplace Neural Operator for Solving Differential Equations](https://arxiv.org/abs/2303.10528)

- [参考代码](https://github.com/qianyingcao/Laplace-Neural-Operator/tree/main/3D_Brusselator)
