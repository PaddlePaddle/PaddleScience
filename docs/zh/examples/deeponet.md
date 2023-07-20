# DeepONet

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6160556?contributionType=1&sUid=438690&shared=1&ts=1683961088129" class="md-button md-button--primary" style>AI Studio快速体验</a>

## 1. 问题简介

根据机器学习领域的万能近似定理，一个神经网络模型不仅可以拟合输入数据到输出数据的函数映射关系，也可以扩展到对函数与函数之间的映射关系进行拟合，称之为“算子”学习。

## 2. 问题定义

假设存在如下 ODE 系统：

$$
\begin{equation}
\left\{\begin{array}{l}
\frac{d}{d x} \mathbf{s}(x)=\mathbf{g}(\mathbf{s}(x), u(x), x) \\
\mathbf{s}(a)=s_0
\end{array}\right.
\end{equation}
$$

其中 $u \in V$（且 $u$ 在 $[a, b]$ 上连续）作为输入信号，$\mathbf{s}: [a,b] \rightarrow \mathbb{R}^K$ 是该方程的解，作为输出信号。
因此可以定义一种算子 $G$，它满足：

$$
\begin{equation}
(G u)(x)=s_0+\int_a^x \mathbf{g}((G u)(t), u(t), t) d t
\end{equation}
$$

因此可以利用神经网络模型，以 $u$、$x$ 为输入，$G(u)(x)$ 为输出，进行监督训练来拟合 $G$ 算子本身。

注：根据上述公式，可以发现算子 $G$ 是一种积分算子 "$\int$"，其作用在给定函数 $u$ 上能求得其符合某种初值条件（本问题中初值条件为 $G(u)(0)=0$）下的原函数 $G(u)$。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

本案例数据集使用 DeepXDE 官方文档提供的数据集，一个 npz 文件内已包含训练集和验证集，[下载地址](https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2Fdeepxde%2Fdeeponet%5Fantiderivative%5Funaligned)

数据文件说明如下：

`antiderivative_unaligned_train.npz`

|字段名 |   说明     |
|:----:|:---------:|
|X_train0 |$u$ 对应的训练输入数据，形状为(10000, 100) |
|X_train1 |$y$ 对应的训练输入数据数据，形状为(10000, 1)  |
|y_train |$G(u)$ 对应的训练标签数据，形状为(10000,1)  |

`antiderivative_unaligned_test.npz`

|字段名 |   说明     |
|:----:|:---------:|
|X_test0 |$u$ 对应的测试输入数据，形状为(100000, 100) |
|X_test1 |$y$ 对应的测试输入数据数据，形状为(100000, 1)  |
|y_test |$G(u)$ 对应的测试标签数据，形状为(100000,1)  |

### 3.2 模型构建

在上述问题中，我们确定了输入为 $u$ 和 $y$，输出为 $G(u)$，按照 DeepONet 论文所述，我们使用含有 branch 和 trunk 两个子分支网络的 `DeepONet` 来创建网络模型，用 PaddleScience 代码表示如下：

``` py linenums="26"
--8<--
examples/operator_learning/deeponet.py:26:42
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `u` 和 `y`，输出变量名是 `G`，接着通过指定 `DeepONet` 的 SENSORS 个数，特征通道数、隐藏层层数、神经元个数以及子网络的激活函数，我们就实例化出了 `DeepONet` 神经网络模型 `model`。

### 3.3 约束构建

本文采用监督学习的方式，对模型输出 $G(u)$ 进行约束。

在定义约束之前，需要给监督约束指定文件路径等数据读取配置，包括文件路径、输入数据字段名、标签数据字段名、数据转换前后的别名字典。

``` py linenums="44"
--8<--
examples/operator_learning/deeponet.py:44:54
--8<--
```

#### 3.3.1 监督约束

由于我们以监督学习方式进行训练，此处采用监督约束 `SupervisedConstraint`：

``` py linenums="56"
--8<--
examples/operator_learning/deeponet.py:56:60
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，此处填入在 [3.4 约束构建](#34) 章节中实例化好的 `train_dataloader_cfg`；

第二个参数是损失函数，此处我们选用常用的MSE函数，且 `reduction` 为默认值 `"mean"`，即我们会将参与计算的所有数据点产生的损失项求和取平均；

第三个参数是方程表达式，用于描述如何计算约束目标，此处我们只需要从输出字典中，获取输出 `G` 这个字段对应的输出即可；

在监督约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="61"
--8<--
examples/operator_learning/deeponet.py:61:62
--8<--
```

### 3.4 超参数设定

接下来我们需要指定训练轮数和学习率，此处我们按实验经验，使用十万轮训练轮数，并每隔 500 个 epochs 评估一次模型精度。

``` py linenums="64"
--8<--
examples/operator_learning/deeponet.py:64:65
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器和 `Step` 间隔衰减学习率。

``` py linenums="67"
--8<--
examples/operator_learning/deeponet.py:67:68
--8<--
```

### 3.6 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="70"
--8<--
examples/operator_learning/deeponet.py:70:87
--8<--
```

评价指标 `metric` 选择 `ppsci.metric.L2Rel` 即可。

其余配置与 [约束构建](#34) 的设置类似。

### 3.7 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="89"
--8<--
examples/operator_learning/deeponet.py:89:
--8<--
```

### 3.8 结果可视化

在模型训练完毕之后，我们可以手动构造 $u$、$y$ 并在适当范围内进行离散化，得到对应输入数据，继而预测出 $G(u)(y)$，并和 $G(u)$ 的标准解共同绘制图像，进行对比。（此处我们构造了 9 组 $u-G(u)$ 函数对）进行测试

``` py linenums="124"
--8<--
examples/operator_learning/deeponet.py:124:
--8<--
```

## 4. 完整代码

``` py linenums="1" title="deeponet.py"
--8<--
examples/operator_learning/deeponet.py
--8<--
```

## 5. 结果展示

<figure markdown>
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_0_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_1_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_2_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_3_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_4_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_5_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_6_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_7_result.png){ loading=lazy }
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_8_result.png){ loading=lazy }
</figure>

## 6. 参考文献

- [DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://export.arxiv.org/pdf/1910.03193.pdf)
- [DeepXDE - Antiderivative operator from an unaligned dataset](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html)
