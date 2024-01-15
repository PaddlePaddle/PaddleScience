# hPINNs(PINN with hard constraints)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6390502" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat --output ./datasets/hpinns_holo_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat --output ./datasets/hpinns_holo_valid.mat
    python holography.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat --output ./datasets/hpinns_holo_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat --output ./datasets/hpinns_holo_valid.mat
    python holography.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/hPINNs/hpinns_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [hpinns_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/hPINNs/hpinns_pretrained.pdparams) | loss(opt_sup): 0.05352<br>MSE.eval_metric(opt_sup): 0.00002<br>loss(val_sup): 0.02205<br>MSE.eval_metric(val_sup): 0.00001 |

## 1. 背景简介

求解偏微分方程(PDE) 是一类基础的物理问题，在过去几十年里，以有限差分(FDM)、有限体积(FVM)、有限元(FEM)为代表的多种偏微分方程组数值解法趋于成熟。随着人工智能技术的高速发展，利用深度学习求解偏微分方程成为新的研究趋势。PINNs(Physics-informed neural networks) 是一种加入物理约束的深度学习网络，因此与纯数据驱动的神经网络学习相比，PINNs 可以用更少的数据样本学习到更具泛化能力的模型，其应用范围包括但不限于流体力学、热传导、电磁场、量子力学等领域。

传统的 PINNs 网络中的约束都是软约束，即 PDE(偏微分方程) 作为 loss 项参与网络训练。而本案例 hPINNs 通过修改网络输出的方法，将约束严格地加入网络结构中，形成一种更有效的硬约束。

同时 hPINNs 设计了不同的约束组合，进行了软约束、带正则化的硬约束和应用增强的拉格朗日硬约束 3 种条件下的实验。本文档主要针对应用增强的拉格朗日方法的硬约束进行说明，但完整代码中可以通过 `train_mode` 参数来切换三种训练模式。

本问题可参考 [AI Studio题目](https://aistudio.baidu.com/aistudio/projectdetail/4117361?channelType=0&channel=0).

## 2. 问题定义

本问题使用 hPINNs 解决基于傅立叶光学的全息领域 (holography) 的问题，旨在设计散射板的介电常数图，这种方法使得介电常数图散射光线的传播强度具备目标函数的形状。

objective 函数：

$$
\begin{aligned}
\mathcal{J}(E) &= \dfrac{1}{Area(\Omega_3)} \left\| |E(x,y)|^2-f(x,y)\right\|^2_{2,\Omega_3} \\
&= \dfrac{1}{Area(\Omega_3)} \int_{\Omega_3} (|E(x,y)|^2-f(x,y))^2 {\rm d}x {\rm d}y
\end{aligned}
$$

其中E为电场强度：$\vert E\vert^2 = (\mathfrak{R} [E])^2+(\mathfrak{I} [E])^2$

target 函数：

$$ f(x,y) =
\begin{cases}
\begin{aligned}
& 1, \ (x,y) \in [-0.5,0.5] \cap [1,2]\\
& 0, \ otherwise
\end{aligned}
\end{cases}
$$

PDE公式：

$$
\nabla^2 E + \varepsilon \omega^2 E = -i \omega \mathcal{J}
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。为了快速理解 PaddleScience，接下来仅对模型构建、约束构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集为处理好的 holography 数据集，包含训练、测试数据的 $x, y$ 以及表征 optimizer area 数据与全区域数据分界的值 $bound$，以字典的形式存储在 `.mat` 文件中。

运行本问题代码前请按照下方命令下载 [训练数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat) 和 [验证数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat)：

``` sh
wget -nc -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat
wget -nc -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat
```

### 3.2 模型构建

holograpy 问题的模型结构图为：

<figure markdown>
  ![holography-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/holograpy_arch.png){ loading=lazy style="margin:0 auto"}
  <figcaption>holography 问题的 hPINNs 网络模型</figcaption>
</figure>

在 holography 问题中，应用 PMLs(perfectly matched layers) 方法后，PDE公式变为：

$$
\dfrac{1}{1+i \dfrac{\sigma_x\left(x\right)}{\omega}} \dfrac{\partial}{\partial x} \left(\dfrac{1}{1+i \dfrac{\sigma_x\left(x\right)}{\omega}} \dfrac{\partial E}{\partial x}\right)+\dfrac{1}{1+i \dfrac{\sigma_y\left(y\right)}{\omega}} \dfrac{\partial}{\partial y} \left(\dfrac{1}{1+i \dfrac{\sigma_y\left(y\right)}{\omega}} \dfrac{\partial E}{\partial y}\right) + \varepsilon \omega^2 E = -i \omega \mathcal{J}
$$

PMLs 方法请参考 [相关论文](https://arxiv.org/abs/2108.05348)。

本问题中频率 $\omega$ 为常量 $\dfrac{2\pi}{\mathcal{P}}$（$\mathcal{P}$ 为Period），待求解的未知量 $E$ 与位置参数 $(x, y)$ 相关，在本例中，介电常数 $\varepsilon$ 同样为未知量, $\sigma_x(x)$ 和 $\sigma_y(y)$ 为由 PMLs 得到的，分别与 $x, y$ 相关的变量。我们在这里使用比较简单的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(x, y)$ 到 $(E, \varepsilon)$ 的映射函数 $f: \mathbb{R}^2 \to \mathbb{R}^2$ ，但如上图所示的网络结构，本问题中将 $E$ 按照实部和虚部分为两个部分 $(\mathfrak{R} [E],\mathfrak{I} [E])$，且使用 3 个并行的 MLP 网络分别对 $(\mathfrak{R} [E], \mathfrak{I} [E], \varepsilon)$ 进行映射，映射函数 $f_i: \mathbb{R}^2 \to \mathbb{R}^1$ ，即：

$$
\mathfrak{R} [E] = f_1(x,y), \ \mathfrak{R} [E] = f_2(x,y), \ \varepsilon = f_3(x,y)
$$

上式中 $f_1,f_2,f_3$ 分别为一个 MLP 模型，三者共同构成了一个 Model List，用 PaddleScience 代码表示如下

``` py linenums="42"
--8<--
examples/hpinns/holography.py:42:44
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("x_cos_1","x_sin_1",...,"x_cos_6","x_sin_6","y","y_cos_1","y_sin_1")` ，输出变量名分别是 `("e_re",)`, `("e_im",)`, `("eps",)`。
注意到这里的输入变量远远多于 $(x, y)$ 这两个变量，这是因为如上图所示，模型的输入实际上是 $(x, y)$ 傅立叶展开的项而不是它们本身。而数据集中提供的训练数据为 $(x, y)$ 值，这也就意味着我们需要对输入进行 transform。同时如上图所示，由于硬约束的存在，模型的输出变量名也不是最终输出，因此也需要对输出进行 transform。

### 3.3 transform构建

输入的 transform 为变量 $(x, y)$ 到 $(\cos(\omega x),\sin(\omega x),...,\cos(6 \omega x),\sin(6 \omega x),y,\cos(\omega y),\sin(\omega y))$ 的变换，输出 transform 分别为对 $(\mathfrak{R} [E], \mathfrak{I} [E], \varepsilon)$ 的硬约束，代码如下

``` py linenums="49"
--8<--
examples/hpinns/functions.py:49:92
--8<--
```

需要对每个 MLP 模型分别注册相应的 transform ，然后将 3 个 MLP 模型组成 Model List

``` py linenums="50"
--8<--
examples/hpinns/holography.py:50:59
--8<--
```

这样我们就实例化出了一个拥有 3 个 MLP 模型，每个 MLP 包含 4 层隐藏神经元，每层神经元数为 48，使用 "tanh" 作为激活函数，并包含输入输出 transform 的神经网络模型 `model list`。

### 3.4 参数和超参数设定

我们需要指定问题相关的参数，如通过 `train_mode` 参数指定应用增强的拉格朗日方法的硬约束进行训练

``` py linenums="35"
--8<--
examples/hpinns/holography.py:35:40
--8<--
```

``` py linenums="46"
--8<--
examples/hpinns/holography.py:46:48
--8<--
```

``` py linenums="28"
--8<--
examples/hpinns/functions.py:28:46
--8<--
```

由于应用了增强的拉格朗日方法，参数 $\mu$ 和 $\lambda$ 不是常量，而是随训练轮次 $k$ 改变，此时 $\beta$ 为改变的系数，即每轮训练

$\mu_k = \beta \mu_{k-1}$, $\lambda_k = \beta \lambda_{k-1}$

同时需要指定训练轮数和学习率等超参数

``` yaml linenums="53"
--8<--
examples/hpinns/conf/hpinns.yaml:53:61
--8<--
```

### 3.5 优化器构建

训练分为两个阶段，先使用 Adam 优化器进行大致训练，再使用 LBFGS 优化器逼近最优点，因此需要两个优化器，这也对应了上一部分超参数中的两种 `EPOCHS` 值

``` py linenums="62"
--8<--
examples/hpinns/holography.py:62:64
--8<--
```

``` py linenums="203"
--8<--
examples/hpinns/holography.py:203:205
--8<--
```

### 3.6 约束构建

本问题采用无监督学习的方式，约束为结果需要满足PDE公式。

虽然我们不是以监督学习方式进行训练，但此处仍然可以采用监督约束 `SupervisedConstraint`，在定义约束之前，需要给监督约束指定文件路径等数据读取配置，因为数据集中没有标签数据，因此在数据读取时我们需要使用训练数据充当标签数据，并注意在之后不要使用这部分“假的”标签数据。

``` py linenums="102"
--8<--
examples/hpinns/holography.py:102:107
--8<--
```

如上，所有输出的标签都会读取输入 `x` 的值。

下面是约束等具体内容，要注意上述提到的给定“假的”标签数据：

``` py linenums="66"
--8<--
examples/hpinns/holography.py:66:127
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `“dataset”` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `"IterableMatDataset"` 表示不分 batch 顺序读取的 `.mat` 类型的数据集；
2. `file_path`： 数据集文件路径；
3. `input_keys`： 输入变量名；
4. `label_keys`： 标签变量名；
5. `alias_dict`： 变量别名。

第二个参数是损失函数，此处的 `FunctionalLoss` 为 PaddleScience 预留的自定义 loss 函数类，该类支持编写代码时自定义 loss 的计算方法，而不是使用诸如 `MSE` 等现有方法，本问题中由于存在多个 loss 项，因此需要定义多个 loss 计算函数，这也是需要构建多个约束的原因。自定义 loss 函数代码请参考 [自定义 loss 和 metric](#38)。

第三个参数是方程表达式，用于描述如何计算约束目标，此处填入 `output_expr`，计算后的值将会按照指定名称存入输出列表中，从而保证 loss 计算时可以使用这些值。

第四个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

在约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="128"
--8<--
examples/hpinns/holography.py:128:131
--8<--
```

### 3.7 评估器构建

与约束同理，虽然本问题使用无监督学习，但仍可以使用 `ppsci.validate.SupervisedValidator` 构建评估器。本问题存在两个采样点区域，一个是较大的完整定义区域，另一个是定义域中的一块 objective 区域，评估器分别对这两个区域进行评估，因此需要构建两个评估器。`opt`对应 objective 区域，`val` 对应整个定义域。

``` py linenums="133"
--8<--
examples/hpinns/holography.py:133:181
--8<--
```

评价指标 `metric` 为 `FunctionalMetric`，这是 PaddleScience 预留的自定义 metric 函数类，该类支持编写代码时自定义 metric 的计算方法，而不是使用诸如 `MSE`、 `L2` 等现有方法。自定义 metric 函数代码请参考下一部分 [自定义 loss 和 metric](#38)。

其余配置与 [约束构建](#36) 的设置类似。

### 3.8 自定义 loss 和 metric

由于本问题采用无监督学习，数据中不存在标签数据，loss 和 metric 根据 PDE 计算得到，因此需要自定义 loss 和 metric。方法为先定义相关函数，再将函数名作为参数传给 `FunctionalLoss` 和 `FunctionalMetric`。

需要注意自定义 loss 和 metric 函数的输入输出参数需要与 PaddleScience 中如 `MSE` 等其他函数保持一致，即输入为模型输出 `output_dict` 等字典变量，loss 函数输出为 loss 值 `paddle.Tensor`，metric 函数输出为字典 `Dict[str, paddle.Tensor]`。

``` py linenums="237"
--8<--
examples/hpinns/functions.py:237:317
--8<--
```

``` py linenums="320"
--8<--
examples/hpinns/functions.py:320:336
--8<--
```

### 3.9 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="183"
--8<--
examples/hpinns/holography.py:183:200
--8<--
```

由于本问题存在多种训练模式，根据每个模式的不同，将进行 $[2,1+k]$ 次完整的训练、评估，具体代码请参考 [完整代码](#4) 中 holography.py 文件。

### 3.10 可视化

PaddleScience 中提供了可视化器，但由于本问题图片数量较多且较为复杂，代码中自定义了可视化函数，调用自定义函数即可实现可视化

``` py linenums="279"
--8<--
examples/hpinns/holography.py:279:292
--8<--
```

自定义代码请参考 [完整代码](#4) 中 plotting.py 文件。

## 4. 完整代码

完整代码包含 PaddleScience 具体实现流程代码 holography.py，所有自定义函数代码 functions.py 和 自定义可视化代码 plotting.py。

``` py linenums="1" title="holography.py"
--8<--
examples/hpinns/holography.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/hpinns/functions.py
--8<--
```

``` py linenums="1" title="plotting.py"
--8<--
examples/hpinns/plotting.py
--8<--
```

## 5. 结果展示

参考 [问题定义](#2)，下图展示了训练过程中 loss 变化、参数 lambda 和参数 mu 与增强的拉格朗日方法中训练论次 k 的变化、电场 E 和介电常数 epsilon 最终预测的值。

下图展示了对于一个定义的方形域内，电磁波传播的情况的预测。预测结果与有限差分频域(FDFD)方法的结果基本一致。

训练过程中的 loss 值变化：

<figure markdown>
  ![holograpy_result_6A](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_A.jpg){ loading=lazy }
  <figcaption> 训练过程 loss 值随 iteration 变化</figcaption>
</figure>

objective loss 值随训练轮次 k 的变化：
<figure markdown>
  ![holograpy_result_6B](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_B.jpg){ loading=lazy }
  <figcaption> k 值对应 objective loss 值</figcaption>
</figure>

k=1,4,9 时对应参数 lambda 实部和虚部的值：
<figure markdown>
  ![holograpy_result_6C](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_C.jpg){ loading=lazy }
  <figcaption> k=1,4,9 时对应 lambda 值</figcaption>
</figure>

参数 lambda 和参数 mu 的比值随训练轮次 k 的变化：
<figure markdown>
  ![holograpy_result_6D](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_D.jpg){ loading=lazy }
  <figcaption> k 值对应 lambda/mu 值</figcaption>
</figure>

参数 lambda 和参数 mu 实部的比值随训练轮次 k=1,4,6,9 时出现的频率，曲线越“尖”说明值越趋于统一，收敛的越好：
<figure markdown>
  ![holograpy_result_6E](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_E.jpg){ loading=lazy }
  <figcaption> k=1,4,6,9 时对应实部 lambda/mu 值出现频率</figcaption>
</figure>

参数 lambda 和参数 mu 虚部的比值随训练轮次 k=1,4,6,9 时出现的频率，曲线越“尖”说明值越趋于统一，收敛的越好：
<figure markdown>
  ![holograpy_result_6F](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_F.jpg){ loading=lazy }
  <figcaption> k=1,4,6,9 时对应虚部 lambda/mu 值出现频率</figcaption>
</figure>

电场 E 值：
<figure markdown>
  ![holograpy_result_7C](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig7_C.jpg){ loading=lazy }
  <figcaption> E 值</figcaption>
</figure>

介电常数 epsilon 值：
<figure markdown>
  ![holograpy_result_7eps](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig7_eps.jpg){ loading=lazy }
  <figcaption> epsilon 值</figcaption>
</figure>

## 6. 参考文献

- [PHYSICS-INFORMED NEURAL NETWORKS WITH HARD CONSTRAINTS FOR INVERSE DESIGN](https://arxiv.org/pdf/2102.04626.pdf)

- [参考代码](https://github.com/lululxvi/hpinn)
