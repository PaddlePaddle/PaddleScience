# TopOpt

<a href="https://aistudio.baidu.com/projectdetail/6956236?sUid=5453289&shared=1&ts=1698891625916" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    python topopt.py
    ```

=== "模型评估命令"

    ``` sh
    python topopt.py -m mode=eval
    ```

## 1. 背景简介

拓扑优化(Topolgy Optimization)是一种数学方法，针对给定的一组负载、边界条件和约束，在给定的设计区域内，以最大化系统性能为目标优化材料的分布。这个问题很有挑战性因为它要求解决方案是二元的，即应该说明设计区域的每个部分是否存在材料或不存在。这种优化的一个常见例子是在给定总重量和边界条件下最小化物体的弹性应变能。随着20世纪汽车和航空航天工业的发展，拓扑优化已经将应用扩展到很多其他学科：如流体、声学、电磁学、光学及其组合。SIMP(Simplied Isotropic Material with Penalization)是目前广泛传播的一种简单而高效的拓扑优化求解方法。它通过对材料密度的中间值进行惩罚，提高了二元解的收敛性。

本案例通过深度学习的方式对拉普拉斯方程的2维形式进行求解。

## 2. 问题定义

拓扑优化问题：

```math
\begin{equation*}
\left\{
\begin{aligned}
& \underset{\boldsymbol{x}}{\text{min}} \quad &&c(\boldsymbol{u}(\boldsymbol{x}), \boldsymbol{x}) = \sum_{j=1}^{N} E_{j}(x_{j})\boldsymbol{u}_{j}^{\intercal}\boldsymbol{k}_{0}\boldsymbol{u}_{j} \\
& \text{s.t.} \quad &&V(\boldsymbol{x})/V_{0} = f_{0} \\
& \quad && \boldsymbol{K}\boldsymbol{U} = \boldsymbol{F} \\
& \quad && x_{j} \in \{0, 1\}, \quad j = 1,...,N
\end{aligned}
\right.
\end{equation*}
```
其中：$x_{j}$ 是材料分布(material distribution)；$c$ 指可塑性(compliance)；$\boldsymbol{u}_{j}$ 是 element displacement vector；$\boldsymbol{k}_{0}$ 是element stiffness matrix for an element with unit Youngs modulu；$\boldsymbol{U}$, $\boldsymbol{F}$ 是 global displacement and force vectors；$\boldsymbol{K}$ 是 global stiffness matrix；$V(\boldsymbol{x})$, $V_{0}$ 是材料体积和设计区域的体积；$f_{0}$ 是预先指定的体积比。

## 3. 问题求解

实际求解上述问题时为做简化，会把最后一个约束条件换成连续的形式：$x_{j} \in [0, 1], \quad j = 1,...,N$。 常见的优化算法是 SIMP 算法，它是一种基于梯度的迭代法，并对非二元解做惩罚：$E_{j}(x_{j}) = E_{\text{min}} + x_{j}^{p}(E_{0} - E_{\text{min}})$，这里我们不对 SIMP 算法做过多展开。由于利用 SIMP 方法, 求解器只需要进行初始的 $N_{0}$ 次迭代就可以得到与结果的最终结果非常相近的基本视图，本案例希望通过将 SIMP 的第 $N_{0}$ 次初始迭代结果与其对应的梯度信息作为 Unet 的输入，预测 SIMP 的100次迭代步骤后给出的优化解。

### 3.1 数据集准备

数据集为整理过的合成数据：[下载地址](https://aistudio.baidu.com/datasetdetail/245115/0)  
整理后的格式为 `"iters": shape = (10000, 100, 40, 40)`，`"target": shape = (10000, 1, 40, 40)`
- 10000 - 随机生成问题的个数
- 100 - SIMP 迭代次数
- 40 - 图像高度
- 40 - 图像宽度

数据集地址请存储于 `./Dataset/PreparedData/top_dataset.h5`  
生成训练集：原始代码利用所有的10000问题生成训练数据

``` py linenums="51"
--8<--
examples/topopt/functions.py:51:80
--8<--
```

``` py linenums="41"
--8<--
examples/topopt/topopt.py:41:45
--8<--
```

### 3.2 模型构建

经过 SIMP 的 $N_{0}$ 次初始迭代步骤得到的图像 $I$ 可以看作是模糊了的最终结构。由于最终的优化解给出的图像 $I^*$ 并不包含中间过程的信息，因此 $I^*$ 可以被解释为图像 $I$ 的掩码。于是 $I \rightarrow I^*$ 这一优化过程可以看作是二分类的图像分割或者前景-背景分割过程，因此构建 Unet 模型进行预测，具体网络结构如下：

``` py linenums="21"
--8<--
examples/laplace/TopOptModel.py:21:154
--8<--
```

### 3.3 参数设定

根据论文以及原始代码给出以下训练参数：

``` py linenums="35"
--8<--
examples/topopt/topopt.py:35:40
--8<--
```

``` yaml linenums="32"
--8<--
examples/topopt/conf/topopt.yaml:32:45
--8<--
```


### 3.4 transform构建

根据论文以及原始代码给出以下自定义transform代码，包括随机水平或垂直翻转和随机90度旋转，对input和label同时transform：

``` py linenums="81"
--8<--
examples/topopt/functions.py:81:133
--8<--
```

### 3.5 损失构建

损失包括 confidence loss + beta * volume fraction constraints:  
$\mathcal{L} = \mathcal{L}_{\text{conf}}(X_{\text{true}}, X_{\text{pred}}) + \beta * \mathcal{L}_{\text{vol}}(X_{\text{true}}, X_{\text{pred}})$  
confidence loss 是 binary cross-entropy:  
$\mathcal{L}_{\text{conf}}(X_{\text{true}}, X_{\text{pred}}) = -\frac{1}{NM}\sum_{i=1}^{N}\sum_{j=1}^{M}\left[X_{\text{true}}^{ij}\log(X_{\text{pred}}^{ij}) +  (1 - X_{\text{true}}^{ij})\log(1 - X_{\text{pred}}^{ij})\right]$  
volume fraction constraints:  
$\mathcal{L}_{\text{vol}}(X_{\text{true}}, X_{\text{pred}}) = (\bar{X}_{\text{pred}} - \bar{X}_{\text{true}})^2$  
代码如下：

``` py linenums="46"
--8<--
examples/topopt/topopt.py:46:55
--8<--
```

### 3.6 约束构建

在本案例中，我们采用监督学习方式进行训练，所以使用监督约束 `SupervisedConstraint`，代码如下：

``` py linenums="56"
--8<--
examples/topopt/topopt.py:56:82
--8<--
```

### 3.7 采样器构建

原始数据有100个通道对应的是 SIMP 算法100次的迭代结果，本案例模型目标是用 SIMP 中间某一步的迭代结果直接预测 SIMP 最后一步的迭代结果，而论文原始代码中的模型输入是原始数据对通道进行采样后的数据，为应用PaddleScience API，本案例将采样步骤放入模型的 forward 方法中，所以需要传入不同的采样器。

``` py linenums="20"
--8<--
examples/topopt/functions.py:20:50
--8<--
```

### 3.8 模型训练

本案例根据采样器的不同选择共有四组子案例，案例参数如下：

``` yaml linenums="18"
--8<--
examples/topopt/conf/topopt.yaml:18:20
--8<--
```

优化器选用 Adam，训练代码如下：

``` py linenums="83"
--8<--
examples/topopt/topopt.py:83:127
--8<--
```

### 3.9 metric构建

本案例选择 binary accuracy 和 IoU 进行评估:  
$\text{Bin. Acc.} = \frac{w_{00}+w_{11}}{n_{0}+n_{1}}$  
$\text{IoU} = \frac{1}{2}\left[\frac{w_{00}}{n_{0}+w_{10}} + \frac{w_{11}}{n_{1}+w_{01}}\right]$  
$n_{0} = w_{00} + w_{01}, \quad n_{1} = w_{10} + w_{11}$

``` py linenums="151"
--8<--
examples/topopt/topopt.py:151:194
--8<--
```

### 3.10 评估模型

对四个训练好的模型，分别使用不同的固定通道采样器(取原始数据的第5，10，15，20，...，80通道作为输入)进行评估，每次评估时只取 num_val_step 个 bacth 的数据；为应用PaddleScience API，此处在每一次评估时构建一个评估器 SupervisedValidator 进行评估：

``` py linenums="195"
--8<--
examples/topopt/topopt.py:195:256
--8<--
```

根据原始代码，评估还需要加上输入的阈值判定结果(0.5作为阈值)：

``` py linenums="196"
--8<--
examples/topopt/topopt.py:196:282
--8<--
```

### 3.11 评估结果可视化

使用 matplotlib 简单绘制 Binary Accuracy 和 IoU 的结果：

``` py linenums="283"
--8<--
examples/topopt/topopt.py:283:306
--8<--
```

## 4. 完整代码

``` py linenums="1" title="topopt.py"
--8<--
examples/topopt/topopt.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/topopt/functions.py
--8<--
```

``` py linenums="1" title="TopOptModel.py"
--8<--
examples/topopt/TopOptModel.py
--8<--
```

## 5. 结果展示

<figure markdown>
  ![bin_acc](https://ai-studio-static-online.cdn.bcebos.com/a1d08919315a47738332aff7aa2e09bb8899c5160cc5467d8b771faea418d0c6){ loading=lazy }
  <figcaption>Binary Accuracy结果</figcaption>
  ![iou](https://ai-studio-static-online.cdn.bcebos.com/c6acd26d35624992b8cd8aad23822616bbe037f415634c1eb335e9d00bff9a07)
  <figcaption>IoU结果</figcaption>
</figure>

结果与[原始代码结果](https://github.com/ISosnovik/nn4topopt/blob/master/results.ipynb)基本一致

## 参考文献
* [Sosnovik I, & Oseledets I. Neural networks for topology optimization](https://arxiv.org/pdf/1709.09578)
* [原始代码](https://github.com/ISosnovik/nn4topopt/blob/master/)
