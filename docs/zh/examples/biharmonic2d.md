# 2D-Biharmonic

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

=== "模型训练命令"

    ``` sh
    python biharmonic2d.py
    ```

=== "模型评估命令"

    ``` sh
    python biharmonic2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/biharmonic2d/biharmonic2d_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [biharmonic2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/biharmonic2d/biharmonic2d_pretrained.pdparams) | l2_error: 0.02774 |

## 1. 背景简介

双调和方程（Biharmonic Equation）是一种表征应力、应变和载荷之间关系的方程，它是一种四阶偏微分方程，因此在传统数值方法中难以解决。本案例尝试使用 PINNs(Physics Informed Neural Networks) 方法解决 Biharmonic 方程在 2D 矩形平板上的应用问题，并使用深度学习方法根据线弹性等方程进行求解。

## 2. 问题定义

本案例结构为一个长、宽和厚分别为 2 m、3 m 和 0.01 m 的矩形平板，平板四周固定，表面则被施加一个正弦分布载荷 $q=q_0sin(\dfrac{\pi x}{a})sin(\dfrac{\pi x}{b})$，其中 $q_0=980 Pa$。PDE 方程为 2D 下的 Biharmonic 方程，公式为：

$$\nabla^4w=(\dfrac{\partial^2}{\partial x^2}+\dfrac{\partial^2}{\partial y^2})(\dfrac{\partial^2}{\partial x^2}+\dfrac{\partial^2}{\partial y^2})w=\dfrac{q}{D}$$

其中 $w$ 为平板挠度，$D$ 为抗弯刚度，可计算如下：

$$D=\dfrac{Et^3}{12(1-\nu^2)}$$

其中 $E=201880.0e+6 Pa$ 为弹性杨氏模量，$\nu=0.25$ 为泊松比。

根据平板挠度$w$，可计算扭矩和剪切力如下：

$$
\begin{cases}
  M_x=-D(\dfrac{\partial^2w}{\partial x^2}+\nu\dfrac{\partial^2w}{\partial y^2}) \\
  M_y=-D(\dfrac{\partial^2w}{\partial y^2}+\nu\dfrac{\partial^2w}{\partial x^2}) \\
  M_{xy}=D(1-\nu\dfrac{\partial^2w}{\partial x y}) \\
  Q_x=-D\dfrac{\partial}{\partial x}(\dfrac{\partial^2w}{\partial x^2}+\dfrac{\partial^2w}{\partial y^2}) \\
  Q_y=-D\dfrac{\partial}{\partial y}(\dfrac{\partial^2w}{\partial x^2}+\dfrac{\partial^2w}{\partial y^2}) \\
\end{cases}
$$

由于平板四周固定，在 $x=0$ 和 $x=x_{max}$ 上，挠度 $w$ 和 $y$ 方向的力矩 $M_y$ 为 0，在 $y=0$ 和 $y=y_{max}$ 上， 挠度 $w$ 和 $x$ 方向的力矩 $M_x$ 为 0，即：

$$
\begin{cases}
  w|_{x=0\ |\ x=\ a}=0 \\
  M_y|_{x=0\ |\ x=\ a}=0 \\
  w|_{y=0\ |\ y=\ b}=0 \\
  M_x|_{y=0\ |\ y=\ b}=0 \\
\end{cases}
$$

目标求解该平板表面每个点的挠度 $w$，并以此计算出力矩和剪切力 $M_x$、$M_y$、$M_{xy}$、$Q_x$、$Q_y$ 共 6 个物理量。常量定义代码如下：

``` yaml linenums="28"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:28:34
--8<--
```

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 biharmonic2d 问题中，每一个已知的坐标点 $(x, y)$ 都有对应的待求解的未知量：受力方向（即 z 方向）的挠度 $w$ 和力矩 $(M_x, M_y, M_{xy})$ 、剪切力 $(Q_x, Q_y)，但由于力矩和剪切力为挠度计算得到，实际需要求出的未知量只有挠度 $w$，因此仅需构建一个模型：

$$w = f(x,y)$$

上式中 $f$ 即为挠度模型 `disp_net`，用 PaddleScience 代码表示如下：

``` py linenums="77"
--8<--
examples/biharmonic2d/biharmonic2d.py:77:78
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定应变模型的输入变量名是 `("x", "y")`，为了与 PaddleScience 内置方程 API ppsci.equation.Biharmonic 匹配，输出变量名是 `("u")` 而不是 `("w")` ，这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数，就实例化出了一个拥有 5 层隐藏神经元，每层神经元数为 20 的神经网络模型 `disp_net`，使用 `tanh` 作为激活函数，并使用 `WeightNorm` 权重归一化。

### 3.2 方程构建

本案例涉及到双调和方程，使用 PaddleScience 内置的 `ppsci.equation.Biharmonic` 即可，由于载荷 $q$ 为非均匀载荷，需要自定义载荷分布函数，并传入 API。

``` py linenums="84"
--8<--
examples/biharmonic2d/biharmonic2d.py:84:91
--8<--
```

### 3.3 计算域构建

由于平板的高很小，本问题的几何区域认为是长为 2 宽为 3 的 2D 矩形，通过 PaddleScience 内置的 `ppsci.geometry.Rectangle` API 构建：

``` py linenums="93"
--8<--
examples/biharmonic2d/biharmonic2d.py:93:95
--8<--
```

### 3.4 约束构建

本案例共涉及到 9 个约束，在具体约束构建之前，可以先构建数据读取配置，以便后续构建多个约束时复用该配置。

``` py linenums="97"
--8<--
examples/biharmonic2d/biharmonic2d.py:97:106
--8<--
```

#### 3.4.1 内部约束

以作用在背板内部点的 `InteriorConstraint` 为例，代码如下：

``` py linenums="205"
--8<--
examples/biharmonic2d/biharmonic2d.py:205:214
--8<--
```

`InteriorConstraint` 的第一个参数是方程（组）表达式，用于描述如何计算约束目标，此处填入在 [3.2 方程构建](#32) 章节中实例化好的 `equation["Biharmonic"].equations`；

第二个参数是约束变量的目标值，在本问题中希望与 Biharmonic 方程相关的 1 个值 `biharmonic` 被优化至 0；

第三个参数是约束方程作用的计算域，此处填入在 [3.3 计算域构建](#33) 章节实例化好的 `geom["geo"]` 即可；

第四个参数是在计算域上的采样配置，此处设置 `batch_size` 为：

``` yaml linenums="57"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:57:59
--8<--
```

第五个参数是损失函数，此处选用常用的 MSE 函数，且 `reduction` 设置为 `"mean"`，即会将参与计算的所有数据点产生的损失项求和；

第六个参数是几何点筛选，由于这个约束只施加在背板区域，因此需要对 geo 上采样出的点进行筛选，此处传入一个 lambda 筛选函数即可，其接受点集构成的张量 `x, y`，返回布尔值张亮，表示每个点是否符合筛选条件，不符合为 `False`，符合为 `True`；

第七个参数是每个点参与损失计算时的权重，此处设置为：

``` yaml linenums="60"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:60:62
--8<--
```

第八个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "INTERIOR" 即可。

#### 3.4.2 边界约束

如 [2. 问题定义](#2) 中所述，$x=0$ 处的挠度 $w$ 为 0，有如下边界条件，其他 7 个边界条件也与之类似：

``` py linenums="108"
--8<--
examples/biharmonic2d/biharmonic2d.py:108:118
--8<--
```

在方程约束、边界约束构建完毕之后，以刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="215"
--8<--
examples/biharmonic2d/biharmonic2d.py:215:226
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择使用 `Adam` 先进行少量训练后，再使用 `LBFGS` 优化器精调。

``` py linenums="81"
--8<--
examples/biharmonic2d/biharmonic2d.py:81:83
--8<--
```

### 3.6 超参数设定

接下来需要在配置文件中指定训练轮数和学习率等优化器参数。

``` yaml linenums="46"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:46:56
--8<--
```

### 3.7 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练，注意两个优化过程需要分别构建 `Solver`。

``` py linenums="228"
--8<--
examples/biharmonic2d/biharmonic2d.py:228:267
--8<--
```

### 3.8 模型评估和可视化

训练完成后，可以在 `eval` 模式中对训练好的模型进行评估和可视化。由于案例的特殊性，不需构建评估器和可视化器，而是使用自定义代码。

``` py linenums="270"
--8<--
examples/biharmonic2d/biharmonic2d.py:270:350
--8<--
```

## 4. 完整代码

``` py linenums="1" title="biharmonic2d.py"
--8<--
examples/biharmonic2d/biharmonic2d.py
--8<--
```

## 5. 结果展示

下面展示了挠度 $w$ 以及力矩 $M_x, M_y, M_{xy}$ 和剪切力 $Q_x, Q_y$ 的模型预测结果和理论解结果。

<figure markdown>
  ![biharmonic2d_pred.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/biharmonic2d/eval_Mx_Mxy_My_Qx_Qy_w.png){ loading=lazy }
  <figcaption>力矩 Mx, My, Mxy、剪切力 Qx, Qy 和挠度 w 的模型预测结果</figcaption>
</figure>

<figure markdown>
  ![biharmonic2d_label_M.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/biharmonic2d/label_M.png){ loading=lazy }
  <figcaption>力矩 Mx, My, Mxy 的理论解结果</figcaption>
</figure>

<figure markdown>
  ![biharmonic2d_label_Q_w.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/biharmonic2d/label_Q_w.png){ loading=lazy }
  <figcaption>剪切力 Qx, Qy 和挠度 w 的理论解结果</figcaption>
</figure>

可以看到模型预测的结果与理论解结果基本一致。

## 6. 参考文献

参考文献：[A Physics Informed Neural Network Approach to Solution and Identification of Biharmonic Equations of Elasticity](https://arxiv.org/abs/2108.07243)
