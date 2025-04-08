# Bracket

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar
    # unzip it
    tar -xvf bracket_dataset.tar
    python bracket.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar
    # unzip it
    tar -xvf bracket_dataset.tar
    python bracket.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/bracket/bracket_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python bracket.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar
    # unzip it
    tar -xvf bracket_dataset.tar
    python bracket.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [bracket_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/bracket/bracket_pretrained.pdparams) | loss(commercial_ref_u_v_w_sigmas): 32.28704<br>MSE.u(commercial_ref_u_v_w_sigmas): 0.00005<br>MSE.v(commercial_ref_u_v_w_sigmas): 0.00000<br>MSE.w(commercial_ref_u_v_w_sigmas): 0.00734<br>MSE.sigma_xx(commercial_ref_u_v_w_sigmas): 27.64751<br>MSE.sigma_yy(commercial_ref_u_v_w_sigmas): 1.23101<br>MSE.sigma_zz(commercial_ref_u_v_w_sigmas): 0.89106<br>MSE.sigma_xy(commercial_ref_u_v_w_sigmas): 0.84370<br>MSE.sigma_xz(commercial_ref_u_v_w_sigmas): 1.42126<br>MSE.sigma_yz(commercial_ref_u_v_w_sigmas): 0.24510 |

## 1. 背景简介

线弹性方程在形变分析中起着核心的作用。在物理和工程领域，形变分析是研究物体在外力作用下的形状和尺寸变化的方法。线弹性方程是描述物体在受力后恢复原状的能力的数学模型。具体来说，线弹性方程通常是指应力和应变之间的关系。应力是一个物理量，用于描述物体内部由于外力而产生的单位面积上的力。应变则描述了物体的形状和尺寸的变化。线弹性方程通常可以表示为应力和应变之间的线性关系，即应力和应变是成比例的。这种关系可以用一个线性方程来表示，其中系数被称为弹性模量（或杨氏模量）。这种模型假设物体在受力后能够完全恢复原状，即没有永久变形。这种假设在许多情况下是合理的，例如在研究金属的力学行为时。然而，对于某些材料（如塑料或橡胶），这种假设可能不准确，因为它们在受力后可能会产生永久变形。线弹性方程只是形变分析中的一部分。要全面理解形变，还需要考虑其他因素，例如物体的初始形状和尺寸、外力的历史、材料的其他物理性质（如热膨胀系数和密度）等。然而，线弹性方程提供了一个基本的框架，用于描述和理解物体在受力后的行为。

本案例主要研究如下金属连接件在给定载荷下的形变情况，并使用深度学习方法根据线弹性等方程进行求解，连接件如下所示（参考 [Matlab deflection-analysis-of-a-bracket](https://www.mathworks.com/help/pde/ug/deflection-analysis-of-a-bracket.html)）。

<figure markdown>
  ![bracket](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/stl.png){ loading=lazy }
  <figcaption>Bracket 金属件载荷示意图，红色区域表示载荷面</figcaption>
</figure>

## 2. 问题定义

上述连接件包括一个垂直于 x 轴的背板和与之连接的垂直于 z 轴的带孔平板。其中背板处于固定状态，带孔平板的最右侧表面（红色区域）受到 z 轴负方向，单位面积大小为 $4 \times 10^4 Pa$ 的应力；除此之外，其他参数包括弹性模量 $E=10^{11} Pa$，泊松比 $\nu=0.3$。通过设置特征长度 $L=1m$，特征位移 $U=0.0001m$，无量纲剪切模量 $0.01\mu$，目标求解该金属件表面每个点的 $u$、$v$、$w$、$\sigma_{xx}$、$\sigma_{yy}$、$\sigma_{zz}$、$\sigma_{xy}$、$\sigma_{xz}$、$\sigma_{yz}$ 共 9 个物理量。常量定义代码如下：

``` py linenums="21"
--8<--
examples/bracket/bracket.py:21:30
--8<--
```

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 bracket 问题中，每一个已知的坐标点 $(x, y, z)$ 都有对应的待求解的未知量：三个方向的应变 $(u, v, w)$ 和应力 $(\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz})$。

这里考虑到两组物理量对应着不同的方程，因此使用两个模型来分别预测这两组物理量：

$$
\begin{cases}
u, v, w = f(x,y,z) \\
\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz} = g(x,y,z)
\end{cases}
$$

上式中 $f$ 即为应变模型 `disp_net`，$g$ 为应力模型 `stress_net`，用 PaddleScience 代码表示如下：

``` py linenums="15"
--8<--
examples/bracket/bracket.py:15:19
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定应变模型的输入变量名是 `("x", "y", "z")`，输出变量名是 `("u", "v", "w")`，这些命名与后续代码保持一致（应力模型同理）。

接着通过指定 MLP 的层数、神经元个数，就实例化出了一个拥有 6 层隐藏神经元，每层神经元数为 512 的神经网络模型 `disp_net`，使用 `silu` 作为激活函数，并使用 `WeightNorm` 权重归一化（应力模型 `stress_net` 同理）。

### 3.2 方程构建

Bracket 案例涉及到以下线弹性方程，使用 PaddleScience 内置的 `LinearElasticity` 即可。

--8<--
ppsci/equation/pde/linear_elasticity.py:30:42
--8<--

对应的方程实例化代码如下：

``` py linenums="32"
--8<--
examples/bracket/bracket.py:32:37
--8<--
```

### 3.3 计算域构建

本问题的几何区域由 stl 文件指定，按照下方命令，下载并解压到 `bracket/` 文件夹下。

**注：数据集中的 stl 文件和测试集数据均来自 [Bracket - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html#linear-elasticity-in-the-differential-form)**。

``` sh
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar

# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar

# unzip it
tar -xvf bracket_dataset.tar
```

解压完毕之后，`bracket/stl` 文件夹下即存放了计算域构建所需的 stl 几何文件。

???+ warning "注意"

    **使用 `Mesh` 类之前，必须先按照[1.4.2 安装Mesh几何[可选]](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/install_setup/#142-mesh)文档，安装好 open3d、pysdf、PyMesh 3 个几何依赖包。**

然后通过 PaddleScience 内置的 STL 几何类 `Mesh` 来读取、解析这些几何文件，并且通过布尔运算，组合出各个计算域，代码如下：

``` py linenums="39"
--8<--
examples/bracket/bracket.py:39:51
--8<--
```

### 3.4 约束构建

本案例共涉及到 5 个约束，在具体约束构建之前，可以先构建数据读取配置，以便后续构建多个约束时复用该配置。

``` py linenums="53"
--8<--
examples/bracket/bracket.py:53:63
--8<--
```

#### 3.4.1 内部点约束

以作用在背板内部点的 `InteriorConstraint` 为例，代码如下：

``` py linenums="106"
--8<--
examples/bracket/bracket.py:106:142
--8<--
```

`InteriorConstraint` 的第一个参数是方程（组）表达式，用于描述如何计算约束目标，此处填入在 [3.2 方程构建](#32) 章节中实例化好的 `equation["LinearElasticity"].equations`；

第二个参数是约束变量的目标值，在本问题中希望与 LinearElasticity 方程相关的 9 个值 `equilibrium_x`, `equilibrium_y`, `equilibrium_z`, `stress_disp_xx`, `stress_disp_yy`, `stress_disp_zz`, `stress_disp_xy`, `stress_disp_xz`, `stress_disp_yz` 均被优化至 0；

第三个参数是约束方程作用的计算域，此处填入在 [3.3 计算域构建](#33) 章节实例化好的 `geom["geo"]` 即可；

第四个参数是在计算域上的采样配置，此处设置 `batch_size` 为 `2048`。

第五个参数是损失函数，此处选用常用的 MSE 函数，且 `reduction` 设置为 `"sum"`，即会将参与计算的所有数据点产生的损失项求和；

第六个参数是几何点筛选，由于这个约束只施加在背板区域，因此需要对 geo 上采样出的点进行筛选，此处传入一个 lambda 筛选函数即可，其接受点集构成的张量 `x, y, z`，返回布尔值张亮，表示每个点是否符合筛选条件，不符合为 `False`，符合为 `True`；

第七个参数是每个点参与损失计算时的权重，此处我们使用 `"sdf"` 表示使用每个点到边界的最短距离（符号距离函数值）来作为权重，这种 sdf 加权的方法可以加大远离边界（难样本）点的权重，减少靠近边界的（简单样本）点的权重，有利于提升模型的精度和收敛速度。

第八个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "support_interior" 即可。

另一个作用在带孔平板上的约束条件则与之类似，代码如下：

``` py linenums="143"
--8<--
examples/bracket/bracket.py:143:179
--8<--
```

#### 3.4.2 边界约束

对于背板后表面，由于被固定，所以其上的点在三个方向的形变均为 0，因此有如下的边界约束条件：

``` py linenums="76"
--8<--
examples/bracket/bracket.py:76:85
--8<--
```

对于带孔平板右侧长方形载荷面，其上的每个点只受 z 正方向的载荷，大小为 $T$，其余方向应力为 0，有如下边界条件约束：

``` py linenums="86"
--8<--
examples/bracket/bracket.py:86:94
--8<--
```

对于除背板后面、带孔平板右侧长方形载荷面外的表面，不受任何载荷，即三个方向的内力平衡，合力为 0，有如下边界条件约束：

``` py linenums="95"
--8<--
examples/bracket/bracket.py:95:105
--8<--
```

在方程约束、边界约束构建完毕之后，以刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="180"
--8<--
examples/bracket/bracket.py:180:187
--8<--
```

### 3.5 超参数设定

接下来需要在配置文件中指定训练轮数，此处按实验经验，使用 2000 轮训练轮数，每轮进行 1000 步优化。

``` yaml linenums="74"
--8<--
examples/bracket/conf/bracket.yaml:74:77
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="189"
--8<--
examples/bracket/bracket.py:189:193
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，而验证集的数据来自外部 txt 文件，因此首先使用 `ppsci.utils.reader` 模块从 txt 文件中读取验证点集：

``` py linenums="195"
--8<--
examples/bracket/bracket.py:195:256
--8<--
```

然后将其转换为字典并进行无量纲化和归一化，再将其包装成字典和 `eval_dataloader_cfg`（验证集dataloader配置，构造方式与 `train_dataloader_cfg` 类似）一起传递给 `ppsci.validate.SupervisedValidator` 构造评估器。

``` py linenums="258"
--8<--
examples/bracket/bracket.py:258:303
--8<--
```

### 3.8 可视化器构建

在模型评估时，如果评估结果是可以可视化的数据，可以选择合适的可视化器来对输出结果进行可视化。

本文中的输入数据是评估器构建中准备好的输入字典 `input_dict`，输出数据是对应的 9 个预测的物理量，因此只需要将评估的输出数据保存成 **vtu格式** 文件，最后用可视化软件打开查看即可。代码如下：

``` py linenums="305"
--8<--
examples/bracket/bracket.py:305:322
--8<--
```

### 3.9 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="324"
--8<--
examples/bracket/bracket.py:324:351
--8<--
```

## 4. 完整代码

``` py linenums="1" title="bracket.py"
--8<--
examples/bracket/bracket.py
--8<--
```

## 5. 结果展示

下面展示了在测试点集上，3 个方向的挠度 $u, v, w$ 以及 6 个应力 $\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz}$ 的模型预测结果、传统算法求解结果以及两者的差值。

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/u.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的挠度 u；中间表示传统算法求解的挠度 u；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/v.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的挠度 v；中间表示传统算法求解的挠度 v；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/w.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的挠度 w；中间表示传统算法求解的挠度 w；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_xx.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的应力 sigma_xx；中间表示传统算法求解的应力 sigma_xx；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_xy.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的应力 sigma_xy；中间表示传统算法求解的应力 sigma_xy；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_xz.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的应力 sigma_xz；中间表示传统算法求解的应力 sigma_xz；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_yy.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的应力 sigma_yy；中间表示传统算法求解的应力 sigma_yy；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_yz.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的应力sigma_yz；中间表示传统算法求解的应力sigma_yz；右侧表示两者差值</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_zz.png){ loading=lazy }
  <figcaption>左侧为金属件表面预测的应力sigma_zz；中间表示传统算法求解的应力sigma_zz；右侧表示两者差值</figcaption>
</figure>

可以看到模型预测的结果与 传统算法求解结果基本一致。

## 6. 参考资料

- [Bracket - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html)
- [Scaling of Differential Equations](https://hplgit.github.io/scaling-book/doc/pub/book/html/sphinx-cbc/index.html)
- [Matlab PDE toolbox](https://www.mathworks.com/help/pde/ug/deflection-analysis-of-a-bracket.html)
