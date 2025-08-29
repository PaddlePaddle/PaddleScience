# Heart

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

=== "模型训练命令"

    === "正问题：受力分析求解"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python forward.py
        ```

    === "逆问题：参数逆推求解"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python inverse.py TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams
        ```

=== "模型评估命令"

    === "正问题：受力分析求解"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python forward.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/forward_pretrained.pdparams
        ```

    === "逆问题：参数逆推求解"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python inverse.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams EVAL.param_E_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/param_E.pdparams
        ```

| 预训练模型  | 指标 |
|:--| :--|
| [forward_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heart/forward_pretrained.pdparams) | loss(ref_u_v_w): 0.00076<br>L2Rel.u(ref_u_v_w): 0.01162<br>L2Rel.v(ref_u_v_w): 0.00511<br>L2Rel.w(ref_u_v_w): 0.00737 |
| [inverse_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams)<br>[param_E.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heart/param_E.pdparams) | loss(ref_u_v_w): 0.00576<br>L2Rel.u(ref_u_v_w): 0.03082<br>L2Rel.v(ref_u_v_w): 0.01412<br>L2Rel.w(ref_u_v_w): 0.02075<br>L2_Error(E): 0.04975 |

## 1. 背景简介

心血管疾病已成为威胁人类健康的头号杀手，基于个体影像的生物力学建模在理解心脏疾病和发展新型诊疗方案的过程中发挥了重要作用。然而，在心脏生物力学建模领域，传统的有限元方法存在网格划分繁琐、求解速度慢等问题，限制了其在个体化的心脏建模与诊疗中的应用。内嵌物理知识的神经网络(Physics Informed Neural Network, PINN)是近年来兴起的一种基于神经网络求解偏微分方程的算法，在流体力学领域已取得一定成果，受到了大量研究者的关注，具有广阔的应用前景，通过 PINN 对心脏生物力学方程进行求解和计算进行仿真，可以大幅提高个体心脏建模效率。

本案例使用 PINN 网络，在某一个体化心脏的左心室模型上，根据弹性力学理论给出了心脏模型的位移场满足的线弹性本构关系、几何方程、平衡方程和边界条件，以及将相同条件下有限元仿真结果的真实位移作为约束，训练了一个求解左心室线弹性 Hooke 定律中的两个材料参数的 PINN 网络。

## 2. 问题定义

模型输入为左心室变形前网格点的坐标 $(x,y,z)$ ，输出为左心室舒张变形后网格点对应的位移 $(u,v,w)$。

在本案例中，认为心脏属于线弹性材料，满足线弹性材料的 Hooke 定律，即：

$$
\begin{pmatrix}
    t_{xx} \\ t_{yy} \\ t_{zz} \\ t_{xy} \\ t_{xz} \\ t_{yz} \\
\end{pmatrix}
=
\begin{bmatrix}
    \frac{1}{E} & -\frac{\nu}{E} & -\frac{\nu}{E} & 0 & 0 & 0 \\
    -\frac{\nu}{E} & \frac{1}{E} & -\frac{\nu}{E} & 0 & 0 & 0 \\
    -\frac{\nu}{E} & -\frac{\nu}{E} & \frac{1}{E} & 0 & 0 & 0 \\
    0 & 0 & 0 & \frac{1}{G} & 0 & 0 \\
    0 & 0 & 0 & 0 & \frac{1}{G} & 0 \\
    0 & 0 & 0 & 0 & 0 & \frac{1}{G} \\
\end{bmatrix}
\begin{pmatrix}
    \varepsilon _{xx} \\ \varepsilon _{yy} \\ \varepsilon _{zz} \\ \varepsilon _{xy} \\ \varepsilon _{xz} \\ \varepsilon _{yz} \\
\end{pmatrix}
$$

其中 $G=\frac{E}{2(1+\nu)}$，$E=9kpa$ 和 $\nu=0.45$ 为两个独立的常数，$\sigma_{xx}$、$\sigma_{yy}$、$\sigma_{zz}$、$\sigma_{xy}$、$\sigma_{xz}$、$\sigma_{yz}$ 为对应坐标点三个维度上的应力，它与位移的关系为：

$$
\begin{pmatrix}
    \sigma_{xx} = \frac{\partial u}{\partial x} \\
    \sigma_{yy} = \frac{\partial v}{\partial y} \\
    \sigma_{zz} = \frac{\partial w}{\partial z} \\
    \sigma_{xy} = \frac{1}{2}(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}) \\
    \sigma_{xz} = \frac{1}{2}(\frac{\partial u}{\partial z}+\frac{\partial w}{\partial x}) \\
    \sigma_{yz} = \frac{1}{2}(\frac{\partial v}{\partial z}+\frac{\partial w}{\partial y}) \\
\end{pmatrix}
$$

在该案例中，认为舒张期左心室被动力学是准静态的，因此认为该案例具有如下边界条件：

1. 在整个几何计算域上需要满足 $\nabla t=0$，即：

$$
\begin{pmatrix}
    \frac{\partial t_{xx}}{\partial x}+\frac{\partial t_{xy}}{\partial y}+\frac{\partial t_{xz}}{\partial z}=0 \\
    \frac{\partial t_{xy}}{\partial x}+\frac{\partial t_{yy}}{\partial y}+\frac{\partial t_{yz}}{\partial z}=0 \\
    \frac{\partial t_{xz}}{\partial x}+\frac{\partial t_{yz}}{\partial y}+\frac{\partial t_{zz}}{\partial z}=0 \\
\end{pmatrix}
$$

2. 在心内膜表面需满足 $tn=-P_{endo}n$，这里 $n$ 是心内膜表面的单位法线方向，$P_{endo}=1.064kpa(8mmHg)$ 代表左心室空腔压力;

3. 在心脏外膜上需满足 $P_{epi}=0$;

4. 在基面上需满足 $u_{x,y,z}=0$，即 $u,v,w=0$

<figure markdown>
  ![boundary conditions](https://paddle-org.bj.bcebos.com/paddlescience/docs/heart/doc1.png){ loading=lazy }
  <figcaption>边界条件示意图</figcaption>
</figure>

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 受力分析求解

#### 3.1.1 模型构建

如上所述，每一个已知的坐标点 $(x, y, z)$ 都有对应的待求解的应变 $(u, v, w)$，使用一个模型来预测：

$u, v, w = f(x,y,z)$

用 PaddleScience 代码表示如下：

``` py linenums="24"
--8<--
examples/heart/forward.py:24:26
--8<--
```

#### 3.1.2 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="27"
--8<--
examples/heart/forward.py:27:32
--8<--
```

#### 3.1.3 方程构建

在 equation.py 文件中构建方程，对应的方程实例化代码如下：

``` py linenums="33"
--8<--
examples/heart/forward.py:33:35
--8<--
```

#### 3.1.4 计算域构建

本问题的几何区域由 stl 文件指定，按照本文档起始处"模型训练命令"下载并解压到 `./stl/` 文件夹下。

???+ warning "注意"

    **使用 `Mesh` 类之前，必须先按照[1.4.2 安装Mesh几何[可选]](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/install_setup/#142-mesh)文档，安装好 open3d、pysdf、PyMesh 3 个几何依赖包。**

然后通过 PaddleScience 内置的 STL 几何类 `ppsci.geometry.Mesh` 即可读取、解析几何文件，得到计算域，并获取几何结构边界：

``` py linenums="36"
--8<--
examples/heart/forward.py:36:44
--8<--
```

#### 3.1.5 超参数设定

接下来需要在配置文件中指定训练轮数，此处按实验经验，使用 200 轮训练轮数，每轮进行 1000 步优化。

``` yaml linenums="61"
--8<--
examples/heart/conf/forward.yaml:61:62
--8<--
```

#### 3.1.6 约束构建

本问题共涉及到[2. 问题定义](#2)中的 4 个约束，在具体约束构建之前，可以先构建数据读取配置，以便后续构建多个约束时复用该配置。

``` py linenums="45"
--8<--
examples/heart/forward.py:45:56
--8<--
```

##### 3.1.6.1 内部点约束

以作用在结构内部点的 `InteriorConstraint` 为例，代码如下：

``` py linenums="85"
--8<--
examples/heart/forward.py:85:102
--8<--
```

`InteriorConstraint` 的第一个参数是方程（组）表达式，用于描述如何计算约束目标，此处填入在 [3.1.3 方程构建](#313) 章节中实例化好的 `equation["Hooke"].equations`；

第二个参数是约束变量的目标值，在本问题为 `hooke_x`, `hooke_y`, `hooke_z` 被优化至 0；

第三个参数是约束方程作用的计算域，此处填入在 [3.1.4 计算域构建](#314) 章节实例化好的 `geom["geo"]` 即可；

第四个参数是在计算域上的采样配置，此处设置 `batch_size` 为：

``` yaml linenums="70"
--8<--
examples/heart/conf/forward.yaml:70:74
--8<--
```

第五个参数是损失函数，此处选用常用的 MSE 函数，且 `reduction` 设置为 `"mean"`，即会将参与计算的所有数据点产生的损失项求均值；

第六个参数是几何点筛选，需要对 geo 上采样出的点进行筛选，此处传入一个 lambda 筛选函数即可，其接受点集构成的张量 `x, y, z`，返回布尔值张量，表示每个点是否符合筛选条件，不符合为 `False`，符合为 `True`，因为本案例结构来源于网络，参数不完全精确，因此增加 `1e-1` 作为可容忍的采样误差；

第七个参数是每个点参与损失计算时的权重；

第八个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "INTERIOR" 即可。

##### 3.1.6.2 边界约束

参照 [2. 问题定义](#2) 分别为心内膜、心外膜和基面上的约束：

``` py linenums="57"
--8<--
examples/heart/forward.py:57:84
--8<--
```

#### 3.1.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，而验证集的数据来自外部 csv 文件，因此首先使用 `ppsci.utils.reader` 模块从 csv 文件中读取验证点集：

``` py linenums="125"
--8<--
examples/heart/forward.py:125:137
--8<--
```

然后将其转换为字典并进行无量纲化和归一化，再将其包装成字典和 `eval_dataloader_cfg`（验证集dataloader配置，构造方式与 `train_dataloader_cfg` 类似）一起传递给 `ppsci.validate.SupervisedValidator` 构造评估器。

``` py linenums="138"
--8<--
examples/heart/forward.py:138:168
--8<--
```

#### 3.1.8 可视化器构建

在模型评估时，如果评估结果是可以可视化的数据，可以选择合适的可视化器来对输出结果进行可视化。

本文中的输入数据是评估器构建中准备好的输入字典 `input_dict`，输出数据是对应的 3 个预测的物理量，因此只需要将评估的输出数据保存成 **vtu格式** 文件，最后用可视化软件打开查看即可。代码如下：

``` py linenums="170"
--8<--
examples/heart/forward.py:170:182
--8<--
```

#### 3.1.8 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练：

``` py linenums="184"
--8<--
examples/heart/forward.py:184:200
--8<--
```

训练后调用 `ppsci.solver.Solver.plot_loss_history` 可以将训练中的 `loss` 画出：

``` py linenums="201"
--8<--
examples/heart/forward.py:201:203
--8<--
```

#### 3.1.9 模型评估与可视化

训练完成或下载预训练模型后，通过本文档起始处“模型评估命令”进行模型评估和可视化。

评估和可视化过程不需要进行优化器等构建，仅需构建模型、计算域、评估器（本案例不包括）、可视化器，然后按顺序传递给 `ppsci.solver.Solver` 启动评估和可视化：

``` py linenums="269"
--8<--
examples/heart/forward.py:269:275
--8<--
```

### 3.2 参数逆推求解

#### 3.2.1 方程构建

本案例尝试在 PINN 框架下对心脏软组织复杂的超弹性本构关系进行建模和实现，在预设方程参数 $E$ 未知的情况下，尝试通过部分数据，同时训练得到该未知参数的值。案例中依然使用 equation.py 文件中构建的方程，但需要将未知参数设为可学习变量，并传入方程：

``` py linenums="31"
--8<--
examples/heart/inverse.py:31:38
--8<--
```

#### 3.2.2 模型构建

模型设置与正问题相同：

``` py linenums="39"
--8<--
examples/heart/inverse.py:39:41
--8<--
```

#### 3.2.3 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 `ExponentialDecay` 学习率调整策略。在设置优化器时需要将方程中的可学习参数传递给优化器，来使该参数参与优化：

``` py linenums="42"
--8<--
examples/heart/inverse.py:42:47
--8<--
```

#### 3.2.4 其它设置

本问题的其它设置与正问题相似，在此不再赘述。

## 4. 完整代码

``` py linenums="1" title="forward.py"
--8<--
examples/heart/forward.py
--8<--
```

``` py linenums="1" title="inverse.py"
--8<--
examples/heart/inverse.py
--8<--
```

``` py linenums="1" title="equation.py"
--8<--
examples/heart/equation.py
--8<--
```

## 5. 结果展示

### 5.1 受力分析求解

下面展示了当力的方向为 x 正方向时 3 个方向的应变 $u, v, w$ 的模型预测结果，结果基本符合认知。

<figure markdown>
  ![forward_result.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/heart/doc2.jpg){ loading=lazy }
  <figcaption>左侧为预测的结构应变 u；中间为预测的结构应变 v；右侧为预测的结构应变 w</figcaption>
</figure>

### 5.2 参数逆推求解

下面展示了可学习方程参数 $E$ 的模型预测结果，误差在 5% 左右。

| data | E |
| :---: | :---: |
| outs | 9 |
| label | 9.44778 |
