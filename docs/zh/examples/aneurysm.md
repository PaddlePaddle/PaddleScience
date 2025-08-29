# Aneurysm

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar
    # unzip it
    tar -xvf aneurysm_dataset.tar
    python aneurysm.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar
    # unzip it
    tar -xvf aneurysm_dataset.tar
    python aneurysm.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python aneurysm.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar
    # unzip it
    tar -xvf aneurysm_dataset.tar
    python aneurysm.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [aneurysm_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams) | loss(ref_u_v_w_p): 0.01488<br>MSE.p(ref_u_v_w_p): 0.01412<br>MSE.u(ref_u_v_w_p): 0.00021<br>MSE.v(ref_u_v_w_p): 0.00024<br>MSE.w(ref_u_v_w_p): 0.00032 |

## 1. 背景简介

深度学习方法可以用于处理颅内动脉瘤问题，其中包括基于物理信息的深度学习方法。这种方法可以用于脑颅内动脉瘤的压力建模，以预测和评估颅内动脉瘤破裂的风险。

针对如下颅内动脉瘤几何模型，本案例通过深度学习方式，在内部和边界施加适当的物理方程约束，以无监督学习的方式对管壁压力进行建模。

<figure markdown>
  ![equation](https://paddle-org.bj.bcebos.com/paddlescience/docs/Aneurysm/aneurysm.png){ loading=lazy style="height:80%;width:80%"}
</figure>

## 2. 问题定义

假设颅内动脉瘤模型中，在入口 inlet 部分，中心点的流速为 1.5，并向四周逐渐减小；在出口 outlet 区域，压力恒为 0；在边界上无滑移，流速为 0；血管内部则符合 N-S 方程运动规律，中间段的平均流量为负（流入），出口段的平均流量为正（流出）。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 aneurysm 问题中，每一个已知的坐标点 $(x, y, z)$ 都有对应的待求解的未知量 $(u, v, w, p)$（速度和压力）
，在这里使用比较简单的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $(x, y, z)$ 到 $(u, v, w, p)$ 的映射函数 $f: \mathbb{R}^3 \to \mathbb{R}^4$ ，即：

$$
(u, v, w, p) = f(x, y, z)
$$

上式中 $f$ 即为 MLP 模型本身，用 PaddleScience 代码表示如下

``` py linenums="14"
--8<--
examples/aneurysm/aneurysm.py:14:15
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定网络模型的输入变量名是 `("x", "y", "z")`，输出变量名是 `("u", "v", "w", "p")`，这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数，就实例化出了一个拥有 6 层隐藏神经元，每层神经元数为 512 的神经网络模型 `model`，使用 `silu` 作为激活函数，并使用 `WeightNorm` 权重归一化。

### 3.2 方程构建

颅内动脉瘤模型涉及到 2 个方程，一是流体 N-S 方程，二是流量计算方程，因此使用 PaddleScience 内置的 `NavierStokes` 和 `NormalDotVec` 即可。

``` py linenums="17"
--8<--
examples/aneurysm/aneurysm.py:17:23
--8<--
```

### 3.3 计算域构建

本问题的几何区域由 stl 文件指定，按照下方命令，下载并解压到 `aneurysm/` 文件夹下。

**注：数据集中的 stl 文件和测试集数据（使用OpenFOAM生成）均来自 [Aneurysm - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)**。

``` sh
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar

# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar

# unzip it
tar -xvf aneurysm_dataset.tar
```

解压完毕之后，`aneurysm/stl` 文件夹下即存放了计算域构建所需的 stl 几何文件。

???+ warning "注意"

    **使用 `Mesh` 类之前，必须先按照[1.4.2 安装Mesh几何[可选]](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/install_setup/#142-mesh)文档，安装好 open3d、pysdf、PyMesh 3 个几何依赖包。**

然后通过 PaddleScience 内置的 STL 几何类 `Mesh` 来读取、解析这些几何文件，并且通过布尔运算，组合出各个计算域，代码如下：

``` py linenums="25"
--8<--
examples/aneurysm/aneurysm.py:25:30
--8<--
```

在此之后可以对几何域进行缩放和平移，以缩放输入数据的坐标范围，促进模型训练收敛。

``` py linenums="32"
--8<--
examples/aneurysm/aneurysm.py:32:44
--8<--
```

### 3.4 约束构建

本案例共涉及到 6 个约束，在具体约束构建之前，可以先构建数据读取配置，以便后续构建多个约束时复用该配置。

``` py linenums="46"
--8<--
examples/aneurysm/aneurysm.py:46:56
--8<--
```

#### 3.4.1 内部点约束

以作用在内部点上的 `InteriorConstraint` 为例，代码如下：

``` py linenums="103"
--8<--
examples/aneurysm/aneurysm.py:103:110
--8<--
```

`InteriorConstraint` 的第一个参数是方程（组）表达式，用于描述如何计算约束目标，此处填入在 [3.2 方程构建](#32) 章节中实例化好的 `equation["NavierStokes"].equations`；

第二个参数是约束变量的目标值，在本问题中希望与 N-S 方程相关的四个值 `continuity`, `momentum_x`, `momentum_y`, `momentum_z` 均被优化至 0；

第三个参数是约束方程作用的计算域，此处填入在 [3.3 计算域构建](#33) 章节实例化好的 `geom["interior_geo"]` 即可；

第四个参数是在计算域上的采样配置，此处设置 `batch_size` 为 `6000`。

第五个参数是损失函数，此处选用常用的 MSE 函数，且 `reduction` 设置为 `"sum"`，即会将参与计算的所有数据点产生的损失项求和；

第六个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "interior" 即可。

#### 3.4.2 边界约束

接着需要对**血管入口、出口、血管壁**这三个表面施加约束，包括入口速度约束、出口压力约束、血管壁无滑移约束。
在 `bc_inlet` 约束中，入口处的流速满足从中心点开始向周围呈二次抛物线衰减，此处使用抛物线函数表示速度随着远离圆心而衰减，再将其作为 `BoundaryConstraint` 的第二个参数(字典)的 value。

``` py linenums="62"
--8<--
examples/aneurysm/aneurysm.py:62:86
--8<--
```

血管出口、血管壁的无滑移约束构建方法类似，如下所示：

``` py linenums="87"
--8<--
examples/aneurysm/aneurysm.py:87:102
--8<--
```

#### 3.4.3 积分边界约束

对于血管入口下方的一段区域和出口区域（面），需额外施加流入和流出的流量约束，由于流量计算涉及到具体面积，因此需要使用离散积分的方式进行计算，这些过程已经内置在了 `IntegralConstraint` 这一约束条件中。如下所示：

``` py linenums="111"
--8<--
examples/aneurysm/aneurysm.py:111:138
--8<--
```

对应的流量计算公式：

$$
flow_i = \sum_{i=1}^{M}{s_{i} (\mathbf{u_i} \cdot \mathbf{n_i})}
$$

其中$M$表示离散积分点个数，$s_i$表示某一个点的（近似）面积，$\mathbf{u_i}$表示某一个点的速度矢量，$\mathbf{n_i}$表示某一个点的外法向矢量。

除前面章节所述的共同参数外，此处额外增加了 `integral_batch_size` 参数，这表示用于离散积分的采样点数量，此处使用 310 个离散点来近似积分计算；同时指定损失函数为 `IntegralLoss`，表示计算损失所用的最终预测值由多个离散点近似积分，再与标签值计算损失。

在微分方程约束、边界约束、初值约束构建完毕之后，以刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="139"
--8<--
examples/aneurysm/aneurysm.py:139:147
--8<--
```

### 3.5 超参数设定

接下来需要指定训练轮数和学习率，此处按实验经验，使用 1500 轮训练轮数，0.001 的初始学习率。

``` yaml linenums="63"
--8<--
examples/aneurysm/conf/aneurysm.yaml:63:79
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="149"
--8<--
examples/aneurysm/aneurysm.py:149:153
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.GeometryValidator` 构建评估器。

``` py linenums="155"
--8<--
examples/aneurysm/aneurysm.py:155:219
--8<--
```

### 3.8 可视化器构建

在模型评估时，如果评估结果是可以可视化的数据，可以选择合适的可视化器来对输出结果进行可视化。

本文中的输出数据是一个区域内的三维点集，因此只需要将评估的输出数据保存成 **vtu格式** 文件，最后用可视化软件打开查看即可。代码如下：

``` py linenums="206"
--8<--
examples/aneurysm/aneurysm.py:206:219
--8<--
```

### 3.9 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="221"
--8<--
examples/aneurysm/aneurysm.py:221:248
--8<--
```

## 4. 完整代码

``` py linenums="1" title="aneurysm.py"
--8<--
examples/aneurysm/aneurysm.py
--8<--
```

## 5. 结果展示

对于颅内动脉瘤测试集（共 2,962,708 个三维坐标点），模型预测结果如下所示。

<figure markdown>
  ![aneurysm_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Aneurysm/aneurysm_compare.png){ loading=lazy }
  <figcaption> 左侧为PaddleScience预测结果，中间为OpenFOAM求解器预测结果，右侧为两者的差值</figcaption>
</figure>

可以看到对于管壁压力$p(x,y,z)$，模型的预测结果和 OpenFOAM 结果基本一致。

## 6. 参考资料

- [Aneurysm - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)
