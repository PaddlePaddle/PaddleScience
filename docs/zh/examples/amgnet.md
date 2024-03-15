# AMGNet

<!-- <a href="https://aistudio.baidu.com/aistudio/projectdetail/6184070?contributionType=1&sUid=438690&shared=1&ts=1684239806160" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

=== "模型训练命令"

    === "amgnet_airfoil"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip --output data.zip
        # unzip it
        unzip data.zip
        python amgnet_airfoil.py
        ```
    === "amgnet_cylinder"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip --output data.zip
        # unzip it
        unzip data.zip
        python amgnet_cylinder.py
        ```

=== "模型评估命令"

    === "amgnet_airfoil"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip --output data.zip
        # unzip it
        unzip data.zip
        python amgnet_airfoil.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_airfoil_pretrained.pdparams
        ```
    === "amgnet_cylinder"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip --output data.zip
        # unzip it
        unzip data.zip
        python amgnet_cylinder.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_cylinder_pretrained.pdparams
        ```

| 预训练模型  | 指标 |
|:--| :--|
| [amgnet_airfoil_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_airfoil_pretrained.pdparams) | loss(RMSE_validator): 0.0001 <br> RMSE.RMSE(RMSE_validator): 0.01315 |
| [amgnet_cylinder_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_cylinder_pretrained.pdparams) | loss(RMSE_validator): 0.00048 <br> RMSE.RMSE(RMSE_validator): 0.02197 |

## 1. 背景简介

近年来，深度学习在计算机视觉和自然语言处理方面的成功应用，促使人们探索人工智能在科学计算领域的应用，尤其是在计算流体力学(CFD)领域的应用。

流体是非常复杂的物理系统，流体的行为由 Navier-Stokes 方程控制。基于网格的有限体积或有限元模拟方法是 CFD 中广泛使用的数值方法。计算流体动力学研究的物理问题往往非常复杂，通常需要大量的计算资源才能求出问题的解，因此需要在求解精度和计算成本之间进行权衡。为了进行数值模拟，计算域通常被网格离散化，由于网格具有良好的几何和物理问题表示能力，同时和图结构相契合，所以这篇文章的作者使用图神经网络，通过训练 CFD 仿真数据，构建了一种数据驱动模型来进行流场预测。

## 2. 问题定义

作者提出了一种基于图神经网络的 CFD 计算模型，称为 AMGNET(A Multi-scale Graph neural Network)，该模型可以预测在不同物理参数下的流场。该方法有以下几个特点：

- AMGNET 把 CFD 中的网格转化为图结构，通过图神经网络进行信息的处理和聚合，相比于传统的 GCN 方法，该方法的预测误差明显更低。

- AMGNET 可以同时计算流体在 x 和 y 方向的速度，同时还能计算流体压强。

- AMGNET 通过 RS 算法(Olson and Schroder, 2018)进行了图的粗化，仅使用少量节点即可进行预测，进一步提高了预测速度。

下图为该方法的网络结构图。该模型的基本原理就是将网格结构转化为图结构，然后通过网格中节点的物理信息、位置信息以及节点类型对图中的节点和边进行编码。接着对得到的图神经网络使用基于代数多重网格算法(RS)的粗化层进行粗化，将所有节点分类为粗节点集和细节点集，其中粗节点集是细节点集的子集。粗图的节点集合就是粗节点集，于是完成了图的粗化，缩小了图的规模。粗化完成后通过设计的图神经网络信息传递块(GN)来总结和提取图的特征。之后图恢复层采用反向操作，使用空间插值法(Qi et al.,2017)对图进行上采样。例如要对节点 $i$ 插值，则在粗图中找到距离节点 $i$ 最近的 $k$ 个节点，然后通过公式计算得到节点 $i$ 的特征。最后，通过解码器得到每个节点的速度与压力信息。

![AMGNet_overview](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/amgnet.png)

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

!!! info "注意事项"

    本案例运行前需通过 `pip install pgl pyamg` 命令，安装 [**P**addle **G**raph **L**earning](https://github.com/PaddlePaddle/PGL) 图学习工具和 [PyAMG](https://github.com/pyamg/pyamg) 代数多重网格工具。

### 3.1 数据集下载

该案例使用的机翼数据集 Airfoil 来自 de Avila Belbute-Peres 等人，其中翼型数据集采用 NACA0012 翼型，包括 train, test 以及对应的网格数据 mesh_fine；圆柱数据集是原作者利用软件计算的 CFD 算例。

执行以下命令，下载并解压数据集。

``` sh
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
unzip data.zip
```

### 3.2 模型构建

在本问题中，我们使用图神经网络 `AMGNet` 作为模型，其接收图结构数据，输出预测结果。

=== "airfoil"

    ``` py linenums="61"
    --8<--
    examples/amgnet/amgnet_airfoil.py:61:62
    --8<--
    ```

=== "cylinder"

    ``` py linenums="61"
    --8<--
    examples/amgnet/amgnet_cylinder.py:61:62
    --8<--
    ```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("input", )`，输出变量名是 `("pred", )`，这些命名与后续代码保持一致。

### 3.3 约束构建

在本案例中，我们使用监督数据集对模型进行训练，因此需要构建监督约束。

在定义约束之前，我们需要指定数据集的路径等相关配置，将这些信息存放到对应的 YAML 文件中，如下所示。

=== "airfoil"

    ``` yaml linenums="21"
    --8<--
    examples/amgnet/conf/amgnet_airfoil.yaml:21:27
    --8<--
    ```

=== "cylinder"

    ``` yaml linenums="21"
    --8<--
    examples/amgnet/conf/amgnet_cylinder.yaml:21:27
    --8<--
    ```

接着定义训练损失函数的计算过程，如下所示。

=== "airfoil"

    ``` py linenums="35"
    --8<--
    examples/amgnet/amgnet_airfoil.py:35:40
    --8<--
    ```

=== "cylinder"

    ``` py linenums="35"
    --8<--
    examples/amgnet/amgnet_cylinder.py:35:40
    --8<--
    ```

最后构建监督约束，如下所示。

=== "airfoil"

    ``` py linenums="82"
    --8<--
    examples/amgnet/amgnet_airfoil.py:82:90
    --8<--
    ```

=== "cylinder"

    ``` py linenums="82"
    --8<--
    examples/amgnet/amgnet_cylinder.py:82:90
    --8<--
    ```

### 3.4 超参数设定

设置训练轮数等参数，如下所示。

=== "airfoil"

    ``` yaml linenums="41"
    --8<--
    examples/amgnet/conf/amgnet_airfoil.yaml:41:51
    --8<--
    ```

=== "cylinder"

    ``` yaml linenums="41"
    --8<--
    examples/amgnet/conf/amgnet_cylinder.yaml:41:51
    --8<--
    ```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并使用固定的 `5e-4` 作为学习率。

=== "airfoil"

    ``` py linenums="92"
    --8<--
    examples/amgnet/amgnet_airfoil.py:92:93
    --8<--
    ```

=== "cylinder"

    ``` py linenums="92"
    --8<--
    examples/amgnet/amgnet_cylinder.py:92:93
    --8<--
    ```

### 3.6 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集(测试集)评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器，构建过程与 [约束构建](#33) 类似，只需把数据目录改为测试集的目录，并在配置文件中设置 `EVAL.batch_size=1` 即可。

=== "airfoil"

    ``` py linenums="95"
    --8<--
    examples/amgnet/amgnet_airfoil.py:95:118
    --8<--
    ```
=== "cylinder"

    ``` py linenums="95"
    --8<--
    examples/amgnet/amgnet_cylinder.py:95:118
    --8<--
    ```

评估指标为预测结果和真实结果的 RMSE 值，因此需自定义指标计算函数，如下所示。

=== "airfoil"

    ``` py linenums="43"
    --8<--
    examples/amgnet/amgnet_airfoil.py:43:52
    --8<--
    ```
=== "cylinder"

    ``` py linenums="43"
    --8<--
    examples/amgnet/amgnet_cylinder.py:43:52
    --8<--
    ```

### 3.7 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

=== "airfoil"

    ``` py linenums="120"
    --8<--
    examples/amgnet/amgnet_airfoil.py:120:136
    --8<--
    ```
=== "cylinder"

    ``` py linenums="120"
    --8<--
    examples/amgnet/amgnet_cylinder.py:120:136
    --8<--
    ```

### 3.8 结果可视化

训练完毕之后程序会对测试集中的数据进行预测，并以图片的形式对结果进行可视化，如下所示。

=== "airfoil"

    ``` py linenums="138"
    --8<--
    examples/amgnet/amgnet_airfoil.py:138:
    --8<--
    ```
=== "cylinder"

    ``` py linenums="138"
    --8<--
    examples/amgnet/amgnet_cylinder.py:138:
    --8<--
    ```

## 4. 完整代码

=== "airfoil"

    ``` py linenums="1" title="amgnet_airfoil.py"
    --8<--
    examples/amgnet/amgnet_airfoil.py
    --8<--
    ```
=== "cylinder"

    ``` py linenums="1" title="amgnet_airfoil.py"
    --8<--
    examples/amgnet/amgnet_cylinder.py
    --8<--
    ```

## 5. 结果展示

下方展示了模型对计算域中每个点的压力$p(x,y)$、x(水平)方向流速$u(x,y)$、y(垂直)方向流速$v(x,y)$的预测结果与参考结果。

=== "airfoil"

    <figure markdown>
        ![Airfoil_0_vec_x](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/airfoil_0field.png0_field.png){ loading=lazy }
        <figcaption>左：预测 x 方向流速 p，右：实际 x 方向流速</figcaption>
        ![Airfoil_0_p](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/airfoil_0field.png1_field.png){ loading=lazy }
        <figcaption>左：预测压力 p，右：实际压力 p</figcaption>
        ![Airfoil_0_vec_y](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/airfoil_0field.png2_field.png){ loading=lazy }
        <figcaption>左：预测y方向流速 p，右：实际 y 方向流速</figcaption>
    </figure>

=== "cylinder"

    <figure markdown>
        ![Cylinder_0_vec_x](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/cylinder_0field.png0_field.png){ loading=lazy }
        <figcaption>左：预测 x 方向流速 p，右：实际 x 方向流速</figcaption>
        ![Cylinder_0_p](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/cylinder_0field.png1_field.png){ loading=lazy }
        <figcaption>左：预测压力 p，右：实际压力 p</figcaption>
        ![Cylinder_0_vec_y](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/cylinder_0field.png2_field.png){ loading=lazy }
        <figcaption>左：预测 y 方向流速 p，右：实际 y 方向流速</figcaption>
    </figure>

可以看到模型预测结果与真实结果基本一致。

## 6. 参考文献

- [AMGNET: multi-scale graph neural networks for flow field prediction](https://doi.org/10.1080/09540091.2022.2131737)
- [AMGNet - Github](https://github.com/baoshiaijhin/amgnet)
- [AMGNet - AIStudio](https://aistudio.baidu.com/projectdetail/5592458)
