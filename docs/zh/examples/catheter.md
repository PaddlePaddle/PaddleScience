# Catheter

`<a href="https://aistudio.baidu.com/projectdetail/8252779?sUid=1952564&shared=1&ts=1727243697832" class="md-button md-button--primary" style>`AI Studio 快速体验 `</a>`

=== "模型训练命令"

    ``` sh
    python catheter.py
    ```

=== "模型评估命令"

    ``` sh
    python catheter.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/GeoFNO/GeoFNO_pretrained.pdparams
    ```

| 预训练模型                                                                                              |
| :------------------------------------------------------------------------------------------------------ |
| [GeoFno.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/GeoFNO/GeoFNO_pretrained.pdparams) |

## 1. 背景简介

人工智能辅助的抗感染导管几何创新设计

在狭窄管道内的流体环境中，细菌能借助流体动力学作用逆流迁移，对使用植入性导管的患者构成泌尿道感染的严重威胁。尽管已有提议采用涂层与结构化表面来抑制导管内的细菌滋生，但遗憾的是，至今尚无一种表面结构或涂层技术能从根本上解决污染难题。鉴于此，我们依据逆流游动的物理原理，创新性地提出了一种几何设计方案，并通过人工智能模型对细菌流入动力学进行预测与优化。相较于传统模拟方法，所采用的傅立叶神经算子人工智能技术实现了显著的速度提升。

在准二维微流体实验中，我们以大肠杆菌为对象，验证了该设计的抗感染机制，并在临床相关流速下，通过 3D 打印的导管原型对其有效性进行了评估。实验结果显示，我们的导管设计在抑制导管上游端细菌污染方面，实现了 1-2 个数量级的提升，有望大幅延长导管的安全留置时间，并整体降低导管相关性尿路感染的风险。

本案例通过深度学习的方式对[论文](https://www.science.org/doi/pdf/10.1126/sciadv.adj1741)所提及的模型进行复现。

## 2. 问题定义

基于细菌上游游泳的物理机制，建立相应的数学模型,通常使用 ABP 模型进行表示：

$$
\frac{d\vec{q}}{dt} = \frac{1}{2} \vec{\omega} + \frac{2}{\tau_R} \eta(t) \times \vec{q}
$$

该模型考虑了细菌与导管壁之间的流体动力学相互作用，以及细菌的形状、大小和表面性质等因素。
其中

- `dt(q)` 代表细菌方向变化率
- `ω` 代表流体涡量
- `η(t)` 代表高斯噪声
- `q` 代表细菌方向向量

## 3. 问题求解

论文采用几何聚焦傅里叶神经算子（Geo-FNO）构建AI模型。该模型能够学习并解决与几何形状相关的偏微分方程（SPDE），从而实现对导管几何形状的优化，并通过微流体实验和3D打印技术，制作具有不同几何形状的导管原型，并测试其抑制细菌上游游泳的效果。
接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。为了快速理解 PaddleScience，接下来仅对模型构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API 文档](../api/arch.md)。

### 3.1 数据集介绍

本案例数据集使用论文作者所提供的数据集，共 8 个 npy 文件，[下载地址](https://aistudio.baidu.com/datasetdetail/291940)

数据文件说明如下：

`./data.zip/training/`

|          字段名          |        说明        |
| :----------------------: | :----------------: |
| x_1d_structured_mesh.npy | 形状为(2001, 3003) |
| y_1d_structured_mesh.npy | 形状为(2001, 3003) |
|      data_info.npy      |  形状为(7, 3003)  |
|   density_1d_data.npy   | 形状为(2001, 3003) |

`./data.zip/test/`

|          字段名          |       说明       |
| :----------------------: | :---------------: |
| x_1d_structured_mesh.npy | 形状为(2001, 300) |
| y_1d_structured_mesh.npy | 形状为(2001, 300) |
|      data_info.npy      |  形状为(7, 300)  |
|   density_1d_data.npy   | 形状为(2001, 300) |

在加载数据之后，需要将 x、y 进行合并，同时对于合并后的训练数据重新 `reshape` 为 `(1000, 2001, 2)` 的格式，具体代码如下

```py
--8<--
examples/catheter/catheter.py:29:61
--8<--
```

### 3.2 GeoFNO 模型

GeoFNO 是一种基于 **几何聚焦傅里叶神经算子 (Geo-FNO** ) 的机器学习模型，它将几何形状转换到傅里叶空间，从而更好地捕捉形状的特征，并利用傅里叶变换的可逆性，可以将结果转换回物理空间。

在论文中，该模型能够学习并解决与几何形状相关的偏微分方程（SPDE），从而实现对导管几何形状的优化，并根据论文所提供 pytorch 模型代码，现用 PaddleScience 代码表示如下

```py
--8<--
ppsci/arch/geofno.py:64:146
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("input",)`，输出变量名是 `("output",)`，这些命名与后续代码保持一致。

接着通过指定 FNO1d 的层数、特征通道数，神经元个数，并通过加载上文所提及的初始化权重模型，我们就实例化出了一个神经网络模型 `model`。

### 3.3 约束构建

#### 3.3.1 监督约束

由于我们以监督学习方式进行训练，此处采用监督约束 `SupervisedConstraint`：

```py
--8<--
examples/catheter/catheter.py:76:92
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，里面包好数据的 `label`，已便后续进行索引使用。同时定义了数据的 batch_size 以及其他相关配置；

第二个参数是损失函数，此处我们选用论文代码仓中的 `LPLoss`损失函数，其定义可看 [Loss 构建](#36)；

第三个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

### 3.4 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，同时使用 `PaddleScience` 中的 `MultiStepDecay` 生成动态学习率。

```py
--8<--
examples/catheter/catheter.py:100:102
--8<--
```

### 3.5 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

```py
--8<--
examples/catheter/catheter.py:110:127
--8<--
```

评价指标 `metric` 选择 `ppsci.metric.L2Rel` 即可。

其余配置与 [约束构建](#33) 的设置类似。

### 3.6 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

```py
--8<--
examples/catheter/catheter.py:130:141
--8<--
```

## 4. 完整代码

=== "catheter.py"

```py
--8<--
examples/catheter/catheter.py
--8<--
```

## 5. 结果展示

下方展示了训练后模型对测试数据的第一次预测结果以及最后一次预测结果。

=== "第一次预测结果"

![1725427977357](https://paddle-org.bj.bcebos.com/paddlescience/docs/catheter/1725427977357.png)

=== "最后一次预测结果"

![1725428017615](https://paddle-org.bj.bcebos.com/paddlescience/docs/catheter/1725428017615.png)

=== "训练测试损失"

![1725894134717](https://paddle-org.bj.bcebos.com/paddlescience/docs/catheter/1725894134717.png)

可以看到模型预测结果与真实结果基本一致，优化后的导管具有特定的几何形状，如障碍物分布和间距等，这些形状特征能够显著影响流体动力学相互作用，从而抑制细菌的上游游泳行为。

## 6. 参考资料

参考文献： [AI-aided geometric design of anti-infection catheters](https://www.science.org/doi/pdf/10.1126/sciadv.adj1741)

参考代码： [Geo-FNO-catheter](https://github.com/zongyi-li/Geo-FNO-catheter)
