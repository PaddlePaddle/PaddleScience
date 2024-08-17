# CVit(Navier-Stokes)

<a href="https://aistudio.baidu.com/projectdetail/8141482" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # download data
    git lfs install
    git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D
    python ns_cvit.py
    ```

=== "模型评估命令"

    ``` sh
    # download data
    git lfs install
    git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D
    python ns_cvit.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/ns_cvit_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python ns_cvit.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # download data
    git lfs install
    git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D
    python ns_cvit.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [ns_cvit_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/ns_cvit_pretrained.pdparams) | 4-step l2_error: 0.0398 |

## 1. 背景简介

sciml 领域所使用的模型现阶段与CV、NLP领域的先进模型有较大差别，并没有很好地利用好这些先进模型所提供的优势。因此论文作者首先提出了一个算子学习的统一视角，按照 Global conditioning 和 Local Conditioning 分别对 DeepONet、FNO、GNO 等模型进行了归纳与总结，然后基于目前广泛应用于CV、NLP领域的 Transformer 结构设计了一种 Global conditioning 的模型 CVit。相比以往的算子学习模型，参数量更小，精度更高。

模型结构如下图所示：

<img src="https://github.com/PredictiveIntelligenceLab/cvit/raw/main/figures/cvit_arch.png" alt="Cvit" width="800">

## 2. 问题定义

CVit 作为一种算子学习模型，以输入函数 $u$、函数 $s$ 的查询点 query coordinate $y$ 为输入，输出经过算子映射后的函数，在查询点 $y$ 处的函数值 $s(y)$。

本问题基于固定方腔的不可压 buoyancy-driven flow 即方腔内的浮力驱动流动问题，求解如下方程：

Formulation We consider the vorticity-stream $(\omega-\psi)$ formulation of the incompressible Navier-Stokes equations on a two-dimensional periodic domain, $D=D_u=D_v=[0,2 \pi]^2$ :

$$
\begin{aligned}
& \frac{\partial \omega}{\partial t}+(v \cdot \nabla) \omega-v \Delta \omega=f^{\prime} \\
& \omega=-\Delta \psi \quad \int_D \psi=0, \\
& v=\left(\frac{\partial \psi}{\partial x_2},-\frac{\partial \psi}{\partial x_1}\right)
\end{aligned}
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在本问题中，对于每一个函数 $u$，其经过算子学习模型映射到 $s$ 后，在 $y$ 上都有对应的标签 $s(y)$，因此在这里使用 CVit 来表示 $(u, y)$ 到 $s(y)$ 的映射关系：

$$
s(y) = G(u)(y)
$$

上式中 $G(u)$ 即为 CVit 模型本身，用 PaddleScience 代码表示如下

``` py linenums="80"
--8<--
examples/ns/ns_cvit.py:80:81
--8<--
```

为了在计算时，准确快速地访问具体变量的值，在这里指定网络模型的输入变量名是 `("u", "y")`，输出变量名是 `("s")`，这些命名与后续代码保持一致。

接着通过指定 CVit 的输入维度、坐标维度、输出维度、模型层数等超参数，就可实例化出了一个 `model`

``` yaml linenums="40"
--8<--
examples/ns/conf/ns_cvit_small_8x8.yaml:40:59
--8<--
```

### 3.2 数据准备

本问题中的数据分片储存在 `NavierStokes2D/*.h5` 文件中，分为训练和测试集，其数据包含内容如下表所示（这些信息会在运行时打印出来）。

| 文件名 | 文件数量 | 数据形状 | 输入形状 | 标签形状 |
| :-- | :-- | :-- | :-- | :-- |
| NavierStokes2D_train_*.h5 | 52 |`[1000, 14, 128, 128, 3]` | `[4000, 10, 128, 128, 3]` | `[4000, 1, 128, 128, 3]` |
| NavierStokes2D_test_*.h5 | 41 | `[5200, 14, 128, 128, 3]` | `[20800, 10, 128, 128, 3]` | `[20800, 1, 128, 128, 3]` |

数据读取函数如下：

``` py linenums="27"
--8<--
examples/ns/ns_cvit.py:27:76
--8<--
```

训练、测试时采用前 10 个时刻预测下一个时刻，并且测试时会以自回归的形式连续预测 4 个时刻。

### 3.3 约束构建

#### 3.3.1 监督约束

在训练时，随机选取 `batch_size` 组来自 $u$ 上的数据、并同时随机选取 `query_point` 个 $y$ 坐标，如此构成了训练输入数据，标签数据则从 $s$ 中随机选取同样的 `batch_size` x `query_point` 个标签点。

``` py linenums="104"
--8<--
examples/ns/ns_cvit.py:104:145
--8<--
```

`SupervisedConstraint` 的第一个参数是用于训练的数据配置，我们使用 `NamedArrayDataset` 作为数据集类型，并且传入自定义的`random_query`作为`transforms`，完成上述的样本随机选取过程；

第二个参数是该约束的计算表达式，我们只需要计算 $s$ 即可，因此填入一个不做任何处理，直接取出模型输出结果"s"的匿名表达式；

第三个参数是损失函数，此处选用 `MSELoss` 函数；

第四个参数是约束条件的名字，需要给每一个约束条件命名，方便后续对其索引。此处命名为 "Sup" 即可。

### 3.4 超参数设定

接下来需要指定训练轮数和学习率，此处按实验经验，使用 200 轮训练轮数，初始学习率为 0.001，预热轮数为 5，全局梯度裁剪系数为 1.0，权重衰减为 1e-5。

``` yaml linenums="61"
--8<--
examples/ns/conf/ns_cvit_small_8x8.yaml:61:79
--8<--
```

### 3.5 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` py linenums="147"
--8<--
examples/ns/ns_cvit.py:147:155
--8<--
```

### 3.6 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="157"
--8<--
examples/ns/ns_cvit.py:157:202
--8<--
```

过程中我们使用了自定义的评估函数 `l2_err_func`，用于评估测试集上所有样本、三个输出物理量的 2-范数误差。

### 3.7 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="204"
--8<--
examples/ns/ns_cvit.py:204:213
--8<--
```

## 4. 完整代码

``` py linenums="1" title="ns_cvit.py"
--8<--
examples/ns/ns_cvit.py
--8<--
```

## 5. 结果展示

在测试集上的预测结果、参考结果以及绝对值误差如下图所示。

<figure markdown>
  ![ns_u.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/ns_u.png){ loading=lazy }
  <figcaption> 左侧为 CVit 对物理量 u 的预测结果，中间为物理量 u 的参考结果，右侧为两者的差值</figcaption>
  ![ns_ux.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/ns_ux.png){ loading=lazy }
  <figcaption> 左侧为 CVit 对物理量 ux 的预测结果，中间为物理量 ux 的参考结果，右侧为两者的差值</figcaption>
  ![ns_uy.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/ns_uy.png){ loading=lazy }
  <figcaption> 左侧为 CVit 对物理量 uy 的预测结果，中间为物理量 uy 的参考结果，右侧为两者的差值</figcaption>
</figure>

可以看到模型的三个预测物理量和参考结果基本一致，通过自回归的方式，连续推理 4 步的平均误差为 0.039%。

## 6. 参考资料

- [Bridging Operator Learning and Conditioned Neural Fields: A Unifying Perspective](https://arxiv.org/abs/2405.13998)
- [PredictiveIntelligenceLab/cvit/ns](https://github.com/PredictiveIntelligenceLab/cvit/blob/main/ns/README.md)
- [The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks](https://arxiv.org/abs/2203.13181)
