# VIV(vortex induced vibration)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6160556?contributionType=1&sUid=438690&shared=1&ts=1683961088129" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    python viv.py
    ```

=== "模型评估命令"

    ``` sh
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/viv/viv_pretrained.pdeqn
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/viv/viv_pretrained.pdparams
    python viv.py mode=eval EVAL.pretrained_model_path=./viv_pretrained
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [viv_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/viv_pretrained.pdparams)<br>[viv_pretrained.pdeqn](https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/viv_pretrained.pdeqn) | 'eta': 1.1416150300647132e-06<br>'f': 4.635014192899689e-06 |

## 1. 背景简介

涡激振动（Vortex-Induced Vibration，VIV）是一种流固耦合振动现象，主要发生在流体绕过柱体或管体时。在海洋工程和风工程中，这种振动现象具有重要应用。

在海洋工程中，涡激振动问题主要涉及海洋平台（如桩基、立管等）的涡激振动响应分析。这些平台在海流中运行，会受到涡激振动的影响。这种振动可能会导致平台结构的疲劳损伤，因此在进行海洋平台设计时，需要考虑这一问题。

在风工程中，涡激振动问题主要涉及风力发电机的涡激振动响应分析。风力发电机叶片在运行过程中受到气流的涡激振动，这种振动可能会导致叶片的疲劳损伤。为了确保风力发电机的安全运行，需要对这一问题进行深入的研究。

总之，涡激振动问题的应用主要涉及海洋工程和风工程领域，对于这些领域的发展具有重要意义。

当涡流脱落频率接近结构的固有频率时，圆柱会发生涡激振动，VIV系统相当于一个弹簧-阻尼系统：

![VIV_1D_SpringDamper](https://paddle-org.bj.bcebos.com/paddlescience/docs/ViV/VIV_1D_SpringDamper.png)

## 2. 问题定义

本问题涉及的控制方程涉及三个物理量：$λ_1$、$λ_2$ 和 $ρ$，分别表示自然阻尼、结构特性刚度和质量，控制方程定义如下所示：

$$
\rho \dfrac{\partial^2 \eta}{\partial t^2} + \lambda_1 \dfrac{\partial \eta}{\partial t} + \lambda_2 \eta = f
$$

该模型基于无量纲速度 $U_r=\dfrac{u}{f_n*d}=8.5$ 对应 $Re=500$ 的假设。我们使用通过圆柱的流体引起的圆柱振动的横向振幅 $\eta$ 和相应的升力 $f$ 作为监督数据。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 VIV 问题中，给定时间 $t$，上述系统都有横向振幅 $\eta$ 和升力 $f$ 作为待求解的未知量，并且该系统本身还包含两个参数 $\lambda_1, \lambda_2$。因此我们在这里使用比较简单的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $t$ 到 $(\eta, f)$ 的映射函数 $g: \mathbb{R}^1 \to \mathbb{R}^2$ ，即：

$$
\eta, f = g(t)
$$

上式中 $g$ 即为 MLP 模型本身，用 PaddleScience 代码表示如下

``` py linenums="22"
--8<--
examples/fsi/viv.py:22:23
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("t_f",)`，输出变量名是 `("eta",)`，
 `t_f` 代表输入时间 $t$，`eta` 代表输出振幅 $\eta$ 这些命名与后续代码保持一致。

接着通过指定 MLP 的层数、神经元个数以及激活函数，我们就实例化出了一个拥有 5 层隐藏神经元，每层神经元数为 50，使用 "tanh" 作为激活函数的神经网络模型 `model`。

### 3.2 方程构建

由于 VIV 使用的是 VIV 方程，因此可以直接使用 PaddleScience 内置的 `VIV`。

``` py linenums="25"
--8<--
examples/fsi/viv.py:25:26
--8<--
```

我们在该方程中添加了两个可学习的参数 `k1` 和 `k2` 来估计 $\lambda_1$ 和 $\lambda_2$，且它们的关系是 $\lambda_1 = e^{k1}, \lambda_2 = e^{k2}$

因此我们在实例化 `VIV` 类时需指定必要的参数：质量 `rho=2`，初始化值`k1=-4`，`k2=0`。

### 3.3 计算域构建

本文中 VIV 问题作用在 $t \in [0.0625, 9.9375]$ 中的 100 个离散时间点上，这 100 个时间点已经保存在文件 `examples/fsi/VIV_Training_Neta100.mat` 作为输入数据，因此不需要显式构建计算域。

### 3.4 约束构建

本文采用监督学习的方式，对模型输出 $\eta$ 和基于 $\eta$ 计算出的升力 $f$，这两个物理量进行约束。

在定义约束之前，需要给监督约束指定文件路径等数据读取配置。

``` py linenums="28"
--8<--
examples/fsi/viv.py:28:44
--8<--
```

#### 3.4.1 监督约束

由于我们以监督学习方式进行训练，此处采用监督约束 `SupervisedConstraint`：

``` py linenums="46"
--8<--
examples/fsi/viv.py:46:52
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，此处填入在 [3.4 方程构建](#34) 章节中实例化好的 `train_dataloader_cfg`；

第二个参数是损失函数，此处我们选用常用的MSE函数，且 `reduction` 设置为 `"mean"`，即我们会将参与计算的所有数据点产生的损失项求和取平均；

第三个参数是方程表达式，用于描述如何计算约束目标，此处填入 `eta` 的计算函数和在 [3.2 方程构建](#32) 章节中实例化好的 `equation["VIV"].equations`；

第四个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。此处我们命名为 "Sup" 即可。

在监督约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="53"
--8<--
examples/fsi/viv.py:53:54
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数和学习率，此处我们按实验经验，使用 10000 轮训练轮数，并每隔 10000 个epochs评估一次模型精度。

``` yaml linenums="41"
--8<--
examples/fsi/conf/viv.yaml:41:56
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器和 `Step` 间隔衰减学习率。

``` py linenums="56"
--8<--
examples/fsi/viv.py:56:58
--8<--
```

???+ note "说明"

    VIV 方程含有两个 **可学习参数** k1和k2，因此需要将方程与 `model` 一起传入优化器。

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="60"
--8<--
examples/fsi/viv.py:60:82
--8<--
```

评价指标 `metric` 选择 `ppsci.metric.MSE` 即可；

其余配置与 [监督约束构建](#341) 的设置类似。

### 3.8 可视化器构建

在模型评估时，如果评估结果是可以可视化的数据，我们可以选择合适的可视化器来对输出结果进行可视化。

本文需要可视化的数据是 $t-\eta$ 和 $t-f$ 两组关系图，假设每个时刻 $t$ 的坐标是 $t_i$，则对应网络输出为 $\eta_i$，升力为 $f_i$，因此我们只需要将评估过程中产生的所有 $(t_i, \eta_i, f_i)$ 保存成图片即可。代码如下：

``` py linenums="84"
--8<--
examples/fsi/viv.py:84:103
--8<--
```

### 3.9 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="105"
--8<--
examples/fsi/viv.py:105:123
--8<--
```

## 4. 完整代码

``` py linenums="1" title="viv.py"
--8<--
examples/fsi/viv.py
--8<--
```

## 5. 结果展示

模型预测结果如下所示，横轴为时间自变量$t$，$\eta_{gt}$为参考振幅，$\eta$为模型预测振幅，$f_{gt}$为参考升力，$f$为模型预测升力。

<figure markdown>
  ![Viv_result](https://paddle-org.bj.bcebos.com/paddlescience/docs/ViV/eta_f_pred.png){ loading=lazy }
  <figcaption> 振幅 eta 与升力 f 随时间t变化的预测结果和参考结果</figcaption>
</figure>

可以看到模型对在$[0,10]$时间范围内，对振幅和升力的预测结果与参考结果基本一致。
