# AI-aided geometric design of anti-infection catheters(人工智能辅助的抗感染导管几何设计)

## 论文信息:
|年份 | 期刊 | 作者|引用数 | 论文PDF | 
|-----|-----|-----|---|-----|
|2024|Science Advance|Tingtao Zhou, X Wan, DZ Huang, Zongyi Li, Z Peng, A Anandkumar, JF Brady, PW Sternberg, C Daraio|15|[AI-aided geometric design of anti-infection catheters](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters.pdf)|

## 代码信息
|问题类型 | 在线运行 |神经网络|预训练模型|指标|
|---------|-----|---------|-|-|
|算子神经网络预测流场|[人工智能辅助的抗感染导管几何设计](https://aistudio.baidu.com/projectdetail/8252779?sUid=1952564&shared=1&ts=172724369783)|傅立叶几何神经算子|[GeoFNO_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/GeoFNO/GeoFNO_pretrained.pdparams)|loss(MAE): 0.4195|


=== "模型训练命令"

    ``` sh
    python catheter.py
    ```

=== "模型评估命令"

    ``` sh
    python catheter.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/GeoFNO/GeoFNO_pretrained.pdparams
    ```

## 1. 背景简介
在狭窄管道内的流体环境中，细菌能借助流体动力学作用逆流迁移，对使用植入性导管的患者构成泌尿道感染的严重威胁。尽管已有提议采用涂层与结构化表面来抑制导管内的细菌滋生，但遗憾的是，至今尚无一种表面结构或涂层技术能从根本上解决污染难题。鉴于此，我们依据逆流游动的物理原理，创新性地提出了一种几何设计方案，并通过AI模型对细菌流入动力学进行预测与优化。相较于传统模拟方法，所采用的傅立叶神经算子人工智能技术实现了显著的速度提升。

在准二维微流体实验中，我们以大肠杆菌为对象，验证了该设计的抗感染机制，并在临床相关流速下，通过 3D 打印的导管原型对其有效性进行了评估。实验结果显示，我们的导管设计在抑制导管上游端细菌污染方面，实现了 1-2 个数量级的提升，有望大幅延长导管的安全留置时间，并整体降低导管相关性尿路感染的风险。

## 2. 问题定义

![图1. 提出的导管相关尿路感染（CAUTI）机制与抗感染设计流程示意图](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter.png)
**图1. 提出的导管相关尿路感染（CAUTI）机制与抗感染设计流程示意图**
 
- **（A）提出的CAUTI机制**：尿液从患者膀胱内通过导管向外流出时，细菌能够逆着尿流方向（即上游）游动，进而可能侵入患者体内并引发感染。
- **（B）细菌的跑动-翻滚运动与上游游动机制**：细菌通过一种特有的跑动-翻滚运动模式，在液体环境中实现上游游动。
- **（C）模拟探索导管形状**：利用模拟技术，探索不同导管形状对细菌上游游动的影响，以期找到能够抑制细菌上游游动的导管设计。
- **（D）人工智能辅助优化**：采用Geo-FNO框架进行人工智能辅助优化，进一步细化导管的设计参数，提升其对细菌上游游动的抑制效果。
- **（E）二维通道微流控实验**：在二维微流控通道中，对优化后的导管设计进行实验验证，评估其在实际流体环境中的抗感染性能。
- **（F）三维实验验证**：使用设计的实际尺寸导管进行三维实验，进一步验证其在临床使用条件下的抗感染效果。

我们致力于设计能够防止细菌向上游移动并最大程度减少污染的导管。为了优化导管的几何形状，我们将设计空间限定为在导管内壁布置三角形障碍物。我们捕捉了自驱动球体所展现的最简单的上游游动物理机制（27），并进行了流体和粒子动力学模拟，以找出几何设计原则（图1C）。通过将流体力学和几何整流效应耦合为随机偏微分方程（SPDE），我们对细菌分布进行了建模。随后，我们使用模拟数据训练了一个基于几何聚焦傅里叶神经算子（Geo-FNO）（62, 63）的人工智能（AI）模型，以学习SPDE的解，并利用训练好的模型来优化导管的几何形状（图1D）。基于优化后的设计，我们制作了准二维（2D）微流控装置（图1E）和3D打印的原型导管（图1F），以评估我们的设计理念的有效性。实验结果表明，与标准导管相比，我们的设计在抑制细菌超标污染方面提高了多达两个数量级，为导管相关尿路感染（CAUTI）的管理提供了一条新途径。

我们采用了一个简单的模型（27）来描述剪切流中细菌的动力学行为。在这个模型中，细菌被近似为可忽略大小的球体，其方向$q$由以下方程得出：
基于细菌上游游泳的物理机制，建立相应的数学模型,通常使用 ABP 模型进行表示：

$$
\frac{d\vec{q}}{dt} = \frac{1}{2} \vec{\omega} + \frac{2}{\tau_R} \eta(t) \times \vec{q}
$$

该模型考虑了细菌与导管壁之间的流体动力学相互作用，以及细菌的形状、大小和表面性质等因素。
其中

- $dt(q)$ 代表细菌方向变化率
- $ω$ 代表局部流体涡量
- $η(t)$ 代表高斯噪声, 满足$<η(t)>=0$ 和 $<η(0)η(t)>=\delta(t)I$
- $q$ 代表细菌方向向量
- $\tau_R$ 代表平均运行时间(更多细节详见补充材料)

我们首先通过数值模拟研究了传统表面改性方法，如抗菌纳米粒子涂层（36, 42）、工程化粗糙度或疏水性处理（65, 66），在抑制细菌上游游动中的作用。这些改性表面能够防止细菌过于接近壁面。为了模拟这些表面的存在，我们假设它们会导致细菌从表面脱离，并至少保持在距离表面3微米的位置，这个距离超过了典型的E.coli大肠杆菌体长（1至2微米）。虽然表面改性也可能影响细菌与壁面之间的流体力学相互作用，但在我们基于点状球体的简单通用模型中忽略了这一点。

我们发现，在所测试的流速范围内，表面排斥对细菌的上游游动行为几乎没有影响。通过比较光滑通道内（图2D）和表面改性通道内（图2E）持续游动细菌的模拟轨迹，我们发现它们的上游游动行为相似。

我们采用两个群体统计指标来量化抑制细菌上游游动的有效性：
- （i）平均上游游动距离$x_{up}=-\int_{0}^{-\infty}\rho(x)xdx$，通过计算细菌分布函数$ρ(x)$的加权平均值得出，其中$x$为细菌位置；
- （ii）前$1\%$上游游动最远的细菌所能到达的距离$x_{1\%}$。模拟结果显示，表面改性仅在中等流速下略微减少了$x_{up}$，但对x1%几乎没有影响（图2F中的蓝线和粉线）。这种表面改性效果不佳的结果与近几年一些论文的实验观察结果一致（39, 40）。

随后，我们通过添加物理障碍物来探索导管表面几何形状的作用。我们发现，对称和不对称的障碍物都能显著抑制细菌的上游游动（如图2F中的黑色和绿色线条所示）。我们确定了两种协同效应：首先，障碍物的斜率会在细菌从障碍物顶部出发时改变其游动方向，从而打断了它们沿着管壁表面的连续攀爬。不对称的形状会使细菌的运动偏向下游（如图2A所示），这在模拟的0流速下的轨迹（补充材料和图S1）以及低流速下上游游动统计数据的差异（图2F中的黑色和绿色线条）中均有所体现。其次，在有限的流速下，流场与光滑通道中的泊肃叶流不同（如图2B所示）。在泊肃叶流中，涡量会使细菌转向下游。而在障碍物附近，涡量会增强，导致细菌转向上游（如图2C和补充材料图S2所示）, 从而加强了细菌的转向机制。结合这两种效应，我们预计在具有优化障碍物几何形状的通道中，细菌的上游游动将显著减少。

设计优化的参数空间由四个参数表征：障碍物基底长度$L$、高度$h$、尖端位置$s$以及障碍物间距$d$；我们用W表示通道宽度（图2G）。为了优化这个空间，我们设定了两个约束条件。首先，如果相邻障碍物过于接近，它们尖端的涡旋就会开始重叠。由于这种重叠，最大有效涡旋强度（正好在障碍物尖端；有效涡旋的数学定义见补充材料）和涡旋的有效尺寸都会减小。此外，还会形成更大的边界层和滞流区（图S2，A和B）。因此，我们将障碍物间距约束为$d > 0.5W$（图S2G）。其次，在其他参数固定的情况下，随着h的增加，障碍物尖端的有效涡旋强度也会增加（图S2，C至H），这有利于促进涡旋重定向效应。然而，当$h = W/2$时，管道显然会发生堵塞。这种随着h增加而堵塞加剧的趋势反映在为了保持相同的有效流速而所需压力降的持续增加上（图S2I）。为了避免堵塞，我们将高度约束为$h < 0.3W$。

![图2. 障碍物抑制上游游动和几何优化的物理机制](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter2.png)

**图2. 障碍物抑制上游游动和几何优化的物理机制**
 
- **（A）无流动时的几何整流效应**：描述了在没有流体流动的情况下，几何形状对细菌游动方向的影响。
 
- **（B）光滑通道中的泊肃叶流**：彩色背景显示流涡量的相对大小，颜色越深表示涡量越大。在光滑通道中，泊肃叶流产生的涡量使细菌头部向下游旋转。
 
- **（C）带有对称障碍物的通道中的流动**：在带有对称障碍物的通道中，障碍物顶部附近的流速和涡量增强，这导致更强的扭矩作用在细菌上，使其重定向至下游。
 
- **（D）和（E）不同条件下的细菌模拟轨迹**：
  - **（D）光滑通道**：在宽度为50微米的二维光滑通道中，细菌的模拟轨迹显示其持续游动状态。
  - **（E）排斥细菌的表面改性通道**：在表面经过改性以排斥细菌的通道中，细菌的游动轨迹受到显著影响。
 
- **（F）上游游动的群体统计**：
  - 实线（左侧y轴）表示平均上游距离，反映了细菌群体在上游方向上的平均游动距离。
  - 虚线（右侧y轴）表示群体中前1%游动者的上游距离，揭示了少数高效游动细菌的表现。
  - 不同颜色的线条代表不同的通道条件：蓝色为光滑通道，橙色为表面改性通道，黑色为对称障碍物通道，绿色为不对称障碍物通道。
 
- **（G）AI算子神经网络模型和结果**：
  - Geo-FnO模型旨在学习导管几何形状与细菌分布之间的关系，通过一系列神经算子层实现。
  - 模型首先将不规则的通道几何形状映射到单位段[0,1]，然后在潜在空间中应用基于傅里叶的内核进行预测。
  - 最后，将预测的细菌分布从潜在空间转换回物理空间。
  - 右图展示了随机初始条件（黑色）和优化后的设计（粉色）的对比，以及通过流体和粒子动力学模拟验证的Geo-FnO预测结果（绿色虚线）。

![图3. 微流控实验](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter3.png)
**图3. 微流控实验**
 
- **（A）微流控实验示意图**：微流控通道的一端连接着装有成像溶液的注射器，另一端则连接着装有大肠杆菌的储液池。长箭头表示流动方向。
 
- **（B）细菌在锐角处的积聚**：由于流动停滞，细菌在通道的锐角处积聚。
 
- **（C）微流控通道的明场图像**：展示了通道的实际结构。
 
- **（D）细菌从通道壁上脱落的典型事件**：
  - 细菌（白色点）的轨迹在过去5秒内以黄色线条显示。
  - 上图展示了一种类型1的轨迹，其中细菌从障碍物尖端脱落。
  - 下图展示了一种典型的类型2轨迹，其中细菌从通道的平滑部分脱落。
  - 左列为实验图像，右列为模拟图像。
 
- **（E）脱落事件的统计**：提供了关于细菌脱落事件的统计数据。

![图4. 3D打印导管原型的实验](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter3.png)
**Fig. 4. 3D打印导管原型的实验**

- **（A）实验设置**：导管的下游端连接到大肠杆菌的储液池，上游端连接到由注射泵控制的装满培养液的注射器。1小时后，将导管切成等长段，并提取内部液体进行24小时培养。在显微镜下计数大肠杆菌菌落的数量，以反映每段导管中的细菌数量。

- **（B）光滑导管中的大肠杆菌超污染**：展示了在光滑导管中大肠杆菌的污染情况。

- **（C）设计导管与光滑导管的比较**：对比了设计导管与光滑导管在细菌污染方面的差异。插图显示了相同数据在对数尺度上的绘制。

## 3. 问题求解

论文采用几何聚焦傅里叶神经算子（Geo-FNO）构建AI模型。该模型能够学习并解决与几何形状相关的偏微分方程（SPDE），从而实现对导管几何形状的优化，并通过微流体实验和3D打印技术，制作具有不同几何形状的导管原型，并测试其抑制细菌上游游泳的效果。
接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。为了快速理解 PaddleScience，接下来仅对模型构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API 文档](../api/arch.md)。

### 3.1 数据集介绍

本案例数据集使用论文作者所提供的数据集，共 8 个 npy 文件，[下载地址](https://aistudio.baidu.com/datasetdetail/291940)

数据文件说明如下：

|`./data.zip/training/`||`./data.zip/test/`||
| :----------------------: | :----------------: | :----------------------: | :---------------: |
|          文件名          |        说明        |          文件名          |       说明       |
| training/x_1d_structured_mesh.npy | 形状为(2001, 3003) | test/x_1d_structured_mesh.npy | 形状为(2001, 300) |
| training/y_1d_structured_mesh.npy | 形状为(2001, 3003) | test/y_1d_structured_mesh.npy | 形状为(2001, 300) |
|      training/data_info.npy      |  形状为(7, 3003)  |      test/data_info.npy      |  形状为(7, 300)  |
|   training/density_1d_data.npy   | 形状为(2001, 3003) |   test/density_1d_data.npy   | 形状为(2001, 300) |

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

## 6. 参考

参考代码： /zongyi-li/Geo-FNO-catheter

参考文献列表

1. J. W. Warren, the catheter and urinary tract infection. Med. Clin. North Am. 75, 481–493
(1991).

2. l. e. nicolle, catheter- related urinary tract infection. Drugs Aging 22, 627–639 (2005).

3. e. K. Shuman, c. e. chenoweth, Urinary catheter- associated infections. Infect. Dis. Clin.
North Am. 32, 885–897 (2018).
4. n. Buetti, A. tabah, J. F. timsit, W. Zingg, What is new in catheter use and catheter
infection prevention in the icU. Curr. Opin. Crit. Care 26, 459–465 (2020).
5. l. chuang, P. A. tambyah, catheter-associated urinary tract infection. J. Infect. Chemother.
27, 1400–1406 (2021).
6. e. Zimlichman, d. henderson, O. tamir, c. Franz, P. Song, c. K. Yamin, c. Keohane,
c. R. denham, d. W. Bates, health care–associated infections. JAMA Intern. Med. 173,
2039–2046 (2013).
7. U. Samuel, J. Guggenbichler, Prevention of catheter-related infections: the potential of a
new nano-silver impregnated catheter. Int. J. Antimicrob. Agents 23, 75–78 (2004).
8. W. Kohnen, B. Jansen, Polymer materials for the prevention of catheter-related infections.
Zentralbl. Bakteriol. 283, 175–186 (1995).
9. A. hameed, F. chinegwundoh, A. thwaini, Prevention of catheter-related urinary tract
infections. Med. Hypotheses 71, 148–152 (2010).
10. h. c. Berg, d. A. Brown, chemotaxis in Escherichia coli analysed by three-dimensional
tracking. Nature 239, 500–504 (1972).
11. h. c. Berg, the rotary motor of bacterial flagella. Annu. Rev. Biochem. 72, 19–54 (2003).
12. h. c. Berg, E. coli in Motion (Springer, 2004).
13. M. Polin, i. tuval, K. drescher, J. P. Gollub, R. e. Goldstein, chlamydomonas swims with two
“gears” in a eukaryotic version of run-and- tumble locomotion. Science 325, 487–490
(2009).
14. A. P. Berke, l. turner, h. c. Berg, e. lauga, hydrodynamic attraction of swimming
microorganisms by surfaces. Phys. Rev. Lett. 101, 038102 (2008).
15. e. lauga, t. R. Powers, the hydrodynamics of swimming microorganisms. Rep. Prog. Phys.
72, 096601 (2009).
16. d. Kaiser, Bacterial swarming: A re-examination of cell-movement patterns. Curr. Biol. 17,
R561–R570 (2007).
17. n. verstraeten, K. Braeken, B. debkumari, M. Fauvart, J. Fransaer, J. vermant, J. Michiels,
living on a surface: Swarming and biofilm formation. Trends Microbiol. 16, 496–506
(2008).
18. d. B. Kearns, A field guide to bacterial swarming motility. Nat. Rev. Microbiol. 8, 634–644
(2010).
19. d. Ghosh, X. cheng, to cross or not to cross: collective swimming of Escherichia coli under
two-dimensional confinement. Phys. Rev. Res. 4, 023105 (2022).
20. J. hill, O. Kalkanci, J. l. McMurry, h. Koser, hydrodynamic surface interactions enable
Escherichia coli to seek efficient routes to swim upstream. Phys. Rev. Lett. 98, 068101 (2007).
21. t. Kaya, h. Koser, direct upstream motility in Escherichia coli. Biophys. J. 102, 1514–1523
(2012).
22. Marcos, h. c. Fu, t. R. Powers, R. Stocker, Bacterial rheotaxis. Proc. Natl. Acad. Sci. U.S.A.
109, 4780–4785 (2012).
23. Y. Shen, A. Siryaporn, S. lecuyer, Z. Gitai, h. A. Stone, Flow directs surface-attached
bacteria to twitch upstream. Biophys. J. 103, 146–151 (2012).
24. A. Zöttl, h. Stark, nonlinear dynamics of a microswimmer in Poiseuille flow. Phys. Rev. Lett.
108, 218104 (2012).
25. c.-K. tung, F. Ardon, A. Roy, d. l. Koch, S. S. Suarez, M. Wu, emergence of upstream
swimming via a hydrodynamic transition. Phys. Rev. Lett. 114, 108102 (2015).
26. A. J. Mathijssen, t. n. Shendruk, J. M. Yeomans, A. doostmohammadi, Upstream
swimming in microbiological flows. Phys. Rev. Lett. 116, 028104 (2016).
27. Z. Peng, J. F. Brady, Upstream swimming and taylor dispersion of active Brownian
particles. Phys. Rev. Fluids 5, 073102 (2020).
28. G. i. taylor, dispersion of soluble matter in solvent flowing slowly through a tube.
Proc. R. Soc. A-Math. Phys. Eng. Sci. 219, 186–203 (1953).
29. t. Kaya, h. Koser, characterization of hydrodynamic surface interactions of Escherichia coli
cell bodies in shear flow. Phys. Rev. Lett. 103, 138103 (2009).
30. v. Kantsler, J. dunkel, M. Blayney, R. e. Goldstein, Rheotaxis facilitates upstream
navigation of mammalian sperm cells. eLife 3, e02403 (2014).
31. t. Omori, t. ishikawa, Upward swimming of a sperm cell in shear flow. Phys. Rev. E. 93,
032402 (2016).
32. n. Figueroa-Morales, A. Rivera, R. Soto, A. lindner, e. Altshuler, É. clément, E. coli
“super-contaminates” narrow ducts fostered by broad run-time distribution. Sci. Adv. 6,
eaay0155 (2020).
33. B. e. logan, t. A. hilbert, R. G. Arnold, Removal of bacteria in laboratory filters: Models and
experiments. Water Res. 27, 955–962 (1993).
34. W. dzik, Use of leukodepletion filters for the removal of bacteria. Immunol. Invest. 24,
95–115 (1995).
35. l. Fernandez Garcia, S. Alvarez Blanco, F. A. Riera Rodriguez, Microfiltration applied to
dairy streams: Removal of bacteria. J. Sci. Food Agric. 93, 187–196 (2013).
36. G. Franci, A. Falanga, S. Galdiero, l. Palomba, M. Rai, G. Morelli, M. Galdiero, Silver
nanoparticles as potential antibacterial agents. Molecules 20, 8856–8874 (2015).
37. M. i. hutchings, A. W. truman, B. Wilkinson, Antibiotics: Past, present and future. Curr.
Opin. Microbiol. 51, 72–80 (2019).
38. J. W. costerton, h. M. lappin-Scott, introduction to microbial biofilms, in Microbial Biofilms,
h. M. lappin-Scott, J. W. costerton, eds. (cambridge Univ. Press, 1995), pp. 1–11.
39. W.- h. Sheng, W.-J. Ko, J.-t. Wang, S.-c. chang, P.-R. hsueh, K.-t. luh, evaluation of
antiseptic-impregnated central venous catheters for prevention of catheter-related
infection in intensive care unit patients. Diagn. Microbiol. Infect. Dis. 38, 1–5 (2000).
40. W. M. dunne Jr., Bacterial adhesion: Seen any good biofilms lately? Clin. Microbiol. Rev. 15,
155–166 (2002).
41. R. P. Allaker, the use of nanoparticles to control oral biofilm formation. J. Dent. Res. 89,
1175–1186 (2010).
42. M. l. Knetsch, l. h. Koole, new strategies in the development of antimicrobial coatings:
the example of increasing usage of silver and silver nanoparticles. Polymers 3, 340–366
(2011).
43. M. Birkett, l. dover, c. cherian lukose, A. Wasy Zia, M. M. tambuwala, Á. Serrano-Aroca,
Recent advances in metal-based antimicrobial coatings for high-touch surfaces. Int. J.
Mol. Sci. 23, 1162 (2022).
44. J. R. lex, R. Koucheki, n. A. Stavropoulos, J. di Michele, J. S. toor, K. tsoi, P. c. Ferguson,
R. e. turcotte, P. J. Papagelopoulos, Megaprosthesis anti-bacterial coatings: A
comprehensive translational review. Acta Biomater. 140, 136–148 (2022).
45. J. Monod, the growth of bacterial cultures. Annu. Rev. Microbiol. 3, 371–394 (1949).
46. M. hecker, W. Schumann, U. völker, heat-shock and general stress response in Bacillus
subtilis. Mol. Microbiol. 19, 417–428 (1996).
47. P. Setlow, Spores of Bacillus subtilis: their resistance to and killing by radiation, heat and
chemicals. J. Appl. Microbiol. 101, 514–525 (2006).
48. M. Falagas, P. thomaidis, i. Kotsantis, K. Sgouros, G. Samonis, d. Karageorgopoulos,
Airborne hydrogen peroxide for disinfection of the hospital environment and infection
control: A systematic review. J. Hosp. Infect. 78, 171–177 (2011).
49. W. A. Rutala, d. J. Weber, disinfection and sterilization in health care facilities: What
clinicians need to know. Clin. Infect. Dis. 39, 702–709 (2004).
50. W. A. Rutala, d. J. Weber, disinfection and sterilization: An overview. Am. J. Infect. Control
41, S2–S5 (2013).
51. n. P. tipnis, d. J. Burgess, Sterilization of implantable polymer-based medical devices: A
review. Int. J. Pharm. 544, 455–460 (2018).
52. M. Berger, R. Shiau, J. M. Weintraub, Review of syndromic surveillance: implications for
waterborne disease detection. J. Epidemiol. Community Health 60, 543–550 (2006).
53. M. v. Storey, B. van der Gaag, B. P. Burns, Advances in on-line drinking water quality
monitoring and early warning systems. Water Res. 45, 741–747 (2011).
54. S. hyllestad, e. Amato, K. nygård, l. vold, P. Aavitsland, the effectiveness of syndromic
surveillance for the early detection of waterborne outbreaks: A systematic review.
BMC Infect. Dis. 21, 696 (2021).
55. F. Baquero, J.- l. Martínez, R. cantón, Antibiotics and antibiotic resistance in water
environments. Curr. Opin. Biotechnol. 19, 260–265 (2008).
56. R. i. Aminov, the role of antibiotics and antibiotic resistance in nature. Environ. Microbiol.
11, 2970–2988 (2009).
57. J. M. Munita, c. A. Arias, Mechanisms of antibiotic resistance. Microbiol Spectr, (2016).
58. U. theuretzbacher, K. Bush, S. harbarth, M. Paul, J. h. Rex, e. tacconelli, G. e. thwaites,
critical analysis of antibacterial agents in clinical development. Nat. Rev. Microbiol. 18,
286–298 (2020).
59. R. di Giacomo, S. Krödel, B. Maresca, P. Benzoni, R. Rusconi, R. Stocker, c. daraio,
deployable micro- traps to sequester motile bacteria. Sci. Rep. 7, 45897 (2017).
60. P. Galajda, J. Keymer, P. chaikin, R. Austin, A wall of funnels concentrates swimming
bacteria. J. Bacteriol. 189, 8704–8707 (2007).
61. c. M. Kjeldbjerg, J. F. Brady, theory for the casimir effect and the partitioning of active
matter. Soft Matter 17, 523–530 (2021).
62. Z. li, n. Kovachki, K. Azizzadenesheli, B. liu, K. Bhattacharya, A. Stuart, A. Anandkumar.
Fourier neural operator for parametric partial differential equations. arXiv:2010.08895
[cs.lG] (2020).
63. Z. li, d. Z. huang, B. liu, A. Anandkumar. Fourier neural operator with learned
deformations for pdes on general geometries. arXiv:2207.05209 [cs.lG] (2022).
64. A. J. Mathijssen, n. Figueroa-Morales, G. Junot, É. clément, A. lindner, A. Zöttl, Oscillatory
surface rheotaxis of swimming E. coli bacteria. Nat. Commun. 10, 3434 (2019).
65. S. B. Goodman, Z. Yao, M. Keeney, F. Yang, the future of biologic coatings for orthopaedic
implants. Biomaterials 34, 3174–3183 (2013).
66. A. Jaggessar, h. Shahali, A. Mathew, P. K. Yarlagadda, Bio-mimicking nano and
micro-structured surface fabrication for antibacterial properties in medical implants.
J. Nanobiotechnol. 15, 64 (2017).
67. e. Macedo, R. Malhotra, R. claure- del Granado, P. Fedullo, R. l. Mehta, defining urine
output criterion for acute kidney injury in critically ill patients. Nephrol Dial Transplant 26,
509–515 (2011).
68. K. B. chenitz, M. B. lane-Fall, decreased urine output and acute kidney injury in the
postanesthesia care unit. Anesthesiol. Clin. 30, 513–526 (2012).
69. J. A. Kellum, F. e. Sileanu, R. Murugan, n. lucko, A. d. Shaw, G. clermont, classifying AKi by
urine output versus serum creatinine level. J. Am. Soc. Nephrol. 26, 2231–2238 (2015).
70. S. Mirjalili, Genetic algorithm, in Evolutionary Algorithms and Neural Networks: Theory and
Applications (Springer, 2019), pp. 43–55.
71. S. Ruder, An overview of gradient descent optimization algorithms. arXiv:1609.04747
[cs.lG] (2016).
72. c. Multiphysics, introduction to cOMSOl multiphysics. cOMSOl Multiphysics, Burlington,
MA, accessed 2018 Feb 9: 32 (1998).
73. G. B. Jeffery, the motion of ellipsoidal particles immersed in a viscous fluid. Proc. R. soc. Lond.
Ser. A-Contain. Pap. Math. Phys. Character 102, 161–179 (1922).
74. F. P. Bretherton, the motion of rigid particles in a shear flow at low Reynolds number.
J. Fluid Mech. 14, 284–304 (1962).
75. t. Zhou, Z. Peng, M. Gulian, J. F. Brady, distribution and pressure of active lévy swimmers
under confinement. J. Phys. A Math. Theor. 54, 275002 (2021).
76. P. i. Frazier, J. Wang, Bayesian optimization for materials design, in Information Science for
Materials Discovery and Design (Springer, 2016), pp. 45–75.
77. Y. Zhang, d. W. Apley, W. chen, Bayesian optimization for materials design with mixed
quantitative and qualitative variables. Sci. Rep. 10, 4924 (2020).
78. J. Schindelin, i. Arganda-carreras, e. Frise, v. Kaynig, M. longair, t. Pietzsch, S. Preibisch,
c. Rueden, S. Saalfeld, B. Schmid, Fiji: An open-source platform for biological-image
analysis. Nat. Methods 9, 676–682 (2012).
79. d. ershov, M.-S. Phan, J. W. Pylvänäinen, S. U. Rigaud, l. le Blanc, A. charles- Orszag,
J. R. conway, R. F. laine, n. h. Roy, d. Bonazzi, Bringing trackMate into the era of
machine-learning and deep-learning. bioRxiv 458852 [Preprint] (2021). https://doi.
org/10.1101/2021.09.03.458852.
