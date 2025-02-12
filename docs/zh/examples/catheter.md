# AI-aided geometric design of anti-infection catheters(人工智能辅助的抗感染导管几何设计)

Distributed under a creative commons Attribution license 4.0 (CC BY).

## 1. 背景简介
### 1.1 论文信息
| 年份           | 期刊            | 作者                                                                                             | 引用数 | 论文PDF                                                                                                                                                                                                                                                                                                                                                                 |
| -------------- | --------------- | ------------------------------------------------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3 January 2024 | Science Advance | Tingtao Zhou, X Wan, DZ Huang, Zongyi Li, Z Peng, A Anandkumar, JF Brady, PW Sternberg, C Daraio | 15     | [Paper](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters.pdf), [Supplementary PDF 1](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/sciadv.adj1741_sm.pdf) |

### 1.2 作者介绍

- 第一作者：加州理工学院 Tingtao Zhou <br> 研究方向：统计物理学、流体力学、活性物质、无序材料 <br> ![alt text](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter5.png)

- 通讯作者：加州理工学院 工程与应用科学部 Chiara Daraio (Cited 21038) <br> 教师主页：https://www.eas.caltech.edu/people/daraio <br> 研究方向：力学 材料 非线性动力学 软物质 生物材料 <br> ![alt text](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter6.png)

- 通讯作者：加州理工学院 生物学和生物工程学部 Paul W. Sternberg (Cited 56555) <br> 教师主页：https://www.bbe.caltech.edu/people/paul-w-sternberg <br> 研究方向：秀丽隐杆线虫发育的系统生物学；性别与睡眠背后的神经回路；线虫功能基因组学与化学生态学；文本挖掘。 <br> ![alt text](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter7.png)

- 其他作者所属机构 <br> 加州理工学院,工程与应用科学部\化学与化学工程系\生物与生物工程系 <br> 北京大学,北京国际数学研究中心 <br> Meta Platforms公司(前Facebook)，Reality Labs部门

### 1.3 模型&复现代码

| 问题类型             | 在线运行                                                                                                                   | 神经网络           | 预训练模型                                                                                                                                                              | 指标              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| 算子神经网络预测流场 | [人工智能辅助的抗感染导管几何设计](https://aistudio.baidu.com/projectdetail/8252779?sUid=1952564&shared=1&ts=172724369783) | 傅立叶几何神经算子 | [GeoFNO_pretrained.pdparams](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/result_GeoFNO.pdparams) | loss(MAE): 0.0664 |


=== "模型训练命令"

    ``` sh
    # linux
    wget https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/data.zip
    # windows
    # curl https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/data.zip -o data.zip
    unzip data.zip
    python catheter.py
    ```

=== "预训练模型快速评估"

    ``` sh
    python catheter.py mode=eval EVAL.pretrained_model=https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/result_GeoFNO.pdparams
    ```

在狭窄管道内的流体环境中，细菌能借助流体动力学作用逆流迁移，对使用植入性导管的患者构成泌尿道感染的严重威胁。尽管已有提议采用涂层与结构化表面来抑制导管内的细菌滋生，但遗憾的是，至今尚无一种表面结构或涂层技术能从根本上解决污染难题。鉴于此，我们依据逆流游动的物理原理，创新性地提出了一种几何设计方案，并通过AI模型对细菌流入动力学进行预测与优化。相较于传统模拟方法，所采用的傅立叶神经算子人工智能技术实现了显著的速度提升。

在准二维微流体实验中，我们以大肠杆菌为对象，验证了该设计的抗感染机制，并在临床相关流速下，通过 3D 打印的导管原型对其有效性进行了评估。实验结果显示，我们的导管设计在抑制导管上游端细菌污染方面，实现了 1-2 个数量级的提升，有望大幅延长导管的安全留置时间，并整体降低导管相关性尿路感染的风险。

## 2. 问题定义

导管相关尿路感染（CAUTIs）（1-5）是住院患者中最常见的感染之一，每年约造成3000万美元的损失（6）。从材料/设备的角度来看，以往预防此类感染的方法包括用抗菌银nm粒子浸渍导管（7）或使用抗生素锁溶液、抗粘附或抗菌材料（8, 9）。然而，这些方法的效果均未能超越严格的护理程序，当前临床实践中预防CAUTI的重点是减少导管的留置时间来预防感染。设计一种在流体存在下能减少细菌活动性的导管，将对当前CAUTI的管理带来显著改善。

这样的设计需要我们了解微生物在受限条件下流体流动中的运动模式。典型的微生物轨迹在奔跑（直线推进）和翻滚（随机改变方向）之间交替，以探索环境（10-13）。流体动力学相互作用和群体感应导致更复杂的动态行为，如增强对表面的吸引力（14, 15）和集体群游运动（16-19）。在剪切流中，微观的奔跑-翻滚（RTP）运动可以导致宏观的逆流游动（20-27）。通常，被动粒子除了扩散扩散外，还会被对流至下游（28）。然而，微生物的自驱动导致其宏观传输在性质上有所不同：细菌体在穿越管道时会被流体涡度旋转，从而使其逆着流动方向游动。生物微游动体和合成主动粒子都表现出逆流运动性。对于生物微游动体，如大肠杆菌和哺乳动物精子，其前后体不对称性以及由此产生的与壁面的流体动力学相互作用，通常被用来解释其逆流游动行为（20, 21, 25, 29-31）。

另一方面，对于可忽略大小的点状主动粒子，逆流游动现象仍然存在（24, 27）。考虑一个点状主动粒子接近壁面的情况：其前端必须指向壁面。在壁面附近，泊肃叶流（在其最大值处）的涡度总是使粒子重新定向到上游方向（也见材料与方法部分）（27），然后它们沿着壁面逆流游动（图1，A和B）。许多其他因素，如体形不对称、鞭毛的手性以及细菌与边界之间的流体动力学相互作用，也会影响逆流游动行为。最近的实验（32）已经证明了在微流控通道中大肠杆菌的超污染现象，这凸显了其幂律运行时间分布的重要性，该分布显著增强了细菌逆流游动的倾向，使细菌能够持续逆着流动方向游动。

预防细菌污染的主流策略包括以下几种：

- （i）物理屏障，如过滤器或膜（33-38）；

- （ii）抗菌剂，如抗生素（36, 37）；

- （iii）对医疗设备进行表面改性以减少细菌粘附和生物膜形成（38-44）；

- （iv）控制物理/化学环境，如高温/低温、低氧水平或使用消毒剂来抑制细菌的生长和存活（45-48）；

- （v）严格的消毒程序，如戴手套和穿隔离衣（49-51）；

- （vi）定期监测患者状况，以便及早发现并治疗细菌污染（52-54）。

虽然已经提出了各种表面改性或涂层来减少细菌粘附，但尚未有研究表明它们能有效防止逆流游动或导管污染（38-40）。其他被动的抗菌方法，如膜或过滤，可能难以直接应用于留置导管的患者。

与抗生素或其他化学方法相比，通过几何形状控制微生物分布在抗生素耐药性方面更为安全（55-58）。在其他情况下，已经使用了特定形状来限制和捕获不需要的细菌（59）。由于“干燥”的几何整流效应，不对称形状也可以影响运动细菌的分区（60, 61），并且挤出的边界形状可以局部增强泊肃叶流的涡度，增强程度与挤出曲率成正比。

我们致力于设计能够防止细菌逆流游动并最大程度减少污染的导管。为了优化导管的几何形状，我们将设计空间限制为在导管内壁放置三角形障碍物。我们捕捉了自驱动球体出现的最简单的逆流游动物理机制（27），并进行了流体和粒子动力学模拟，以找出几何设计原则（图1C）。我们将流体动力学和几何整流效应结合为一个随机偏微分方程（SPDE），以此模拟细菌的分布。然后，我们使用模拟数据训练了一个基于几何聚焦傅里叶神经算子（Geo-FNO）的人工智能（AI）模型（62, 63），以学习SPDE的解，并使用训练好的模型来优化导管的几何形状（图1D）。基于优化后的设计，我们制造了准二维（2D）微流控装置（图1E）和3D打印的原型导管（图1F），以评估我们的概念的有效性。实验结果表明，与我们的标准导管相比，细菌超污染抑制效果提高了多达两个数量级，这为导管相关尿路感染（CAUTI）的管理提供了一条新途径。

![图1. 提出的导管相关尿路感染（CAUTI）机制与抗感染设计流程示意图](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter.png)

**图1. 提出的导管相关尿路感染（CAUTI）机制与抗感染设计流程示意图**

- **（A）提出的CAUTI机制**：尿液从患者膀胱内通过导管向外流出时，细菌能够逆着尿流方向（即上游）游动，进而可能侵入患者体内并引发感染。
- **（B）细菌的跑动-翻滚运动与上游游动机制**：细菌通过一种特有的跑动-翻滚运动模式，在液体环境中实现上游游动。
- **（C）模拟探索导管形状**：利用模拟技术，探索不同导管形状对细菌上游游动的影响，以期找到能够抑制细菌上游游动的导管设计。
- **（D）人工智能辅助优化**：采用Geo-FNO框架进行人工智能辅助优化，进一步细化导管的设计参数，提升其对细菌上游游动的抑制效果。
- **（E）二维通道微流控实验**：在二维微流控通道中，对优化后的导管设计进行实验验证，评估其在实际流体环境中的抗感染性能。
- **（F）三维实验验证**：使用设计的实际尺寸导管进行三维实验，进一步验证其在临床使用条件下的抗感染效果。

我们致力于设计能够防止细菌向上游移动并最大程度减少污染的导管。为了优化导管的几何形状，我们将设计空间限定为在导管内壁布置三角形障碍物。我们捕捉了自驱动球体所展现的最简单的上游游动物理机制（27），并进行了流体和粒子动力学模拟，以找出几何设计原则（图1C）。通过将流体力学和几何整流效应耦合为随机偏微分方程（SPDE），我们对细菌分布进行了建模。随后，我们使用模拟数据训练了一个基于几何聚焦傅里叶神经算子（Geo-FNO）（62, 63）的人工智能（AI）模型，以学习SPDE的解，并利用训练好的模型来优化导管的几何形状（图1D）。基于优化后的设计，我们制作了准二维（2D）微流控装置（图1E）和3D打印的原型导管（图1F），以评估我们的设计理念的有效性。实验结果表明，与标准导管相比，我们的设计在抑制细菌超标污染方面提高了多达两个数量级，为导管相关尿路感染（CAUTI）的管理提供了一条新途径。

### 2.1 探究微观机制

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
- $\vec{q}$ 代表细菌方向向量
- $\tau_R$ 代表平均运行时间(更多细节详见补充材料)

我们首先通过数值模拟研究了传统表面改性方法，如抗菌nm粒子涂层（36, 42）、工程化粗糙度或疏水性处理（65, 66），在抑制细菌上游游动中的作用。这些改性表面能够防止细菌过于接近壁面。为了模拟这些表面的存在，我们假设它们会导致细菌从表面脱离，并至少保持在距离表面3μm的位置，这个距离超过了典型的E.coli大肠杆菌体长（1至2μm）。虽然表面改性也可能影响细菌与壁面之间的流体力学相互作用，但在我们基于点状球体的简单通用模型中忽略了这一点。

我们发现，在所测试的流速范围内，表面排斥对细菌的上游游动行为几乎没有影响。通过比较光滑通道内（图2D）和表面改性通道内（图2E）持续游动细菌的模拟轨迹，我们发现它们的上游游动行为相似。

我们采用两个群体统计指标来量化抑制细菌上游游动的有效性：

- （i）平均上游游动距离$x_{up}=-\int_{0}^{-\infty}\rho(x)xdx$，通过计算细菌分布函数$ρ(x)$的加权平均值得出，其中$x$为细菌位置；

- （ii）前$1\%$上游游动最远的细菌所能到达的距离$x_{1\%}$。模拟结果显示，表面改性仅在中等流速下略微减少了$x_{up}$，但对$x_{1\%}$几乎没有影响（图2F中的蓝线和粉线）。这种表面改性效果不佳的结果与近几年一些论文的实验观察结果一致（39, 40）。

随后，我们通过添加物理障碍物来探索导管表面几何形状的作用。我们发现，对称和不对称的障碍物都能显著抑制细菌的上游游动（如图2F中的黑色和绿色线条所示）。我们确定了两种协同效应：首先，障碍物的斜率会在细菌从障碍物顶部出发时改变其游动方向，从而打断了它们沿着管壁表面的连续攀爬。不对称的形状会使细菌的运动偏向下游（如图2A所示），这在模拟的0流速下的轨迹（补充材料和图S1）以及低流速下上游游动统计数据的差异（图2F中的黑色和绿色线条）中均有所体现。其次，在有限的流速下，流场与光滑通道中的泊肃叶流不同（如图2B所示）。在泊肃叶流中，涡量会使细菌转向下游。而在障碍物附近，涡量会增强，导致细菌转向上游（如图2C和补充材料图S2所示）, 从而加强了细菌的转向机制。结合这两种效应，我们预计在具有优化障碍物几何形状的通道中，细菌的上游游动将显著减少。

设计优化的参数空间由四个参数表征：障碍物基底长度$L$、高度$h$、尖端位置$s$以及障碍物间距$d$；我们用W表示通道宽度（图2G）。为了优化这个空间，我们设定了两个约束条件。首先，如果相邻障碍物过于接近，它们尖端的涡旋就会开始重叠。由于这种重叠，最大有效涡旋强度（正好在障碍物尖端；有效涡旋的数学定义见补充材料）和涡旋的有效尺寸都会减小。此外，还会形成更大的边界层和滞流区（图S2，A和B）。因此，我们将障碍物间距约束为$d > 0.5W$（图S2G）。其次，在其他参数固定的情况下，随着h的增加，障碍物尖端的有效涡旋强度也会增加（图S2，C至H），这有利于促进涡旋重定向效应。然而，当$h = W/2$时，管道显然会发生堵塞。这种随着$h$增加而堵塞加剧的趋势反映在为了保持相同的有效流速而所需压力降的持续增加上（图S2I）。为了避免堵塞，我们将高度约束为$h < 0.3W$。

![图2. 障碍物抑制上游游动和几何优化的物理机制](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter2.png)

**图2. 障碍物抑制上游游动和几何优化的物理机制**

- **（A）无流动时的几何整流效应**：描述了在没有流体流动的情况下，几何形状对细菌游动方向的影响。

- **（B）光滑通道中的泊肃叶流**：彩色背景显示流涡量的相对大小，颜色越深表示涡量越大。在光滑通道中，泊肃叶流产生的涡量使细菌头部向下游旋转。

- **（C）带有对称障碍物的通道中的流动**：在带有对称障碍物的通道中，障碍物顶部附近的流速和涡量增强，这导致更强的扭矩作用在细菌上，使其重定向至下游。

- **（D）和（E）不同条件下的细菌模拟轨迹**：
  - - **（D）光滑通道**：在宽度为50μm的二维光滑通道中，细菌的模拟轨迹显示其持续游动状态。
  - - **（E）排斥细菌的表面改性通道**：在表面经过改性以排斥细菌的通道中，细菌的游动轨迹受到显著影响。

- **（F）上游游动的群体统计**：
  - - 实线（左侧y轴）表示平均上游距离，反映了细菌群体在上游方向上的平均游动距离。
  - - 虚线（右侧y轴）表示群体中前1%游动者的上游距离，揭示了少数高效游动细菌的表现。
  - - 不同颜色的线条代表不同的通道条件：蓝色为光滑通道，橙色为表面改性通道，黑色为对称障碍物通道，绿色为不对称障碍物通道。

- **（G）AI算子神经网络模型和结果**：
  - - Geo-FnO模型旨在学习导管几何形状与细菌分布之间的关系，通过一系列神经算子层实现。
  - - 模型首先将不规则的通道几何形状映射到单位段[0,1]，然后在潜在空间中应用基于傅里叶的内核进行预测。
  - - 最后，将预测的细菌分布从潜在空间转换回物理空间。
  - - 右图展示了随机初始条件（黑色）和优化后的设计（粉色）的对比，以及通过流体和粒子动力学模拟验证的Geo-FnO预测结果（绿色虚线）。

![图3. 微流控实验](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter3.png)

**图3. 微流控实验**

- **（A）微流控实验示意图**：微流控通道的一端连接着装有成像溶液的注射器，另一端则连接着装有大肠杆菌的储液池。长箭头表示流动方向。

- **（B）细菌在锐角处的积聚**：由于流动停滞，细菌在通道的锐角处积聚。

- **（C）微流控通道的明场图像**：展示了通道的实际结构。

- **（D）细菌从通道壁上脱落的典型事件**：
  - - 细菌（白色点）的轨迹在过去5s内以黄色线条显示。
  - - 上图展示了一种类型1的轨迹，其中细菌从障碍物尖端脱落。
  - - 下图展示了一种典型的类型2轨迹，其中细菌从通道的平滑部分脱落。
  - - 左列为实验图像，右列为模拟图像。

- **（E）脱落事件的统计**：提供了关于细菌脱落事件的统计数据。

![图4. 3D打印导管原型的实验](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter4.png)

**Fig. 4. 3D打印导管原型的实验**

- **（A）实验设置**：导管的下游端连接到大肠杆菌的储液池，上游端连接到由注射泵控制的装满培养液的注射器。1小时后，将导管切成等长段，并提取内部液体进行24小时培养。在显微镜下计数大肠杆菌菌落的数量，以反映每段导管中的细菌数量。

- **（B）光滑导管中的大肠杆菌超污染**：展示了在光滑导管中大肠杆菌的污染情况。

- **（C）设计导管与光滑导管的比较**：对比了设计导管与光滑导管在细菌污染方面的差异。插图显示了相同数据在对数尺度上的绘制。

### 2.2 AI辅助优化的几何设计

近年来，基于人工智能（AI）的模型，如神经算子，已被用于学习流体动力学和其他领域的正向模拟或观测模型的替代品。由于这些模型是可微分的，因此它们可以直接用于逆向设计，即我们可以使用梯度直接在设计空间中进行优化。这使得生成以前未研究过的设计变得更加简洁高效。我们使用了一个AI模型来优化通道形状，该形状由上述描述的四个参数和两个约束条件表征（图2G）。这种方法首先将不规则的通道几何形状映射到潜在空间（一个单位的导管片段长度$[0,1]$）中的一个函数，然后在潜在空间中应用傅里叶神经算子（FNO）模型，最后将细菌分布转换回物理空间（图2G）。然后，我们使用这个训练好的替代模型进行逆向设计优化，以确定最佳的通道形状。为了评估每种设计的有效性，我们测量了在$T=500$s时，三种流速（$5、10$和$15μm/s$）下的平均⟨$x_{up}$⟩值。我们基于几何感知傅里叶神经算子的AI辅助形状设计，在加权细菌分布方面比训练数据中的给定形状提高了约$20\%$。整个设计优化过程非常快速：并行生成1000个训练实例（在50个GPU上运行10小时），每个实例需要30分钟；在1个GPU上训练模型需要20分钟；而我们训练好的AI模型在1个GPU上生成最优设计仅需15s。优化过程得出了以下最优结构参数：$d=62.26μm，h=30.0μm，s=-19.56μm，L=12.27μm$，对于通道宽度$W=100μm$。根据上文所述的机制，这种结构提供了强大的几何整流和涡旋重定向效应，以抑制细菌的逆流游动。

### 2.3 微流控实验

为了评估优化结构的有效性，我们制作了宽度$W=100μm$（壁到壁的距离）且垂直深度为20μm的准二维微流控通道，以便在显微镜下观察细菌的运动情况（图3A）。我们选取了逆流游动的细菌子集，并根据它们从壁上脱落的位置进行了分类。如果细菌从障碍物的顶部脱落，则将其轨迹标记为“类型1”（图3D，上方）；如果细菌从壁的平滑部分脱落，则将其标记为“类型2”（图3D，下方）。类型1的轨迹会同时受到几何整流和增强的流体动力学旋转破坏效应的影响。而类型2的轨迹则不会受到几何整流效应的影响，仅会受到轻微的涡旋重定向效应，因为涡量的增强在障碍物尖端最为强烈。对于流速$U_0<100μm/s$的情况，70%到80%的逆流游动轨迹属于类型1（图3E）。我们还注意到，在这些实验中观察到的所有逆流游动轨迹都被重定向到了下游（图3E，红线）。在尖锐的拐角附近观察到了细菌积聚现象（图3B），这可能是由于停滞区的存在（图2C和附图S1，拐角附近的白色区域）。为了防止细菌在拐角处积聚，我们将几何形状用半径$r=h/2$的圆弧进行了圆滑处理（图3C）。

### 2.4 宏观尺度导管实验

上述展示的机制和设计原则很容易扩展到导管上。在三维管道中，细菌可以通过横截面的任何切割线穿过管道（附图S2J）。由于与上述相同的机制（图2，A、B、F至I，以及附图S1），仅在壁附近的无量纲剪切率起作用（27），因此靠近边界移动的细菌（附图S2J中的轨迹1）仍然可以逆流游动。超污染细菌的游动距离可以超过1mm（32），这与重新缩放的障碍物尺寸相当，预计在这些尺度上整流效应会持续存在（61）。数量级估计表明，也可以采用伴随方法的雷诺数下降（71）。我们注意到，几何设计不能完全消除细菌的逆流游动，特别是在接近零流速的情况下。然而，它极大地减少了超污染的数量，并可能显著延长导管的留置时间。使用我们设计的导管预计不需要改变常规临床方案或重新培训医务人员。此外，我们的解决方案不会向导管中引入化学物质，因此是安全的，并且不需要额外的维护。我们的几何设计方法预计与其他程序措施、抗菌表面改性和环境控制方法兼容。


![S1. 微流控实验](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheterS1.png)

**图S1. 主动布朗粒子的模拟轨迹示例**

在（A）（C）具有对称障碍物的通道中和（B）（D）具有不对称障碍物的通道中的轨迹。（A）（B）无流体流动。（C）（D）有流体流动。颜色表示局部归一化涡量。

![S2. 主动布朗粒子的模拟轨迹示例](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheterS2-1.png)
![S2. 主动布朗粒子的模拟轨迹示例](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheterS2-2.png)

**图S2. 几何优化约束和缩放的考虑因素**

- （A-H）归一化涡量作为障碍物高度ℎ和障碍物间距𝑑的函数。
- （I）沿通道一个周期内的归一化压降作为归一化障碍物高度ℎ/𝑑的函数。
- （J）宏观圆柱形管内细菌运动的横截面视图。与轨迹2相比，轨迹2中的细菌在管中心附近经历强烈的下游流动，而采取轨迹1的另一个细菌则靠近管壁。因此，细菌1所经历的流动条件与我们考虑的微流控通道中的条件更为相似。其上游游动行为和几何抑制机制将与微流控条件相似，只是存在定量上的差异。

### 2.5 流体和粒子动力学模拟

我们使用COMSOL软件（72）模拟了具有无滑移边界条件的通道内的斯托克斯流。随后，将得到的速度和涡量场耦合到粒子动力学模拟中，而在稀释悬浮液和小粒子尺寸的极限情况下，忽略了粒子运动对流体动力学的反馈。粒子动力学由具有高斯统计特性的主动布朗粒子（ABP）模型和具有幂律（Levy）统计特性的运行-休止（RTP）模型描述。模拟是使用我们内部开发的GPU Julia代码进行的，模拟时间步长为10^-4s。在ABP模型中，单个粒子的动力学是根据过阻尼的朗之万方程进行积分的。

$$0=-\zeta(U-u)+\zeta U_0q(t)+\sqrt{2D_T\xi(t)}$$

$$d{q}/dt=\left[1/2{\omega}+B{q}\times(\mathbf{E}\cdot{q})+\sqrt{2/\tau_R{\eta}({t})}\right]\times{q}$$

其中，$ζ$是粘性阻力系数，$U$是粒子的速度，${q}$是粒子的方向向量，$u$是局部流速，$ω$是局部流场的涡量向量，$E$是流场的局部应变率张量。$B$是一个几何系数（3、74），对于无限细的杆状物体，它等于1，对于球体，它等于0。由于B的值对上游游动统计的影响不显著（27），因此我们在这里展示的结果中设$B=0$。$ξ(t)$是满足$⟨ξ(t)⟩=0和⟨ξ(0)ξ(t)⟩=δ(t)I$的高斯随机噪声。由于细菌是μm级的粒子，它们的布朗运动相对较弱，因此在模拟中我们将平移扩散系数DT设置为$0.1 μm²/s$。只要这个值保持较小，其变化对结果的影响就不大。

η是满足$⟨η(t)⟩=0和⟨η(0)η(t)⟩=δ(t)I$的高斯噪声，τR是平均运行时间。在RTP（Run-and-Tumble，奔跑-翻滚）模型中，单个粒子在$0<t<τ_R$的时间内以$η(t)=0$（即“奔跑”阶段）进行位移。然后，${q}$会瞬间改变为一个随机的新方向${q'}$（即“翻滚”），并以新的运行时间$τ_R$重复此过程。对于Lévy游动粒子，运行时间是从Pareto分布中采样的，该分布的形式为:

$$ϕ(τ)=\frac{(ατ₀^α)}{(τ+τ₀)^{(α+1)}}$$

，其中$1<α<2$的参数控制幂律指数（75）。

为了简化，我们将细菌的形状视为可忽略大小的球体。在图2J的机制演示中，我们在一个50μm宽的二维通道中模拟了1,000,000个粒子，每个粒子的持续运行时间$τ_R$为2s，模拟总时间为200s。沿通道方向，对流场和粒子动力学都施加了周期性边界条件。因此，通道在效果上是无限长的，并且障碍物也是每隔100μm重复一次。粒子在计算域的$x=$处释放，最初在通道中均匀分布且随机定向。

对于设计的通道，除了表面涂层的情况外，在墙壁的几何边界上施加了粒子动力学的滑动边界条件和流体力学的无滑移边界条件。在表面涂层的情况下，无滑移边界位于墙壁处，而粒子的滑动边界条件则设置在距离墙壁$3μm$的位置。

### 2.6 Geo-Fno模型和Machine Learning配置

导管设计问题是一个受随机偏微分方程（SPDE）约束的优化问题，其中目标函数

$$⟨x_{up}⟩ = - ∫^{-∞}_{0} ρ(x)xdx ≈ - \frac{1}{N} ∑_{i=1}^{N} xi$$

依赖于流体和粒子动力学问题的SPDE解。这里，ρ(x)是在$T=500 s$时的经验细菌分布函数，由$N$个细菌近似表示。传统的优化方法需要反复评估这种计算成本高昂的模型，并且在应用基于梯度的优化时还需要一个伴随求解器。为了克服这些计算挑战，我们训练了一个几何傅里叶神经算子（Geo-FNO）G作为前向流体和粒子动力学模拟的替代模型，该模型将通道几何形状映射到细菌种群函数$G: c → ρ$。相比之下，先前使用人工智能方法解决各种设计问题的工作仅选择了少数几个参数作为SPDE传统求解器的输入（76, 77）。完整模型由五个带有高斯误差线性单元（GeLU）激活层的傅里叶神经层组成，具有快速的准线性时间复杂度。

我们使用ABP和Levy RTP模型进行了三种最大流速（5、10和15μm/s）下的流体和粒子动力学模拟，以生成Geo-FNO的训练和测试数据。对于训练数据，我们在50个GPU上并行生成了1000个模拟，耗时10小时，每个模拟中的设计随机选自以下参数空间：

- 高度为$20μm < h < 30μm$的障碍物周期性地放置在通道壁上
- 障碍物间距离为$60μm < d < 250μm$
- 基底长度满足$15μm < L < \frac{d}{4}$，并且尖端位置满足$-\frac{d}{4} < s < \frac{d}{4}$。

这些参数的约束条件是为了满足制造限制和涡旋生成机制的物理条件（图2，B和C；更多讨论见补充文本和附图S2）。数据集被存储起来以供未来任务重复使用。我们使用相对经验均方误差作为损失函数。模型训练在1个GPU上使用Adam优化器进行了20分钟。它在100个测试数据点上获得了大约$4\%$的相对误差。

### 2.7 基于梯度的快速逆向设计优化

我们的人工智能方法相较于传统求解器具有显著的速度优势，并且其可微性使得能够快速应用基于梯度的方法进行几何设计优化。在GPU上，每次评估仅需0.005s，而使用基于GPU的流体和粒子动力学模拟则需要10分钟，因此，在优化过程中进行数千次评估变得切实可行。此外，我们利用深度学习包中的自动微分工具，高效计算相对于设计变量的梯度，从而能够应用基于梯度的设计优化方法。

在优化过程中，我们从初始设计参数（$d = 100 μm, h = 25 μm, s = 10 μm, L = 20 μm$）开始，并使用Broyden–Fletcher–Goldfarb–Shanno（BFGS）算法来最小化由Geo-FNO预测的细菌种群后处理得到的目标函数$⟨x_{up}⟩$。当优化陷入局部最小化点时，我们会通过向记录的全局最小化点添加一个从$N(0, I)$中采样的随机高斯噪声来生成新的初始条件，并重新启动优化过程。

这种方法不仅提高了优化效率，还增强了设计优化的灵活性和准确性。通过快速迭代和精确梯度计算，我们能够更快地找到满足特定性能要求的设计参数，为导管设计等复杂工程问题提供了有力的解决方案。

我们提出的随机化BFGS算法能够确保记录的全局最小化值单调递减。优化损失轨迹如图S1所示，损失值从$L = 6.68 × 10^5$。基于人工智能的优化方法大约经过1500次迭代后达到了最优设计。整个流程，从数据生成（在50个GPU上并行对1000个实例进行，每个实例耗时30分钟，总共10小时）到模型训练（在1个GPU上耗时20分钟），再到设计优化（在1个GPU上耗时15s）以及最终的验证（在1个GPU上耗时10分钟），总共耗时不到1天。

在给定的参数约束条件下，目标函数⟨$x_{up}$⟩关于这些设计变量既不是凸函数也不是单调函数，但通常随着h的增大、d的减小和s的增大而减小（见图S3）。最终优化得到的设计参数为：$d = 62.26 μm，h = 30.0 μm，s = −19.56 μm$，以及$L = 15.27 μm$。

![S3. 优化细节](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheterS3.png)

**图S3. 优化细节**

- A. Geo-FNO的训练和测试误差：Geo-FNO（可能是某种基于几何特征的神经网络优化器）的训练和测试误差均收敛，且没有出现过拟合现象。这表明模型在训练集上学习得很好，同时也在测试集上保持了良好的泛化能力。

- B. 由随机化BFGS算法加速的Geo-FNO代理模型得到的优化损失：使用Geo-FNO代理模型加速的随机化BFGS算法（一种优化算法）在大约1500次迭代后达到了记录的全局最小损失。这显示了算法在寻找最优解方面的效率和准确性。

- C. Geo-FNO在$ℎ-𝑑$ 横截面优化设计周围的损失景观可视化

- D. Geo-FNO在$L-s$ 横截面优化设计周围的损失景观可视化

- E. Geo-FNO在$d-s$ 横截面优化设计周围的损失景观可视化

### 2.8 细菌菌株、培养条件、材料和化学品

对于3D导管长期实验，我们使用了具有卡那霉素抗性的野生型$BW25113$大肠杆菌；而对于微流控实验，则使用了表达mScarlet红色荧光蛋白并具有卡那霉素抗性的$BW25113$大肠杆菌。

从新鲜划线平板上挑取一个目标细菌的单菌落，并将其悬浮在LB培养基中，以制备细菌接种物。起始培养物在$37°C$的LB培养基中过夜培养，直至达到最终浓度约为$OD_{600}（600nm处的光密度）= 0.4$。

对于微流控实验，将300微升的起始培养物转移到含有100毫升LB培养基的新烧瓶中，并在$16°C$下培养，直至OD600达到0.1至0.2。随后，通过离心（2300g，15分钟）两次清洗细菌，并将其悬浮在由$10mM$ potassium phosphate（$pH=7.0$）、$0.1mM$ K-EDTA、$34mM$ K-acetate、$20mM$ sodium lactate和$0.005\%$ polyvinylpyrrolidone(34) 组成的运动成像培养基中。这种培养基可以保持细菌的运动能力，同时抑制细胞分裂。储液池中细菌的最终浓度为$OD_{600} = 0.02$。

对于3D导管长期实验，将3毫升的起始培养物转移到含有500毫升LB培养基的新烧瓶中，并在$16°C$下培养，直至$OD_{600}$达到0.4。然后直接使用这些细菌，并将其注入到细菌储液池中。

在所有培养基和LB平板中都添加了卡那霉素。在实验开始前10分钟，使用荧光显微镜（对于$BW25113$使用差分干涉对比（DIC）光学系统观察，对于$BW25113$ mScarlet菌株则使用红色通道落射荧光观察）检查细菌的运动能力。

### 2.9 微流控实验

为了展示设计的机制和测试优化结构的有效性，研究者们制作了准二维（quasi-2D）微流控通道，以便在显微镜下观察细菌的运动。这些微流控设备是通过光刻和聚二甲基硅氧烷（polydimethylsiloxane，PDMS）软光刻技术制作的。如图3A所示，微流控通道的一端连接着装有成像溶液的注射器，另一端连接着装有E. coli（大肠杆菌）的储液池。通过调整注射器相对于下游出口的高度来控制流速。为了实时监测流速，向成像溶液中注入了荧光珠作为被动示踪剂。

实验使用了奥林巴斯BX51WI显微镜，并配备了两个Photometrics Prime95B相机，通过Hamamatsu的W-View Gemini-2光学分配器连接。使用了奥林巴斯20×干物镜。以12.4帧/s的速度获取时间延迟图像，$488nm$激光强度设置为$20%$。显微镜的焦平面固定在通道深度$z$方向的中部，以避免记录到在通道顶部和底部爬行的细菌。

实验在三天内进行，每天使用独立的E. coli培养批次，每天进行五次15分钟的记录。使用ImageJ软件（Fiji）进行视频后处理，以提取细菌的轨迹。通过轨迹的前向进展线性度进行过滤，以消除快速向下游移动的轨迹，并直观地突出向上游游动的轨迹。研究者估计向上游游动的时间间隔为$10s$，然后细菌会脱落。最大流速定义为沿通道中心线的最高流速。通过计算向上游脱落间隔期间，细菌和荧光珠沿中心线的最快速度的平均值，来估计瞬时最大流速。在补充材料中提供了几个视频记录。

<video width="640"  controls>
    <source src="https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/adj1741_Movie_S1.mp4" type="video/mp4">
</video>

<video width="640"  controls>
    <source src="https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/adj1741_Movie_S2.mp4" type="video/mp4">
</video>

**视频S1-S2. 不同流动条件下细菌从壁面脱落的记录**

<video width="640"  controls>
    <source src="https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/adj1741_Movie_S3.mp4" type="video/mp4">
</video>

**视频S3. 实时的优化设计**

### 2.10 3D导管长期实验

使用Connex-Triplex 3D打印机打印了原型导管管（包括几何设计款和光滑款）。设计有障碍物的管内部结构与准二维结构相似，但进行了放大，并围绕通道的中心线旋转，使得障碍物成为内壁上的挤出环。考虑到可用的3D打印精度和典型导管的尺寸，这些原型的内径为1.6cm。对于设计有障碍物的管，挤出环之间的间距为1mm。为了便于清除3D打印产生的支撑材料，每根管被打印成两半，长边呈榫头形状，在去除支撑材料后组装成完整的管。

如图4A所示，管的上端连接到一个由机械泵控制的注射器，以保持恒定的流速。管的下端连接到一个直径为80mm的培养皿，作为E. coli（大肠杆菌）的储液池。1小时后，将管切成$2cm$长的段，并将每段内的液体转移到培养板上，同时丢弃最上游和最下游的段。在室温下培养培养板24小时后，计数每个培养板上的细菌菌落数量，以反映管相应部分的污染量。

为了计数菌落数量，在培养板上选择了四个圆形、等距、直径为8mm的区域（见图S5）。通过计算这四个区域内菌落的总数，并乘以整个培养板面积与这四个区域面积的比例（即25倍），来估算整个培养板上的菌落总数。当培养板上的菌落过多，变得过于拥挤或重叠以至于无法精确计数时，我们将整个培养板上的菌落总数记为30,000。

### 2.11 讨论

在本研究中，我们介绍了一种医用导管内表面的有效几何设计，旨在抑制细菌的逆流游动和过度污染。我们的设计思路是基于阻碍细菌逆流游动的物理机制，同时考虑了具有幂律动力学的球形粒子流变导向的一般模型。由于传染性微生物在形状、鞭毛特征和流体动力学相互作用方面存在差异，为简化设计和提高设计的通用性，本研究采用的简化模型忽略了细菌运动的细节，如鞭毛的螺旋性（29）和与边界的流体动力学相互作用（20）。模拟结果用于指导实验设计，而非特定预测大肠杆菌的实验结果。未来研究可采用更复杂的模型，考虑特定微生物种类的细节。

我们发现，由于涡旋重叠的相互作用，在障碍物尖端附近会产生有效的涡量，为此我们确定了障碍物之间间距的下限（图S2和补充材料）。障碍物高度的限制是在增强有效涡量和避免管道堵塞之间做出的权衡（图S2）。虽然我们选择使用这种人工智能框架来优化导管的几何形状，但也可以采用其他方法，如结合数值求解器的遗传算法（70）或结合伴随方法的梯度下降法（71）。

我们注意到，几何设计无法完全消除细菌的逆流游动，尤其是在流速接近零的情况下。然而，它能显著减少过度污染的量，并可能大幅延长导管的留置时间。使用我们设计的导管预计不需要改变常规临床方案或重新培训医务人员。此外，我们的解决方案不会在导管中引入化学物质，因此是安全的，也不需要额外的维护。我们预计，这种几何设计方法将与其他程序措施、抗菌表面改性和环境控制方法相兼容。

## 3. 问题求解

论文采用几何聚焦傅里叶神经算子（Geo-FNO）构建AI模型。该模型能够学习并解决与几何形状相关的随机偏微分方程（SPDE），从而实现对导管几何形状的优化，并通过微流体实验和3D打印技术，制作具有不同几何形状的导管原型，并测试其抑制细菌上游游泳的效果。
接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。为了快速理解 PaddleScience，接下来仅对模型构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API 文档](../api/arch.md)。

### 3.1 数据集介绍

数据文件说明如下：

|      `./data.zip/training/`       |                    |      `./data.zip/test/`       |                   |
| :-------------------------------: | :----------------: | :---------------------------: | :---------------: |
|              文件名               |        说明        |            文件名             |       说明        |
| training/x_1d_structured_mesh.npy | 形状为(2001, 3003) | test/x_1d_structured_mesh.npy | 形状为(2001, 300) |
| training/y_1d_structured_mesh.npy | 形状为(2001, 3003) | test/y_1d_structured_mesh.npy | 形状为(2001, 300) |
|      training/data_info.npy       |  形状为(7, 3003)   |      test/data_info.npy       |  形状为(7, 300)   |
|   training/density_1d_data.npy    | 形状为(2001, 3003) |   test/density_1d_data.npy    | 形状为(2001, 300) |

在加载数据之后，需要将 x、y 进行合并，同时对于合并后的训练数据重新 `reshape` 为 `(1000, 2001, 2)` 的格式，具体代码如下

```py
--8<--
examples/catheter/catheter.py:31:75
--8<--
```

### 3.2 GeoFNO 模型

GeoFNO 是一种基于 **几何聚焦傅里叶神经算子 (Geo-FNO** ) 的机器学习模型，它将几何形状转换到傅里叶空间，从而更好地捕捉形状的特征，并利用傅里叶变换的可逆性，可以将结果转换回物理空间。

在论文中，该模型能够学习并解决与几何形状相关的偏微分方程（SPDE），从而实现对导管几何形状的优化, 代码表示如下

```py
--8<--
ppsci/arch/geofno.py:95:205
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("input",)`，输出变量名是 `("output",)`，这些命名与后续代码保持一致。

接着通过指定 FNO1d 的层数、特征通道数，神经元个数，并通过加载上文所提及的初始化权重模型，我们就实例化出了一个神经网络模型 `model`。

### 3.3 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

```python
--8<--
examples/catheter/catheter.py:162:177
--8<--
```

## 4. 结果展示

=== "训练、推理loss"

下方展示了训练后模型对测试数据的第一次预测结果以及最后一次预测结果。

=== "第一次预测结果"

    ![1725427977357](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter10.png)

=== "最后一次预测结果"

    ![1725428017615](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter9.png)

=== "训练测试损失"

    ![1725894134717](https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/catheter8.png)

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
