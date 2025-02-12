# DrivAerNet

DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction(DrivAerNet：一个用于数据驱动空气动力学设计和基于图的阻力预测的参数化汽车数据集)

## 论文信息

|年份 | 期刊 | 作者|引用数 | 论文PDF |
|-----|-----|-----|---|-----|
|2024|Design Automation Conference|Mohamed Elrefaie, Angela Dai, Faez Ahmed|3|DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction|

## 代码信息

|预训练模型 |神经网络|指标|
|:-------:|:-------:|:-:|
|[CdPrediction_DrivAerNet_r2_100epochs_5k_best_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/DrivAerNet/CdPrediction_DrivAerNet_r2_100epochs_5k_pretrained.pdparams)|RegDGCNN|    $R^2:87.5%$ |

=== "模型训练命令"

    ``` sh
    python drivaernet.py
    ```

=== "模型评估命令"

    ``` sh
    python drivaernet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/DrivAerNet/CdPrediction_DrivAerNet_r2_100epochs_5k_pretrained.pdparams
    ```

## 1. 背景简介

本研究引入了DrivAerNet，一个大规模高保真的3D工业标准汽车外形CFD数据集，和RegDGCNN，一个动态图卷积神经网络模型，两者都旨在通过机器学习进行汽车气动设计。DrivAerNet拥有4000个详细的3D汽车网格，使用50万个表面网格面和全面的空气动力学性能数据，包括完整的3D压力，速度场和壁面剪切应力，解决了在工程应用中训练深度学习模型的广泛数据集的关键需求。它比之前可用的最大的公共汽车数据集大60%，并且是唯一一个也可以模拟车轮和底盘的开源数据集。RegDGCNN利用这个大型数据集直接从3D网格中提供高精度的阻力估计，绕过了传统的限制，如需要2D图像渲染或签名距离域（SDF）。RegDGCNN通过在几秒钟内实现快速阻力估计，促进了快速的空气动力学评估，为将数据驱动方法集成到汽车设计中提供了实质性的飞跃。DrivAerNet和RegDGCNN共同加速汽车设计过程，并为开发更高效的汽车做出贡献。为了为该领域未来的创新奠定基础。

通过先进的空气动力学设计降低油耗和CO2排放对汽车行业至关重要。这有助于更快地向电动汽车转型，补充2035年对内燃机汽车的禁令，并与2050年实现碳中和的宏伟目标保持一致，以对抗全球变暖。在气动设计中，通过复杂的设计选择进行导航涉及到对气动性能和设计约束的详细检查，这往往因高保真CFD模拟和实验风洞试验的耗时性而减慢。高保真的CFD模拟可以在每个设计中花费几天到几周的时间，而风洞试验，尽管其准确性，但由于时间和成本的限制，仅限于检查少数设计。数据驱动的方法可以通过利用现有的数据集来导航通过设计和性能空间来缓解这一瓶颈，从而加速设计探索过程并有效地评估气动设计。

尽管基于数据驱动的气动设计方法最近取得了令人鼓舞的进展，但这些方法通常集中于较简单的二维案例或较低保真的CFD模拟，而忽略了真实世界三维设计中固有的复杂性和高保真CFD模拟带来的挑战。通过排除车轮和反射镜等部件来简化汽车设计，而不对下车体进行建模，这导致了对气动阻力的严重低估。考虑这些因素使阻力增加了1.4倍以上，突出了详细建模对于精确气动分析的重要性。此外，缺乏公开可用的高保真汽车仿真数据集，这可能会减缓数据方面的研究。

针对这一挑战，本文介绍了DrivAerNet，这是一个包含4000个高保真汽车CFD模拟的完整三维流场信息的综合数据集。它已公开发布，可作为在气动评估、生成设计和其他机器学习应用中训练深度学习模型的基准。

为了说明大规模数据集的重要性，本研究还开发了基于动态图卷积神经网络的气动阻力预测代理模型。模型RegDGCNN直接在非常大的3D网格上运行，不需要进行2D图像绘制或生成符号距离场( Signed Distance Fields，SDF )。RegDGCNN快速识别空气动力学改进的能力为通过简化设计调整的评估来创造更高效的车辆开辟了新的途径。这标志着对汽车设计进行更有效的优化迈出了重要的一步。

**总的来说，本研究的贡献是：**

- 发布了DrivAerNet，一个包含4000个汽车设计的广泛的高保真数据集，完整的具有详细的三维模型，每个模型有50万个表面网格，完整的三维流场和气动性能系数。该数据集比先前可获得的最大的汽车公开数据集大60 %，并且是唯一的也可以对车轮和车身进行建模的开源数据集，允许对阻力进行准确的估计。

- 引入了一种基于动态图卷积神经网络的代理模型，命名为RegDGCNN，用于气动阻力的预测。在ShapeNet基准数据集上，RegDGCNN在使用1000 ×更少参数的情况下，拖拽预测性能比目前最先进的基于注意力机制的模型提升了3.57 %，取得了较好的效果。

此外，本研究的数据集规模较大，这表明将训练数据集从DrivAerNet中的560个汽车设计扩展到2800个汽车设计后，误差降低了75 %，说明了数据集规模与模型性能之间的直接相关性。进一步验证了本研究模型的有效性和大型数据集在代理模型建模中的内在价值。

**先前研究的局限性：**

尽管在先前研究中采用了新的方法，但它们面临着源于ShapeNet数据集固有缺陷的限制，例如较低的网格分辨率，数据集尺寸小，以及过度简化，如将汽车建模为单身实体，而没有详细考虑车轮、下车体和侧镜等部件，这可能会显著影响真实世界的空气动力学性能。这种过度简化会显著影响真实世界的气动性能；在DrivAer模型快背模型中包括这些细节，在CFD模拟中，阻力值从0.115增加到0.278，在风洞实验中，阻力值从0.125增加到0.275。这些增加分别代表阻力的大幅增加约142 %和120 %，强调了综合建模在实现准确的气动评估中的关键作用。代理模型和设计优化中的另一个共同障碍是数据的稀缺性，这使得复制结果或基准测试各种模型和方法的工作复杂化。为了应对这一挑战，本研究的贡献引入了DrivAerNet，这是一个为数据驱动的气动设计量身定制的综合基准数据集，旨在促进未来方法的比较和验证。

## 2. 问题定义

数据下载：
    ``` sh
    wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar
    tar -xvf data.tar
    ```

![fig1](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig1.jpg)

图1：参数化的DrivAer模型使用变形盒子在ANSA 软件中进行几何变换，总共使用了50个几何参数和32个可变形实体。对变形框进行颜色编码，突出易受参数修改影响的区域，便于创建" DrivAerNet "数据集。利用这种变形技术，本研究生成了4000个独特的汽车设计。

**DrivAer模型Net数据集和模型背景介绍：**DrivAer模型是由慕尼黑工业大学( TUM )的研究人员开发的一种行之有效的传统汽车参考模型。它是宝马3系和奥迪A4汽车设计的结合，以代表大多数传统汽车。DrivAer模型是为了弥补Ahmed和SAE等机构的开源过度简化模型与制造公司的复杂设计之间的差距而开发的，而这些模型并不公开。为了准确地评估真实世界的气动设计，本研究选择了具有详细的下车体、车轮和反射镜( FDwWwM )的快背构型作为本研究的基准模型，如图2a所示。FDwWwM模型的这种选择是由车轮、反射镜和车身底部几何形状对气动阻力的巨大影响所驱动的，这一结论得到了文献[ 17 ]的研究结果的支持。具体来说，详细的底部几何结构增加了32 ~ 34个计数，镜面的加入增加了14 ~ 16个计数，车轮的存在使总阻力系数增加了102个计数。

![fig2](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig2.jpg)

图2：DrivAer模型Fastback模型，包含详细的特征，并附有计算网格，说明了网格细化区域和附加层，以准确模拟空气动力学现象。

**基线参数化模型的选择：**为了创建用于训练代理模型和设计优化的深度学习模型的全面数据集，本研究首先创建了DrivAer模型的参数化模型。这种方法是由原始模型的局限性所决定的，原始模型提供了一个单一的、非参数的. stl文件。为了充分捕获与实际汽车设计挑战相关的几何变化和设计修改，本研究使用商业软件ANSA；开发了由50个几何参数定义的版本DrivAer模型，包括32个可变形实体(见图1)。该参数化模型允许对设计空间进行更详细的探索，通过应用最优拉丁超立方采样方法，特别是使用如[ 12 ]所概述的增强随机进化算法( ESE )，促进了4000个独特设计变体的生成。

**生成多样化汽车设计的技术：**参数化模型，以及在实验设计( Design of Experiment，DoE )过程中应用的约束和边界，极大地丰富了数据集，使其成为开发和训练用于代理建模和设计优化任务的高级深度学习模型的坚实基础。本研究提供了包含变形特征的参数化模型的访问，以便进一步参考和利用。与[ 19 ]中的方法不同，本研究实现了更广泛的变形技术，使本研究能够探索更多样化的汽车设计。该方法旨在增强深度学习模型的适应性，使其能够泛化到各种汽车设计中，而不是局限于单个设计中的微小几何修改。图3描述了网格质量的变化，从粗到高分辨率的不同数据集。与[ 22 ]和[ 36 ]的研究相比，本研究的原始网格具有540k的网格面，提供了更稠密和更详细的表示，从而揭示了更详细的几何和设计特征。此外，图4给出了来自DrivAerNet数据集的汽车外形图谱，说明了设计尺寸和特征的可变性。这个范围从最大到最小的体积模型强调了数据集覆盖全面的空气动力学剖面的能力。

![fig3](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig3.jpg)

图3：不同数据集的网格分辨率比较；第一行特征是Li et al ( 2023 ) [ 22 ]的Ahmed体网格，显示了粗略的分辨率。第二行显示来自ShapeNet数据集的中等分辨率网格，正如Song等人( 2023 ) [ 36 ]所使用的。最后一行展示了我们的高分辨率网格，为深入的气动设计提供了更多的细节。

![fig4](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig4.jpg)

图4：来自DrivAerNet数据集的汽车模型，说明了一系列的空气动力学设计。最左边的模型代表了数据集中的最大体积，而最右边的模型代表了体积最小的设计，突出了所研究的气动外形的多样性和范围。

#### 2.1 数值模拟

##### 2.1.1 域及边界条件

选取缩比为1：1的DrivAer模型快背式模型进行CFD数值模拟。模拟是使用开源软件OpenFOAM &进行的，OpenFOAM是一个全面的C + +模块集合，用于在CFD研究中裁剪求解器和实用程序。在这项研究中，压力和速度之间的耦合是通过SIMPLE算法(压力关联方程的半隐式方法)来实现的，该算法是在SimpleFoam求解器中实现的，它是为稳态、湍流和不可压缩流动模拟而设计的。基于Menter ' s公式[ 24 ]的k - ω - SST模型，由于其能够克服标准k - ω模型的局限性，特别是其对自由流k和ω值的依赖性，以及其在预测流动分离方面的有效性，被选作Reynolds - Averaged Navier-Stokes ( RANS )模拟的湍流模型。

以车长为特征长度尺度，在流速($u_\infty$)为30 m / s，对应雷诺数约为9.39 × 106的条件下进行模拟。计算网格使用SnappyHexMesh ( SHM )工具构建，具有四个不同的细化区域。在车体周围添加了额外的层，以精确地表示尾流动力学和边界层演变(见图2b)。边界条件在入口定义为均匀速度，在出口定义为基于压力的条件。为避免回流进入模拟域，将出口的速度边界条件配置为入口出口条件。汽车表面和地面被赋予无滑移条件，而车轮被模拟为旋转的WallVelocity边界条件。滑移条件被应用于区域的外侧和顶部边界。

近壁区的黏度效应采用nutUSpaldingWallFunction壁面函数法处理。黏性项选取的壁面函数采用基于速度的近壁面连续湍流黏性剖面，采用文献[ 37 ]提出的方法。对于散度项，采用默认的高斯线性格式，速度对流项采用有界的高斯线性UpwindV格式离散，并施加速度梯度，保证二阶精度。梯度计算采用高斯线性方法，并辅以多维限制器，以增强解的稳定性。关注的物理量包括三维速度场、表面压力、壁面剪应力以及气动力系数。

##### 2.1.2数值结果的验证

DrivAer模型快退模型的选择是由计算和实验参考的可用性来证明的，这使得本研究能够将本研究的结果与既定的数据[ 17、43 ]进行比较。在开始模拟之前，本研究对网格细化对结果的影响进行了初步评估。这涉及将三种不同网格分辨率下得到的阻力系数与实验值和参考模拟进行比较，详见表2。目的是在模拟精度和计算效率之间找到一个最佳的平衡。这种平衡是至关重要的，因为本研究的目标是生成一个用于训练深度学习模型的大规模数据集，这需要仿真结果的高保真度和可管理的磁盘存储和仿真时间，以适应广泛的计算需求。阻力系数$C_d$由方程确定：

$$
C_d = \frac{F_d}{\frac{1}{2} \rho u_\infty^2 A_{\mathrm{ref}}}
$$

物体所受的阻力$F_d$是其有效迎风面积$A_{ref}$、来流速度$u_\infty$和空气密度$\rho$的函数。该力由压力和摩擦力两部分组成。

评估不仅包括阻力系数，还包括网格尺寸和所需的计算资源。仿真在装有AMD EPYC 7763 64 - Core处理器的机器上进行，共256个CPU核，4个Nvidia A100 80GB GPU。

本研究的分析揭示了本研究的模拟与基准实验数据和参考模拟之间的一致相关性，其中800万和1600万细胞网格显示出特别好的一致性。在4000万单元格网格中观察到的差异可能源于网格粒度的差异，因为参考模拟使用了1600万单元格网格。更精细的网格捕捉到更复杂的流动动力学，这可能在更粗的网格中无法表示，从而导致发散。在计算流体力学中，特别是在设计的初步阶段，高达5 %的误差裕度通常被认为是工程上可以接受的。此外，在DrivAerNet数据集中对所有4000个设计实现" RANS fine "需要约120TB的存储，由于巨大的存储需求，对数据共享和可重复性提出了重大挑战。因此，考虑到精度和计算资源分配之间的平衡，本研究决定使用800万和1600万单元格网格进行模拟。这些构型在计算效率和精确气动分析所需的细节水平之间提供了折衷。

**利用多保真数据和迁移学习进行高效的Surrogate模型开发：**如文献[ 15 ]所示，利用多保真CFD模拟被证明是一种稳健的精确三维流场估计策略。该方法包括使用一个数据集，该数据集结合了RANS，相对更容易和更便宜，可以捕获一般的流动行为，以及直接数值模拟( DNS )数据，尽管计算费用昂贵，但其详细的流动信息是众所周知的。使用这种多样化的数据集训练深度学习模型，不仅可以使模型有效地推广到真实世界场景，如风洞试验所证实的那样，而且还可以简化两个阶段的训练过程。这个过程从中等精度的RANS数据开始，以掌握一般的流型，然后过渡到用高精度的DNS数据进行微调，从而增强模型的精度和真实世界的适用性。[ 33、35 ]也展示了利用多保真数据集训练代理模型的类似结果，突出了该方法在空气动力学分析中的有效性。DrivAerNet数据集可以类似地使用，允许与低保真度或高保真度的数据集集成，以增强模型训练并提高预测能力。

##### 2.1.3 CFD模拟结果

**包括多样化的汽车外形尺寸和复杂的流动动力学：**与[ 36 ]的方法不同，所有的汽车模型都标准化为3.5米的统一长度，以适应预定义的计算域，本研究的数据集允许汽车尺寸的多样性，调整网格，边界框和每个设计的附加层。这种灵活性对于捕捉汽车周围复杂的流动动力学，包括流动分离、再附和回流区等现象，以及确保精确的气动力系数估计至关重要。这种方法解决了一些研究中观察到的数据集大小优先于模拟保真度的局限性，往往忽略了收敛、精确建模和适当的边界条件对于复杂三维模型的重要性。

**车轮、侧反射镜和底盘的建模：**正如之前所强调的，大多数文献和可用的数据集往往忽略了车轮、侧镜和下半身的建模，如表1所示。相比之下，本研究的方法包括对这些组件的详细建模。图5说明了汽车上的速度分布：在这里，由于无滑移边界条件，车身显示零速度，而车轮显示非零速度。此外，该图可视化了汽车周围的流线，为包括这些特征的影响的流动动力学提供了见解。DrivAerNet数据集具有完整的三维流场信息，如图6a中的速度数据所示，此外，它还提供了汽车表面的压力分布。压力系数$C_p$由压差$p - p_\infty$与动压的比值$\frac{1}{2}\rho u^2$计算，具体表达式为：

$$
C_p=\frac{p-p_\infty}{\frac{1}{2}\rho u^2}
$$

$C_p$在汽车表面的分布如图6b所示。

![table1](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/table1.jpg)

表1：对各种空气动力学数据集进行了比较分析，重点介绍了数据集中的设计数量(大小)、气动力系数(阻力系数$C_d$和升力系数$C_l$)的包含、速度( $u$ )和压力( $p$ )场的包含、车轮/车体建模的存在、进行参数研究的能力、设计参数的数量和开源可用性等关键方面。

#### 2.2 几何可行性

![fig6](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig6.jpg)

图6：DrivAerNet数据集包括详细的速度、压力和壁面剪切应力的三维场，以及气动力系数，以及每个入口的车身和前后轮的详细三维网格。

**在自动设计变形中确保几何完整性：**在本研究通过ANSA自动变形生成大量不同的汽车标志的方法中，确保每个变体的几何质量和可行性至关重要。为了解决由变形操作引起的潜在问题，如非水密的几何形状、表面交叉点或内部孔洞，本研究采用了自动化的网格质量评估和修复过程。该过程不仅识别并纠正了常见的几何异常，确保了本研究的数据集中只包含模拟准备好的模型。不满足这些标准的几何被系统地排除在模拟之外。DrivAerNet数据集使用了一系列不同的参数(共50个参数)来变形汽车几何结构，包括修改侧镜放置位置、消声器位置、挡风玻璃、后窗长度/倾角、发动机下盖尺寸、门和翼子板的偏移量、发动机罩位置和前照灯的尺度，以及对整个汽车长度和宽度的修改。此外，对汽车的上、下车体缩放，以及坡道、扩散器和行李箱盖角等关键角度进行了调整，这对于探索不同设计变更对汽车空气动力学的影响至关重要。为了详细说明变形参数，包括它们的下界和上界，请参考本研究的GitHub存储库。

当本研究对整个汽车的几何形状进行变形时，在变形过程中，车轮定位在x、y、z轴上进行调整。对于所有的仿真，本研究使用相同形状的前轮和后轮。为了精确模拟车轮旋转，本研究将其导出为单独的. stl文件后变形，这允许本研究应用旋转的WallVelocity物理边界条件。此外，变形会影响汽车的垂直定位，需要计算z轴位移，以确保车身和车轮与地平面正确对齐。为了模拟目的，本研究提供了3个不同的. stl：一个用于车身，一个用于前轮，一个用于后轮，以准确地模拟它们的相互作用。

#### 2.3 Drivaernet数据集特征

在仿真中，本研究使用Open FOAM®版本11，在128个CPU核和4个Nvidia A100 80GB GPU上执行计算任务。这导致了大约352，000个CPU小时的总计计算成本。本研究提供了完整的数据集，包括原始CFD输出和衍生的后处理数据集。

本研究的数据集作为评估深度学习模型的基准，旨在促进有效的模型测试。为了管理CFD模拟的大量数据，本研究采用了一种聚焦于流场关键区域的数据缩减策略。这涉及从汽车前方和后方的区域中保留数据，定义在特定的边界框内，这有助于显著降低整体数据规模。此外，本研究提供了一个脚本，将CFD模拟数据转换为适合训练深度学习模型的格式。鉴于ParaView和VisIt等数据可视化工具的广泛使用，这些工具依赖于可视化工具包( Visualization Toolkit，vtk )，因此本研究的数据集以vtk格式提供。这确保了数据在这些常见的可视化环境中易于访问和使用，支持广泛的研究和应用需求。

DrivAerNet数据集提供了一套全面的与汽车几何结构相关的空气动力学数据，包括总力矩系数$C_m$、总阻力系数$C_d$、总升力系数$C_l$、前方升力系数$C_{l,f}$，和后方升力系数$C_{l,r}$等关键指标。包括在数据集中的重要参数，如流体剪切力和$y^+$度量，用于网格质量评估的积分。此外，该数据集提供了沿x和y轴方向的流动轨迹和详细的压力和速度场的横截面分析，丰富了对气动相互作用的理解。

该数据集包括：

- 综合CFD模拟数据~16TB

- CFD模拟的固化版本~1TB

- 4000辆汽车设计的3D网格和相应的气动性能系数( $C_d、C_l、C_{l,r}、C_{l,f}、C_m$)~84GB

- 2D切片包括汽车x方向的尾流和y方向的对称面~12GB。

**Drivaernet中汽车设计之间的气动性能变异性：**图7展示了DrivAerNet数据集中阻力系数($C_d$ )和各种升力系数( $C_l、C_{l,r}、C_{l,f}$)之间关系的三个散点图。数据被划分为训练集、验证集和测试集，其中70 %用于训练，15 %用于验证和测试。这样的划分对于模型训练过程的完整性和后续的性能评估至关重要。

![fig7](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig7.jpg)

图7：散点图显示了DrivAerNet数据集的阻力系数( $C_d$)和升力系数($C_l、C_{l,r}、C_{l,f}$)之间的关系。该数据集代表了利用增强型随机进化算法( Enhanced Stochastic Evolution Algorithm，ESE )通过最优拉丁超立方抽样方法生成的独特设计变体。数据点分为训练集、验证集和测试集( 70 %、15 %、15 %)。

图8所示的核密度估计( KDE )图比较了两个气动数据集的阻力系数分布。在这里，本研究比较了文献[ 36 ]中的数据集，该数据集跨越了广泛的阻力值，反映了ShapeNet中各种各样的汽车设计。相比之下，本研究的DrivAerNet数据集针对传统的汽车设计，考虑了更详细的几何修改。这一关注点在工程设计过程中尤其相关，因为在工程设计过程中，最初的汽车设计通常是通过增量变化来优化气动性能。因此，DrivAerNet数据集提供了更具体的检查细微的设计调整及其对气动性能的影响。

![fig8](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig8.jpg)

图8：两个气动数据集阻力系数的比较核密度估计( KDE )和小提琴图。蓝色曲线代表来自Song等人2023 [ 36 ]的数据集，橙色曲线对应DrivAerNet数据集。DrivAerNet专注于传统的汽车设计，强调微小的几何修改对气动效率的影响。

在图9中，本研究给出了不同设计下的气动性能。左上方说明了阻力系数$C_d$最低的设计。相反，右上角显示了$C_d$最高的设计，识别了气动优化的机会。左下方的设计升力系数$C_l$ (表示最大下压力)最低，有利于高速时的稳定性，而右下方的设计升力系数$C_l$最高，可能使气动稳定性复杂化。

![fig9](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig9.jpg)

图9：来自DrivAerNet的汽车设计的气动性能显示了一系列系数。左上：阻力系数$C_d$最小的设计，表明气动效率最优。右上角：$C_d$最大的设计。左下：设计最小升力系数$C_l$  (最大下压力)。右下方：采用最大$C_l$ 设计。

## 3. 问题求解

### 3.1 用于回归的动态图卷积神经网络

正如[ 1、20、26、29、30、32、34]的研究表明，几何深度学习在解决涉及不规则几何体的流体动力学挑战方面具有重要的前景。

![fig10](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig10.jpg)

图10：用于气动阻力预测的RegDGCNN的结构。模型通过将三维网格转换为点云表示的方式进行处理。它取n个输入点，在一个EdgeConv层为每个点计算一个大小为k的边缘特征集，并对每个集合内的特征进行聚合，计算对应点的EdgeConv响应。最后一个EdgeConv层的输出特征进行全局聚合，形成1D全局描述符，然后用于预测气动阻力系数$C_d$，从而可以直接从物体的3D几何结构中学习。Edge Conv块接受一个维度为$n × f$的输入张量，利用多层感知器( MLP )确定每个点的边缘特征。在MLP应用后，通过对相邻的边缘特征进行池化操作，块输出一个维度为$n × a_n$的张量。

在这项工作中，本研究扩展了动态图卷积神经网络( DGCNN )框架[ 42 ]，传统上与点网络[ 27 ]和图CNN方法相关联，以解决回归任务，这标志着其在分类中的常规应用有重大偏离。本研究的贡献在于采用DGCNN架构来预测连续值，特别关注在流体动力学和工程设计中至关重要的气动力系数。利用点网络的空间编码能力和图CNNs提供的关系推理，本研究提出的RegDGCNN模型(如图10所示)旨在捕获物体周围流体流动的复杂相互作用，为关键气动参数的精确估计提供了一种新方法。该方法利用局部几何结构，通过构建局部邻域图，并在相邻点的边连接对上应用类似卷积的操作，与图神经网络原理一致。被称为边缘卷积( EdgeConv )的技术显示出桥接平移不变性和非局部性的特性。与标准图CNN不同的是，RegDGCNN的图不是静态的，而是在每一层网络之后动态更新的，这使得图结构能够适应不断变化的特征空间。

首先初始化具有节点特征X的图G，以及Edge Conv层参数θ和全连接层参数φ。RegDGCNN的一个显著特点是其在每个EdgeConv层中的动态图构建，其中每个节点的k近邻基于特征空间中的欧氏距离进行识别，从而自适应地更新图的连通性，以反映最重要的局部结构。EdgeConv操作定义为：

$$
h_{ij}=\Theta\left(x_i,x_j-x_i\right)
$$

通过使用共享的多层感知器( Multi-Layer感知器，MLP )来聚合来自这些邻居的信息来增强节点特征，该方法同时处理了单个节点特征及其与相邻节点的差异，有效地捕获了局部几何上下文。

通过Edge Conv变换，进行全局特征聚合，将所有节点的特征聚合成一个奇异的全局特征向量：

$$
x_i^{\prime}=\max_{j\in\mathcal{N}(i)}h_{ij}
$$

在这里，最大池化被用来封装图的整体信息。该全局特征向量随后通过几个FC层进行处理，其中包括ReLU和dropout等非线性激活函数，以分别引入非线性和防止过拟合。该架构最终形成了一个输出层，旨在适应手头的具体任务，例如对回归任务使用线性激活。

$$
h_{ij}=\mathrm{MLP}\left(
\begin{bmatrix}
x_i,x_j-x_i
\end{bmatrix}\right)
$$

$$
X^{\prime}=\max_{i\in\mathscr{G}}x_i^{\prime}
$$

模型的性能通过使用均方误差( MSE )计算其预测输出与真实拖拽值之间的损失来量化，反向传播算法通过优化算法(如Adam [ 21 ] )调整模型参数θ和φ以最小化该损失。这一迭代精化过程凸显了RegDGCNN从图结构数据中动态利用和整合层次特征的能力。

#### 3.1.1 实现细节

**网络结构：**本研究使用k -最近邻算法为RegDGCNN构建图，k设置为40。该参数对于定义执行卷积操作的局部邻域至关重要。RegDGCNN模型通过特定的参数进行实例化，以适应回归任务的性质。Edge Conv层配置大小为{ 256，512，512，1024 }的通道，后面的MLP层为{ 128，64，32，16 }。最后，网络的嵌入维数设置为512，提供了一个高维空间来捕获手头回归任务所需的复杂特征。本研究的RegDGCNN模型是完全可微的，可以与3D生成式AI应用程序无缝集成，以增强设计优化。

**模型超参数：**采用PyTorch框架进行实验。模型采用32个批次进行训练，每个输入的点数设置为5000。训练分布在4个NVIDIA A100 80GB GPU上，利用数据并行性提高计算效率。网络的学习率最初设置为0.001，并使用学习率调度器来降低验证损失平台化后的速率，具体使用ReduceLROnPlateau调度器，其耐心为10个历元，缩减因子为0.1。该方法通过调整学习率来帮助微调模型，以响应在验证集上的性能。模型共训练了100个epoch，在保证充分学习的同时防止过拟合。对于优化，本研究使用了Adam优化器[ 21 ]，因为它具有自适应学习速率的能力。

**推理时间：**RegDGCNN模型，其紧凑尺寸约为300万参数，存储需求约为10MB，在4 A100 80GB GPU上对具有540k网格面的汽车设计进行阻力估计，在1.2秒内完成，与在128 CPU核上使用4 A100 80GB GPU进行标准CFD模拟所花费的2.3小时相比，效率显著提高。

### 3.2 气动阻力的代理建模

在这一部分中，本研究在两个空气动力学数据集DrivAerNet和ShapeNet上评估了本研究的RegDGCNN模型，强调了较大的训练量对模型性能的影响，并调查了模型学习到的特征。

#### 3.2.1 Drivaernet：高分辨率网格的气动阻力预测

对RegDGCNN在DrivAerNet数据集上的性能进行测试，如图11所示，其预测值与CFD真实数据之间具有较好的相关性，说明了模型的有效性。DrivAerNet数据集的复杂性归因于其包含了行业标准形状，通过50个几何参数变化，在空气动力学预测方面提出了全面的挑战。本研究的模型有效地导航了数据集的复杂性，并直接处理了3D网格数据，这标志着传统方法的重大转变，通常依赖于生成符号距离场( SDF )或渲染2D图像。这种直接的方法使本研究在看不见的测试集上达到了0.9的R2分数，强调了模型准确识别细微气动差异的能力。

![fig11](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig11.jpg)

图11：本研究的RegDGCNN模型预测的阻力系数$C_d$与DrivAerNet未见测试集的真实值的相关性图，取得了0.9的R2分数。点线表示完全相关的直线，代表理想的预测场景。

#### 3.2.2 Shapenet：任意形状车辆的气动阻力预测

为了测试所提出的RegDGCNN模型的可推广性，本研究还在现有的基准数据集上评估了其适应复杂几何形状的能力，使用了来自ShapeNet数据集[ 36 ] (见图12)的2，479种不同的汽车设计，该数据集显示出比本研究的DrivAerNet数据集更广泛的汽车形状。

![fig12](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig12.jpg)

图12：从ShapeNet数据集中选取汽车样本，展示了汽车形状和网格分辨率的多样性，用于评估RegDGCNN的泛化能力。这些样本与本研究在DrivAerNet数据集中发现的高分辨率网格提供了一个比较基准。

在表3中，本研究比较了两个模型的性能：attn - ResNeXt模型来自文献[ 36 ]的研究，该模型实现了自注意力机制，以促进对图像各个区域之间相互作用的理解。它使用2D深度/正常渲染作为输入，具有大约20亿个参数，实现了0.84的$R^2$分数；本研究提出的RegDGCNN模型，直接处理三维网格数据，显著减少了参数数量至300万，并取得了优异的R2评分0.87。这种比较强调了本研究的模型在气动阻力预测任务中的效率和有效性。

![table3](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/table3.jpg)

#### 3.2.3 训练数据集大小的影响

对于ShapeNet4和DrivAerNet两个数据集，本研究首先分配70 %用于训练，15 %用于验证和测试。随后，本研究在训练部分的20 %、40 %、60 %、80 %和100 %的训练子集上进行实验。ShapeNet子集范围为1270 ~ 6352个样本。同时，对于DrivAerNet数据集，对应的样本量分别为560、1120、1680、2240和2800个样本。

图13显示了一个明显的趋势，拖曳系数预测的平均相对误差随着用于训练的数据集百分比的增加而减少。这种趋势对于两个数据集都是一致的，强调了共同的机器学习原理，即更多的训练数据通常会导致更好的模型性能。DrivAerNet Dataset在所有大小的训练数据上的性能提升凸显了更大数据集在空气动力学机器学习模型中的关键作用，并进一步确立了DrivAerNet数据集的价值，其价值显著大于以往的开源数据集。

![fig13](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig13.jpg)

图13：基于训练集大小，本研究的模型RegDGCNN在未知测试集上阻力系数预测的平均相对误差。ShapeNet拖拽数据集[ 36 ]的结果用蓝色表示，而DrivAerNet数据集的结果用橙色表示。训练集大小从20 %到100 %不等。本研究观察到数据集规模的增加会导致显著的误差降低，表明了在气动代理模型建模中使用更大数据集的必要性。

该图还表明，与ShapeNet数据集相比，RegDGCNN在DrivAerNet数据集上获得了更好的性能。这可归因于几个因素：

- Shape Net中形状的巨大变化与足够数量的样本不对应，无法涵盖整个气动阻力值范围；

- Shape Net模型将汽车建模为单体式实体，省略了车轮和车身等关键细节，而这些细节对于准确的气动建模至关重要。

- ShapeNet中的所有阻力值都是使用单一的参考区域计算的，这并不能解释不同汽车设计的正面投影面积的显著变化。

- 在ShapeNet数据集中，网格分辨率有很大的变化，这可能导致气动预测的不一致。

这一分析旨在展示本研究的模型的泛化能力，强调开发有效泛化到域外分布的模型的目标。

#### 3.2.4 特征学习

为了进一步评估模型的性能，本研究分析了边缘卷积操作后中间层学习到的特征。图14说明了从DrivAerNet获取的汽车样本的上采样点云的特征重要性分布，颜色编码从浅黄色(低重要性)到深红色(高重要性)。最初，RegDGCNN对汽车的前部和后部区域进行调零，这对塑造气动性能至关重要。这一关注点对于气动设计具有显著的针对性，因为前部区域对压差阻力有重要影响，后部区域由于在气流分离和尾流区形成中的作用而显得尤为重要。随着模型向更深的层次发展，它开始识别更复杂的几何细节。相反，屋顶和窗户等区域对阻力的影响较小，突出了模型在识别具有更显著气动影响的区域方面的能力。

![fig14](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig14.jpg)

图14：从RegDGCNN导出的DrivAerNet的汽车模型的上采样点云的特征重要性的可视化，特别关注来自EdgeConv层的特征。特征强度由低(浅黄色)到高(深红色)的颜色编码，表明了卷积层显著学习焦点的区域。这种映射突出了学习到的有助于模型预测的特征。

在PDF中，问题求解通常涉及数据预处理、模型设计、训练过程以及评估与优化的多个环节。在此过程中，涉及到如何处理数据集、构建合适的约束条件、选择优化器和评估器等环节。以下是具体的介绍，包括数据集、模型（以RegDGCNN为例）、约束构建、优化器构建、评估器构建、以及模型的训练和评估。

**1.数据增强类：`DataAugmentation：**

用于对点云进行随机变换，包括平移、加噪声和随机丢点，以提升模型的泛化能力。

``` py linenums="46"
--8<--
ppsci/arch/regdgcnn.py:46:112
--8<--
```

**2.数据集类：`DrivAerNetDataset`：**

用于加载 DrivAerNet 数据集，并处理点云数据（如采样、增强和归一化）。

``` py linenums="35"
--8<--
ppsci/data/dataset/drivaernet_dataset.py:35:261
--8<--
```

### 3.3 RegDGCNN 模型

RegDGCNN 是一种专为图形数据设计的深度学习模型，常用于处理 3D 点云、图结构数据等任务。在本问题中，RegDGCNN 用于根据输入的 3D 点云顶点坐标预测相应的空气阻力系数（$C_d$）。模型架构基于图卷积（Graph Convolution），通过动态构建 K 近邻（KNN）图结构，在输入点云中捕获局部和全局几何特征，完成特征提取和表示。局部特征通过图神经网络层（如 EdgeConv）进行聚合，并逐步整合为全局特征，以描述整个 3D 模型的形状和属性，最终实现对目标值的回归预测。

在 DrivAerNet 数据集中，RegDGCNN 模型以固定大小的 3D 点云作为输入，通过以下流程完成空气阻力系数预测任务：

1. **输入**：标准化处理的点云数据（顶点坐标）。
2. **特征学习**：捕获点云的局部和全局几何特征。
3. **输出**：预测的空气阻力系数（$C_d$），作为模型的回归输出。

```python
model = ppsci.arch.RegDGCNN(input_keys=cfg.MODEL.input_keys,
                            label_keys=cfg.MODEL.output_keys,
                            weight_keys=cfg.MODEL.weight_keys,
                            args=cfg.MODEL)
```

模型参数具体如下：

```yaml
MODEL:
  input_keys: ["vertices"]  # 输入数据的关键字（3D顶点数据）
  output_keys: ["cd_value"]  # 输出数据的关键字（空气阻力系数）
  weight_keys: ["weight_keys"]  # 权重数据的关键字（用于加权损失函数等）
  dropout: 0.4  # Dropout比例，用于防止过拟合
  emb_dims: 512  # 嵌入层的维度，控制模型的表示能力
  k: 40  # k近邻数，表示每个节点的邻居数量
  output_channels: 1  # 输出通道数，对于回归任务是1
```

### 3.4 约束构建

#### 3.4.1 监督约束

由于我们以监督学习方式进行训练，此处采用监督约束 `SupervisedConstraint`：

``` py linenums="34"
--8<--
examples/drivaernet/drivaernet.py:34:58
--8<--
```

### 3.5 优化器构建

优化器是模型训练中的关键部分，用于通过梯度下降法（或其他算法）调整模型参数。在本场景中，使用了`Adam`和`SGD`优化器，并通过学习率调度器来动态调整学习率。

``` py linenums="86"
--8<--
examples/drivaernet/drivaernet.py:86:109
--8<--
```

### 3.6 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="60"
--8<--
examples/drivaernet/drivaernet.py:60:81
--8<--
```

评价指标 `metric` 选择 `ppsci.metric.MSE` 即可,也可根据需求自己选择其他评估指标。

### 3.7 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="112"
--8<--
examples/drivaernet/drivaernet.py:112:128
--8<--
```

## 4. 完整代码

=== "drivaernet.py"

``` py linenums="15"
--8<--
examples/drivaernet/drivaernet.py:15:200
--8<--
```

## 5. 结果展示

### 5.1 局限性和未来工作

本部分讨论了本研究研究的局限性。尽管仔细选择以确保细节和计算效率之间的平衡，但模型的参数化面临固有的局限性。这源于表征的紧凑性和捕捉广泛的空气动力学现象所需的灵活性之间的权衡。因此，虽然本研究的方法为许多应用提供了有价值的见解，但它可能无法完全涵盖与汽车工程有关的所有空气动力学变化。该数据集包含4，000个实例，这些实例虽然重要，但可能无法完全捕捉真实汽车设计的广谱性。此外，本研究的关注点主要集中在阻力预测上；但是，本研究计划在未来的工作中扩展RegDGCNN的应用，将表面场预测纳入其中。虽然本研究的数据集规模大且保真度高，但重要的是，本研究仍处于接近人工智能领域(如图像处理和自然语言处理)的规模和基础影响的早期阶段，其中大型数据集是一种常态。

应用基于图的方法(如RegDGCNN )的关键挑战之一是显著的GPU内存需求。这是由于需要计算点与点之间的所有成对距离，这可能是高内存密集型的。此外，点云的不均匀密度引入了额外的复杂度；固定的k近邻方法可能不适用于点密度变化的区域。

RegDGCNN的另一个限制是，在当前形式下，对于大规模点云，RegDGCNN没有减少前向传递过程中的点数，导致高计算需求，并可能限制模型对更大数据集的可扩展性。解决这些挑战对于提升基于图的神经网络处理复杂气动数据的能力和应用至关重要。

### 5.2 结论与飞桨版结果

在本研究的结论中，本研究强调了DrivAerNet的独特优势，它通过关注详细的几何修改，优于更广泛的数据集，如[ 22、31、36 ]中引用的数据集，特别是在真实世界的气动设计应用背景下。此外，紧凑的RegDGCNN模型具有300万个参数和10MB的尺寸，在540k网格面的行业标准设计中，仅在1.2秒内有效地估计了阻力，大大超过了传统的CFD模拟。此外，本研究的RegDGCNN模型通过直接处理3D网格显示出优越的性能，从而无需2D图像渲染或生成符号距离函数( SDF )，简化了预处理阶段，增加了模型的可访问性。重要的是，RegDGCNN模型在不需要水密网格的情况下提供精确的阻力预测的能力突出了其在利用真实数据方面的适应性和有效性。通过将DrivAerNet数据集从560个样本扩充到2800个样本，本研究实现了大约75 %的误差显著降低。同样，在文献[ 36 ]的数据集上，将训练样本从1270个增加到6352个，误差降低了56 %，突出了数据集规模对增强深度学习模型在空气动力学研究中的性能的重要影响。在本研究的DrivAerNet数据集中包含特定的参数修改( 50个几何参数)，显著改善了模型学习，从而显著提高了预测精度，这对于空气动力学设计的精细化至关重要。这强调了大型详细、高保真数据集在精心设计能够处理气动代理模型固有复杂性的模型中的关键作用。

下方展示实验结果：

| Training Set Size (%) | DrivAerNet Dataset (Relative Error) | PaddlePaddle Reproduction (Relative Error) |
| :-------------------: | :---------------------------------: | :----------------------------------------: |
|          20           |                 27%                 |                    NULL                    |
|          40           |                 22%                 |                    NULL                    |
|          60           |                 20%                 |                    NULL                    |
|          80           |                 15%                 |                    NULL                    |
|          100          |                 13%                 |                   7.48%                    |

![fig15](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer/fig/fig15.png)

图15：训练过程中的过程损失曲线变化。

## 6. 参考

参考代码：<https://github.com/Mohamedelrefaie/DrivAerNet>

参考文献列表

[1] A. Abbas, A. Rafiee, M. Haase, and A. Malcolm. Geometrical deep learning for performance prediction of high-speed craft. Ocean Engineering, 258:111716, 2022.

[2] S. R. Ahmed, G. Ramm, and G. Faltin. Some salient features of the time -averaged ground vehicle wake. SAE Transactions, 93:473–503, 1984.

[3] N. Arechiga, F. Permenter, B. Song, and C. Yuan. Drag-guided diffusion models for vehicle image generation. arXiv preprint arXiv:2306.09935, 6 2023.

[4] N. Ashton, P. Batten, A. Cary, and K. Holst. Summary of the 4th high-lift prediction workshop hybrid rans/les technology focus group. Journal of Aircraft, pages 1–30, 2023.

[5] N. Ashton and W. van Noordt. Overview and summary of the first automotive cfd prediction workshop: Drivaer model. SAE International Journal of Commercial Vehicles, 16(02-16-01-0005), 2022.

[6] M. Aultman, Z. Wang, R. Auza-Gutierrez, and L. Duan. Evaluation of cfd methodologies for prediction of flows around simplified and complex automotive models. Computers & Fluids, 236:105297, 2022.

[7] P. Baque, E. Remelli, F. Fleuret, and P. Fua. Geodesic convolutional shape optimization. In J. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 472–481. PMLR, 10–15 Jul 2018.

[8] F. Bonnet, J. Mazari, P. Cinnella, and P. Gallinari. Airfrans: High fidelity computational fluid dynamics dataset for approximating reynolds-averaged navier–stokes solutions. Advances in Neural Information Processing Systems, 35:23463–23478, 2022.

[9] C. Brand, J. Anable, I. Ketsopoulou, and J. Watson. Road to zero or road to nowhere? disrupting transport and energy in a zero carbon world. Energy Policy, 139:111334, 2020.

[10] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[11] A. Cogotti. A parametric study on the ground effect of a simplified car model. SAE transactions, pages 180–204, 1998.

[12] G. Damblin, M. Couplet, and B. Iooss. Numerical studies of space-filling designs: optimization of latin hypercube samples and subprojection properties. Journal of Simulation, 7(4):276–289, 2013.

[13] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[14] M. Elrefaie, T. Ayman, M. A. Elrefaie, E. Sayed, M. Ayyad, and M. M. AbdelRahman. Surrogate modeling of the aerodynamic performance for airfoils in transonic regime. In AIAA SCITECH 2024 Forum, page 2220, 2024.

[15] M. Elrefaie, S. Hüttig, M. Gladkova, T. Gericke, D. Cremers, and C. Breitsamter. Real-time and on-site aerodynamics using stereoscopic piv and deep optical flow learning. arXiv preprint arXiv:2401.09932, 2024.

[16] E. Gunpinar, U. C. Coskun, M. Ozsipahi, and S. Gunpinar. A generative design and drag coefficient prediction system for sedan car side silhouettes based on computational fluid dynamics. CAD Computer Aided Design, 111:65–79, 6 2019.

[17] A. I. Heft, T. Indinger, and N. A. Adams. Experimental and numerical investigation of the drivaer model. In Fluids Engineering Division Summer Meeting, volume 44755, pages 41–51. American Society of Mechanical Engineers, 2012.

[18] A. I. Heft, T. Indinger, and N. A. Adams. Introduction of a new realistic generic car model for aerodynamic investigations. Technical report, SAE Technical Paper, 2012.

[19] S. J. Jacob, M. Mrosek, C. Othmer, and H. Köstler. Deep learning for realtime aerodynamic evaluations of arbitrary vehicle shapes. SAE International Journal of Passenger Vehicle Systems, 15(2):77–90, mar 2022.

[20] A. Kashefi and T. Mukerji. Physics-informed pointnet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries. Journal of Computational Physics, 468:111510, 2022.

[21] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[22] Z. Li, N. B. Kovachki, C. Choy, B. Li, J. Kossaifi, S. P. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, and A. Anandkumar. Geometryinformed neural operator for large-scale 3d pdes, 2023.

[23] H. Martins, C. Henriques, J. Figueira, C. Silva, and A. Costa. Assessing policy interventions to stimulate the transition of electric vehicle technology in the european union. Socio-Economic Planning Sciences, 87:101505, 2023.

[24] F. R. Menter, M. Kuntz, R. Langtry, et al. Ten years of industrial experience with the sst turbulence model. Turbulence, heat and mass transfer, 4(1):625632, 2003.

[25] P. Mock and S. Díaz. Pathways to decarbonization: the european passenger car market in the years 2021–2035. communications, 49:847129–848102, 2021.

[26] T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, and P. W. Battaglia. Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409, 2020.

[27] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017.

[28] E. Remelli, A. Lukoianov, S. Richter, B. Guillard, T. Bagautdinov, P. Baque, and P. Fua. Meshsdf: Differentiable iso-surface extraction. Advances in Neural Information Processing Systems, 33:22468–22478, 2020.

[29] T. Rios, B. Sendhoff, S. Menzel, T. Back, and B. V. Stein. On the efficiency of a point cloud autoencoder as a geometric representation for shape optimization. pages 791–798. Institute of Electrical and Electronics Engineers Inc., 12 2019.

[30] T. Rios, B. V. Stein, T. Back, B. Sendhoff, and S. Menzel. Point2ffd: Learning shape representations of simulation-ready 3d models for engineering design optimization. pages 1024–1033. Institute of Electrical and Electronics Engineers Inc., 2021.

[31] T. Rios, B. van Stein, P. Wollstadt, T. Bäck, B. Sendhoff, and S. Menzel. Exploiting local geometric features in vehicle design optimization with 3d point cloud autoencoders. In 2021 IEEE Congress on Evolutionary Computation (CEC), pages 514–521, 2021.

[32] T. Rios, P. Wollstadt, B. V. Stein, T. Back, Z. Xu, B. Sendhoff, and S. Menzel. Scalability of learning tasks on 3d cae models using point cloud autoencoders. pages 1367–1374. Institute of Electrical and Electronics Engineers Inc., 12 2019.

[33] F. Romor, M. Tezzele, M. Mrosek, C. Othmer, and G. Rozza. Multi-fidelity data fusion through parameter space reduction with applications to automotive engineering. International Journal for Numerical Methods in Engineering, 124(23):5293–5311, 2023.

[34] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. Battaglia. Learning to simulate complex physics with graph networks. In International conference on machine learning, pages 8459–8468. PMLR, 2020.

[35] Y. Shen, H. C. Patel, Z. Xu, and J. J. Alonso. Application of multi-fidelity transfer learning with autoencoders for efficient construction of surrogate models. In AIAA SCITECH 2024 Forum, page 0013, 2024.

[36] B. Song, C. Yuan, F. Permenter, N. Arechiga, and F. Ahmed. Surrogate modeling of car drag coefficient with depth and normal renderings. arXiv preprint arXiv:2306.06110, 2023.

[37] D. B. Spalding. The numerical computation of turbulent flow. Comp. Methods Appl. Mech. Eng., 3:269, 1974.

[38] N. Thuerey, K. Weißenow, L. Prantl, and X. Hu. Deep learning methods for reynolds-averaged navier–stokes simulations of airfoil flows. AIAA Journal, 58(1):25–36, 2020.

[39] T. L. Trinh, F. Chen, T. Nanri, and K. Akasaka. 3d super-resolution model for vehicle flow field enrichment. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5826–5835, 2024.

[40] N. Umetani and B. Bickel. Learning three-dimensional flow for interactive aerodynamic design. ACM Transactions on Graphics, 37, 2018.

[41] M. Usama, A. Arif, F. Haris, S. Khan, S. K. Afaq, and S. Rashid. A data-driven interactive system for aerodynamic and user-centred generative vehicle design. In 2021 International Conference on Artificial Intelligence (ICAI), pages 119–127, 2021.

[42] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12, 2019.

[43] D. Wieser, H.-J. Schmidt, S. Mueller, C. Strangfeld, C. Nayeri, and C. Paschereit. Experimental comparison of the aerodynamic behavior of fastback and notchback drivaer models. SAE International Journal of Passenger Cars-Mechanical Systems, 7(2014-01-0613):682–691, 2014.
