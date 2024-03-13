# PaddleScience

<!-- --8<-- [start:status] -->
> *Developed with [PaddlePaddle](https://www.paddlepaddle.org.cn/)*

[![Version](https://img.shields.io/pypi/v/paddlesci)](https://pypi.org/project/paddlesci/)
[![Python Version](https://img.shields.io/pypi/pyversions/paddlesci)](https://pypi.org/project/paddlesci/)
[![Doc](https://img.shields.io/readthedocs/paddlescience-docs/latest)](https://paddlescience-docs.readthedocs.io/zh/latest/)
[![Code Style](https://img.shields.io/badge/code_style-black-black)](https://github.com/psf/black)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![License](https://img.shields.io/github/license/PaddlePaddle/PaddleScience)](https://github.com/PaddlePaddle/PaddleScience/blob/develop/LICENSE)
<!-- --8<-- [end:status] -->

[**PaddleScience使用文档**](https://paddlescience-docs.readthedocs.io/zh/latest/)

<!-- --8<-- [start:description] -->
## 👀简介

PaddleScience 是一个基于深度学习框架 PaddlePaddle 开发的科学计算套件，利用深度神经网络的学习能力和 PaddlePaddle 框架的自动(高阶)微分机制，解决物理、化学、气象等领域的问题。支持物理机理驱动、数据驱动、数理融合三种求解方式，并提供了基础 API 和详尽文档供用户使用与二次开发。
<!-- --8<-- [end:description] -->

## 📝案例列表

<p align="center"><b>数学(AI for Math)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 微分方程 | [拉普拉斯方程](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/laplace2d) | 机理驱动 | MLP | 无监督学习 | -        | - |
| 微分方程 | [伯格斯方程](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/deephpms) | 机理驱动 | MLP | 无监督学习 | [Data](https://github.com/maziarraissi/DeepHPMs/tree/master/Data) | [Paper](https://arxiv.org/pdf/1801.06637.pdf) |</center>
| 微分方程 | [非线性偏微分方程](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/pirbn) | 机理驱动 | PIRBN | 无监督学习 | - | [Paper](https://arxiv.org/abs/2304.06234) |
| 微分方程 | [洛伦兹方程](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/lorenz) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
| 微分方程 | [若斯叻方程](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/rossler) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
| 算子学习 | [DeepONet](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/deeponet) | 数据驱动 | MLP | 监督学习 | [Data](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html) | [Paper](https://export.arxiv.org/pdf/1910.03193.pdf) |
| 微分方程 | [梯度增强的物理知识融合 PDE 求解](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/gpinn/poisson_1d.py) | 机理驱动 | gPINN | 无监督学习 | - |  [Paper](https://doi.org/10.1016/j.cma.2022.114823) |
| 积分方程 | [沃尔泰拉积分方程](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide) | 机理驱动 | MLP | 无监督学习 | - | [Project](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py) |
| 微分方程 | [分数阶微分方程](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fpde/fractional_poisson_2d.py) | 机理驱动 | MLP | 无监督学习 | - | - |

<br>
<p align="center"><b>技术科学(AI for Technology)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 定常不可压流体 | [2D 定常方腔流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/ldc2d_steady) | 机理驱动 | MLP | 无监督学习 | - |  |
| 定常不可压流体 | [2D 达西流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/darcy2d) | 机理驱动 | MLP | 无监督学习 | - |   |
| 定常不可压流体 | [2D 管道流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/labelfree_DNN_surrogate) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/1906.02382) |
| 定常不可压流体 | [3D 血管瘤](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/aneurysm) | 机理驱动 | MLP | 无监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar) | [Project](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)|
| 定常不可压流体 | [任意 2D 几何体绕流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/deepcfd) | 数据驱动 | DeepCFD | 监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [2D 非定常方腔流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/ldc2d_unsteady) | 机理驱动 | MLP | 无监督学习 | - | - |
| 非定常不可压流体 | [Re100 2D 圆柱绕流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/cylinder2d_unsteady) | 机理驱动 | MLP | 半监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar) | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [Re100~750 2D 圆柱绕流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/cylinder2d_unsteady_transformer_physx) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957)|
| 可压缩流体 | [2D 空气激波](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/shock_wave) | 机理驱动 | PINN-WE | 无监督学习 | - | [Paper](https://arxiv.org/abs/2206.03864)|
| 飞行器设计 | [MeshGraphNets](https://aistudio.baidu.com/projectdetail/5322713) | 数据驱动 | GNN | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/184320) | [Paper](https://arxiv.org/abs/2010.03409)|
| 飞行器设计 | [火箭发动机真空羽流](https://aistudio.baidu.com/projectdetail/4486133) | 数据驱动 | CNN | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/167250) | - |
| 飞行器设计 | [Deep-Flow-Prediction](https://aistudio.baidu.com/projectdetail/5671596) | 数据驱动 | TurbNetG | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/197778) | [Paper](https://arxiv.org/abs/1810.08217) |
| 通用流场模拟 | [气动外形设计](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/amgnet) | 数据驱动 | AMGNet | 监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip) | [Paper](https://arxiv.org/abs/1810.08217) |
| 流固耦合 | [涡激振动](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/viv) | 机理驱动 | MLP | 半监督学习 | [Data](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fsi/VIV_Training_Neta100.mat) | [Paper](https://arxiv.org/abs/2206.03864)|
| 多相流 | [气液两相流](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/bubble) | 机理驱动 | BubbleNet | 半监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat) | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 多相流 | [twophasePINN](https://aistudio.baidu.com/projectdetail/5379212) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://doi.org/10.1016/j.mlwa.2021.100029)|
| 多相流 | 非高斯渗透率场估计<sup>coming soon</sup> | 机理驱动 | cINN | 监督学习 | - | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 流场高分辨率重构 | [2D 湍流流场重构](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/tempoGAN) | 数据驱动 | tempoGAN | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://dl.acm.org/doi/10.1145/3197517.3201304)|
| 流场高分辨率重构 | [2D 湍流流场重构](https://aistudio.baidu.com/projectdetail/4493261?contributionType=1) | 数据驱动 | cycleGAN | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://arxiv.org/abs/2007.15324)|
| 流场高分辨率重构 | [基于Voronoi嵌入辅助深度学习的稀疏传感器全局场重建](https://aistudio.baidu.com/projectdetail/5807904) | 数据驱动 | CNN | 监督学习 | [Data1](https://drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c)<br>[Data2](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)<br>[Data3](https://drive.google.com/drive/folders/1xIY_jIu-hNcRY-TTf4oYX1Xg4_fx8ZvD) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 流场高分辨率重构 | 基于扩散的流体超分重构<sup>coming soon</sup> | 数理融合 | DDPM | 监督学习 | - | [Paper](https://www.sciencedirect.com/science/article/pii/S0021999123000670)|
| 求解器耦合 | [CFD-GCN](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/cfdgcn) | 数据驱动 | GCN | 监督学习 | [Data](https://aistudio.baidu.com/aistudio/datasetdetail/184778)<br>[Mesh](https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/meshes.tar) | [Paper](https://arxiv.org/abs/2007.04439)|
| 受力分析 | [1D 欧拉梁变形](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/euler_beam) | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | [2D 平板变形](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/biharmonic2d) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/2108.07243) |
| 受力分析 | [3D 连接件变形](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/bracket) | 机理驱动 | MLP | 无监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar) | [Tutorial](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html) |
| 受力分析 | [结构震动模拟](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/phylstm) | 机理驱动 | PhyLSTM | 监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat) | [Paper](https://arxiv.org/abs/2002.10253) |
| 受力分析 | [2D 弹塑性结构](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/epnn) | 机理驱动 | EPNN | 无监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat) | [Paper](https://arxiv.org/abs/2204.12088) |
| 受力分析和逆问题 | [3D 汽车控制臂变形](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/control_arm) | 机理驱动 | MLP | 无监督学习 | - | - |
| 拓扑优化 | [2D 拓扑优化](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/topopt) | 数据驱动 | TopOptNN | 监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5) | [Paper](https://arxiv.org/pdf/1709.09578) |
| 热仿真 | [1D 换热器热仿真](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/heat_exchanger) | 机理驱动 | PI-DeepONet | 无监督学习 | - | - |
| 热仿真 | [2D 热仿真](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/heat_pinn) | 机理驱动 | PINN | 无监督学习 | - | [Paper](https://arxiv.org/abs/1711.10561)|

<br>
<p align="center"><b>材料科学(AI for Material)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 材料设计 | [散射板设计(反问题)](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/hpinns) | 数理融合 | 数据驱动 | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat) | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |
| 材料生成 | 面向对称感知的周期性材料生成<sup>coming soon</sup> | 数据驱动 | SyMat | 监督学习 | - | - |

<br>
<p align="center"><b>地球科学(AI for Earth Science)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 天气预报 | [FourCastNet 气象预报](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/fourcastnet) | 数据驱动 | FourCastNet | 监督学习 | [ERA5](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 天气预报 | [NowCastNet 气象预报](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/nowcastnet) | 数据驱动 | NowCastNet | 监督学习 | [MRMS](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F) | [Paper](https://www.nature.com/articles/s41586-023-06184-4) |
| 天气预报 | [GraphCast 气象预报](jointContribution/graphcast/README.md) | 数据驱动 | GraphCastNet | 监督学习 | - | [Paper](https://arxiv.org/abs/2212.12794) |
| 大气污染物 | [UNet 污染物扩散](https://aistudio.baidu.com/projectdetail/5663515?channel=0&channelType=0&sUid=438690&shared=1&ts=1698221963752) | 数据驱动 | UNet | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/198102) | - |

<!-- --8<-- [start:update] -->
## 🕘最近更新

- 添加多目标优化算法 [Relobralo](https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/loss/mtl/#ppsci.loss.mtl.Relobralo) 。
- 添加气泡流求解案例([Bubble](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/bubble))、机翼优化案例([DeepCFD](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/deepcfd/))、热传导仿真案例([HeatPINN](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/heat_pinn))、非线性短临预报模型([Nowcasting(仅推理)](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/nowcastnet))、拓扑优化案例([TopOpt](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/topopt))、矩形平板线弹性方程求解案例([Biharmonic2D](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/biharmonic2d))。
- 添加二维血管案例([LabelFree-DNN-Surrogate](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/labelfree_DNN_surrogate/#4))、空气激波案例([ShockWave](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/shock_wave/))、去噪网络模型([DUCNN](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/DU_CNN))、风电预测模型([Deep Spatial Temporal](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/Deep-Spatio-Temporal))、域分解模型([XPINNs](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/XPINNs))、积分方程求解案例([Volterra Equation](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide))、分数阶方程求解案例([Fractional Poisson 2D](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fpde/fractional_poisson_2d.py))。
- 针对串联方程和复杂方程场景，`Equation` 模块支持基于 [sympy](https://docs.sympy.org/dev/tutorials/intro-tutorial/intro.html) 的符号计算，并支持和 python 函数混合使用([#507](https://github.com/PaddlePaddle/PaddleScience/pull/507)、[#505](https://github.com/PaddlePaddle/PaddleScience/pull/505))。
- `Geometry` 模块和 `InteriorConstraint`、`InitialConstraint` 支持计算 SDF 微分功能([#539](https://github.com/PaddlePaddle/PaddleScience/pull/539))。
- 添加 **M**ulti**T**ask**L**earning(`ppsci.loss.mtl`) 多任务学习模块，针对多任务优化(如 PINN 方法)进一步提升性能，使用方式：[多任务学习指南](https://paddlescience-docs.readthedocs.io/zh/latest/zh/user_guide/#24)([#493](https://github.com/PaddlePaddle/PaddleScience/pull/505)、[#492](https://github.com/PaddlePaddle/PaddleScience/pull/505))。
<!-- --8<-- [end:update] -->

<!-- --8<-- [start:feature] -->
## ✨特性

- 支持简单几何和复杂 STL 几何的采样与布尔运算。
- 支持包括 Dirichlet、Neumann、Robin 以及自定义边界条件。
- 支持物理机理驱动、数据驱动、数理融合三种问题求解方式。涵盖流体、结构、气象等领域 20+ 案例。
- 支持结果可视化输出与日志结构化保存。
- 完善的 type hints，用户使用和代码贡献全流程文档，经典案例 AI studio 快速体验，降低使用门槛，提高开发效率。
- 支持基于 sympy 符号计算库的方程表示与联立方程组计算。
- 更多特性正在开发中...
<!-- --8<-- [end:feature] -->

## 🚀安装使用

### 安装 PaddlePaddle

<!-- --8<-- [start:paddle_install] -->
请在 [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 官网按照您的运行环境，安装 <font color="red"><b>develop</b></font> 版的 PaddlePaddle。

安装完毕之后，运行以下命令，验证 Paddle 是否安装成功。

``` shell
python -c "import paddle; paddle.utils.run_check()"
```

如果出现 `PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.` 信息，说明您已成功安装，可以继续安装 PaddleScience。
<!-- --8<-- [end:paddle_install] -->

### 安装 PaddleScience

1. 基础功能安装

    **从以下三种安装方式中，任选一种均可安装。**

    - git 源码安装[**推荐**]

        执行以下命令，从 github 上 clone PaddleScience 源代码，并以 editable 的方式安装 PaddleScience。
        <!-- --8<-- [start:git_install] -->
        ``` shell
        git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
        # 若 github clone 速度比较慢，可以使用 gitee clone
        # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

        cd PaddleScience

        # install paddlesci with editable mode
        pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
        ```
        <!-- --8<-- [end:git_install] -->

    - pip 安装

        执行以下命令以 pip 的方式安装最新版本的 PaddleScience。
        <!-- --8<-- [start:pip_install] -->
        ``` shell
        pip install -U paddlesci
        ```
        <!-- --8<-- [end:pip_install] -->

    - 设置 PYTHONPATH

        如果在您的环境中，上述两种方式都无法正常安装，则可以选择本方式，在终端内将环境变量 `PYTHONPATH` 临时设置为 `PaddleScience` 的**绝对路径**，如下所示。

        ``` shell
        cd PaddleScience
        export PYTHONPATH=$PYTHONPATH:$PWD # for linux
        set PYTHONPATH=%cd% # for windows
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple # manually install requirements
        ```

        注：上述方式的优点是步骤简单无需安装，缺点是当环境变量生效的终端被关闭后，需要重新执行上述命令设置 `PYTHONPATH` 才能再次使用 PaddleScience，较为繁琐。

2. 验证安装

    ``` py
    python -c "import ppsci; ppsci.utils.run_check()"
    ```

3. 开始使用

    ``` py
    import ppsci

    # write your code here...
    ```

如需基于复杂几何文件（`*.stl`, `*.mesh`, `*.obj`）文件进行训练、测试等流程，请参考完整安装流程：[**安装与使用**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/)

## ⚡️快速开始

请参考 [**快速开始**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/quickstart/)

## 🎈其他领域支持

除 PaddleScience 套件外，Paddle 框架还支持了 [DeepXDE](https://github.com/lululxvi/deepxde/tree/master?tab=readme-ov-file#deepxde) 的所有案例，分子动力学套件 [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit/tree/paddle2?tab=readme-ov-file#deepmd-kitpaddlepaddle-backend) 部分案例和功能，以及正在适配中的 Modulus。

<!-- --8<-- [start:support] -->
## 💬支持与建议

如使用过程中遇到问题或想提出开发建议，欢迎在 [**Discussion**](https://github.com/PaddlePaddle/PaddleScience/discussions/new?category=general) 提出建议，或者在 [**Issue**](https://github.com/PaddlePaddle/PaddleScience/issues/new/choose) 页面新建 issue，会有专业的研发人员进行解答。
<!-- --8<-- [end:support] -->

<!-- --8<-- [start:contribution] -->
## 👫开源共建

PaddleScience 项目欢迎并依赖开发人员和开源社区中的用户，会不定期推出开源活动。

> 在开源活动中如需使用 PaddleScience 进行开发，可参考 [**PaddleScience 开发与贡献指南**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/) 以提升开发效率和质量。

- 🎁快乐开源

    旨在鼓励更多的开发者参与到飞桨科学计算社区的开源建设中，帮助社区修复 bug 或贡献 feature，加入开源、共建飞桨。了解编程基本知识的入门用户即可参与，活动进行中：
    [PaddleScience 快乐开源活动表单](https://github.com/PaddlePaddle/PaddleScience/issues/379)

- 🔥第五期黑客松

    面向全球开发者的深度学习领域编程活动，鼓励开发者了解与参与飞桨深度学习开源项目与文心大模型开发实践。活动进行中：[【PaddlePaddle Hackathon 5th】开源贡献个人挑战赛](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/%E3%80%90PaddlePaddle%20Hackathon%205th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E4%BB%BB%E5%8A%A1%E5%BC%80%E5%8F%91%E6%B5%81%E7%A8%8B%E4%B8%8E%E9%AA%8C%E6%94%B6%E6%A0%87%E5%87%86)
<!-- --8<-- [end:contribution] -->

<!-- --8<-- [start:collaboration] -->
## 🎯共创计划

PaddleScience 作为一个开源项目，欢迎来各行各业的伙伴携手共建基于飞桨的 AI for Science 领域顶尖开源项目, 打造活跃的前瞻性的 AI for Science 开源社区，建立产学研闭环，推动科研创新与产业赋能。点击了解 [飞桨AI for Science共创计划](https://www.paddlepaddle.org.cn/science)。
<!-- --8<-- [end:collaboration] -->

<!-- --8<-- [start:thanks] -->
## ❤️致谢

- PaddleScience 的部分模块和案例设计受 [NVIDIA-Modulus](https://github.com/NVIDIA/modulus/tree/main)、[DeepXDE](https://github.com/lululxvi/deepxde/tree/master)、[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop) 等优秀开源套件的启发。
<!-- --8<-- [end:thanks] -->
- PaddleScience 的部分案例和代码由以下优秀社区开发者贡献（按 [Contributors](https://github.com/PaddlePaddle/PaddleScience/graphs/contributors) 排序）：
    [Asthestarsfalll](https://github.com/Asthestarsfalll)，
    [co63oc](https://github.com/co63oc)，
    [MayYouBeProsperous](https://github.com/MayYouBeProsperous)，
    [AndPuQing](https://github.com/AndPuQing)，
    [lknt](https://github.com/lknt)，
    [mrcangye](https://github.com/mrcangye)，
    [yangguohao](https://github.com/yangguohao)，
    [ooooo-create](https://github.com/ooooo-create)，
    [megemini](https://github.com/megemini)，
    [DUCH714](https://github.com/DUCH714)，
    [zlynna](https://github.com/zlynna)，
    [jjyaoao](https://github.com/jjyaoao)，
    [jiamingkong](https://github.com/jiamingkong)，
    [Liyulingyue](https://github.com/Liyulingyue)，
    [DrRyanHuang](https://github.com/DrRyanHuang)，
    [zbt78](https://github.com/zbt78)，
    [Gxinhu](https://github.com/Gxinhu)，
    [XYM](https://github.com/XYM)，
    [xusuyong](https://github.com/xusuyong)，
    [DrownFish19](https://github.com/DrownFish19)，
    [NKNaN](https://github.com/NKNaN)，
    [ruoyunbai](https://github.com/ruoyunbai)，
    [sanbuphy](https://github.com/sanbuphy)，
    [ccsuzzh](https://github.com/ccsuzzh)，
    [enkilee](https://github.com/enkilee)，
    [GreatV](https://github.com/GreatV)

## 🤝合作单位

![cooperation](./docs/images/overview/cooperation.png)

<!-- --8<-- [start:license] -->
## 📜开源协议

[Apache License 2.0](https://github.com/PaddlePaddle/PaddleScience/blob/develop/LICENSE)
<!-- --8<-- [end:license] -->
