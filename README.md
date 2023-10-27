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
| 微分方程 | [拉普拉斯方程](./docs/zh/examples/laplace2d.md) | 机理驱动 | MLP | 无监督学习 | -        | - |
| 微分方程 | [伯格斯方程](./docs/zh/examples/deephpms.md) | 机理驱动 | MLP | 无监督学习 | [Data](https://github.com/maziarraissi/DeepHPMs/tree/master/Data) | [Paper](https://arxiv.org/pdf/1801.06637.pdf) |</center>
| 微分方程 | [洛伦兹方程](./docs/zh/examples/lorenz.md) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
| 微分方程 | [若斯叻方程](./docs/zh/examples/rossler.md) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
| 算子学习 | [DeepONet](./docs/zh/examples/deeponet.md) | 数据驱动 | MLP | 监督学习 | [Data](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html) | [Paper](https://export.arxiv.org/pdf/1910.03193.pdf) |
| 微分方程 | 梯度增强的物理知识融合PDE求解<sup>coming soon</sup> | 机理驱动 | gPINN | 半监督学习 | - |  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782522001438?via%3Dihub) |
| 积分方程 | [沃尔泰拉积分方程](./docs/zh/examples/volterra_ide.md) | 机理驱动 | MLP | 无监督学习 | - | [Project](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py) |

<br>
<p align="center"><b>技术科学(AI for Technology)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 定常不可压流体 | [2D 定常方腔流](./docs/zh/examples/ldc2d_steady.md) | 机理驱动 | MLP | 无监督学习 | - |  |
| 定常不可压流体 | [2D 达西流](./docs/zh/examples/darcy2d.md) | 机理驱动 | MLP | 无监督学习 | - |   |
| 定常不可压流体 | [2D 管道流](./docs/zh/examples/labelfree_DNN_surrogate.md) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/1906.02382) |
| 定常不可压流体 | [3D 血管瘤](./docs/zh/examples/aneurysm.md) | 机理驱动 | MLP | 无监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar) | [Project](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)|
| 定常不可压流体 | [任意 2D 几何体绕流](./docs/zh/examples/deepcfd.md) | 数据驱动 | DeepCFD | 监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [2D 非定常方腔流](./docs/zh/examples/ldc2d_unsteady.md) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [Re100 2D 圆柱绕流](./docs/zh/examples/cylinder2d_unsteady.md) | 机理驱动 | MLP | 半监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar) | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [Re100~750 2D 圆柱绕流](./docs/zh/examples/cylinder2d_unsteady_transformer_physx.md) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957)|
| 可压缩流体 | [2D 空气激波](./docs/zh/examples/shock_wave.md) | 机理驱动 | PINN-WE | 无监督学习 | - | [Paper](https://arxiv.org/abs/2206.03864)|
| 流固耦合 | [涡激振动](./docs/zh/examples/viv.md) | 机理驱动 | MLP | 半监督学习 | [Data](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fsi/VIV_Training_Neta100.mat) | [Paper](https://arxiv.org/abs/2206.03864)|
| 多相流 | [气液两相流](./docs/zh/examples/bubble.md) | 机理驱动 | BubbleNet | 半监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat) | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 多相流 | 非高斯渗透率场估计<sup>coming soon</sup> | 机理驱动 | cINN | 监督学习 | - | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 流场高分辨率重构 | [2D 湍流流场重构](./docs/zh/examples/tempoGAN.md) | 数据驱动 | tempoGAN | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)]<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://dl.acm.org/doi/10.1145/3197517.3201304)|
| 流场高分辨率重构 | 基于扩散的流体超分重构<sup>coming soon</sup> | 数理融合 | DDPM | 监督学习 | - | [Paper](https://www.sciencedirect.com/science/article/pii/S0021999123000670)|
| 受力分析 | [1D 欧拉梁变形](https://github.com/HydrogenSulfate/PaddleScience/blob/add_exm_table/examples/euler_beam/euler_beam.py) | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | 2D 平板变形<sup>coming soon</sup> | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | [3D 连接件变形](./docs/zh/examples/bracket.md) | 机理驱动 | MLP | 无监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar) | [Tutorial](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html) |
| 受力分析 | [结构震动模拟](./docs/zh/examples/phylstm.md) | 机理驱动 | PhyLSTM | 监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat) | [Paper](https://arxiv.org/abs/2002.10253) |

<br>
<p align="center"><b>材料科学(AI for Material)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 材料设计 | [散射板设计(反问题)](./zh/examples/hpinns.md) | 数理融合 | 数据驱动 | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat) | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |
| 材料生成 | 面向对称感知的周期性材料生成<sup>coming soon</sup> | 数据驱动 | SyMat | 监督学习 | - | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |

<br>
<p align="center"><b>地球科学(AI for Earth Science)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 天气预报 | [FourCastNet 气象预报](./docs/zh/examples/fourcastnet.md) | 数据驱动 | FourCastNet | 监督学习 | [ERA5](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 天气预报 | GraphCast 气象预报<sup>coming soon</sup> | 数据驱动 | GraphCastNet* | 监督学习 | - | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 大气污染物 | [UNet 污染物扩散](https://aistudio.baidu.com/projectdetail/5663515?channel=0&channelType=0&sUid=438690&shared=1&ts=1698221963752) | 数据驱动 | UNet | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/198102) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |

<!-- --8<-- [start:update] -->
## 🕘最近更新

- 添加二维血管案例([LabelFree-DNN-Surrogate](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/labelfree_DNN_surrogate/#4))、空气激波案例([ShockWave](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/shock_wave/))、去噪网络模型([DUCNN](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/DU_CNN))、风电预测模型([Deep Spatial Temporal](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/Deep-Spatio-Temporal))、域分解模型([XPINNs](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/XPINNs))、积分方程求解案例([Volterra Equation](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide/))、分数阶方程求解案例([Fractional Poisson 2D](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fpde/fractional_poisson_2d.py))。
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
- 支持基于 sympy 符号计算库的方程表示。
- 更多特性正在开发中...
<!-- --8<-- [end:feature] -->

## 🚀安装使用

1. 执行以下命令，从 github 上克隆 PaddleScience 项目，进入 PaddleScience 目录，并将该目录添加到系统环境变量中

    <!-- --8<-- [start:git_install] -->
    ``` shell
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # 若 github clone 速度比较慢，可以使用 gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience
    # install paddlesci with editable mode
    pip install -e .
    ```
    <!-- --8<-- [end:git_install] -->

2. 安装必要的依赖包

    ``` shell
    pip install -r requirements.txt
    ```

3. 验证安装

    ``` py
    python -c "import ppsci; ppsci.utils.run_check()"
    ```

4. 开始使用

    ``` py
    import ppsci

    # write your code here...
    ```

完整安装流程请参考 [**安装与使用**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/)

## ⚡️快速开始

请参考 [**快速开始**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/quickstart/)

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

<!-- --8<-- [start:thanks] -->
## ❤️致谢

- PaddleScience 的部分模块和案例设计受 [NVIDIA-Modulus](https://github.com/NVIDIA/modulus/tree/main)、[DeepXDE](https://github.com/lululxvi/deepxde/tree/master)、[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop) 等优秀开源套件的启发。
<!-- --8<-- [end:thanks] -->
- PaddleScience 的部分案例和代码由以下优秀社区开发者贡献（按 [Contributors](https://github.com/PaddlePaddle/PaddleScience/graphs/contributors) 排序）：
    [Asthestarsfalll](https://github.com/Asthestarsfalll)，
    [co63oc](https://github.com/co63oc)，
    [AndPuQing](https://github.com/AndPuQing)，
    [MayYouBeProsperous](https://github.com/MayYouBeProsperous)，
    [yangguohao](https://github.com/yangguohao)，
    [mrcangye](https://github.com/mrcangye)，
    [jjyaoao](https://github.com/jjyaoao)，
    [jiamingkong](https://github.com/jiamingkong)，
    [Liyulingyue](https://github.com/Liyulingyue)，
    [XYM](https://github.com/XYM)，
    [xusuyong](https://github.com/xusuyong)，
    [NKNaN](https://github.com/NKNaN)，
    [ruoyunbai](https://github.com/ruoyunbai)，
    [sanbuphy](https://github.com/sanbuphy)，
    [ccsuzzh](https://github.com/ccsuzzh)，
    [enkilee](https://github.com/enkilee)，
    [GreatV](https://github.com/GreatV)

<!-- --8<-- [start:license] -->
## 📜证书

[Apache License 2.0](https://github.com/PaddlePaddle/PaddleScience/blob/develop/LICENSE)
<!-- --8<-- [end:license] -->
