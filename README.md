# PaddleScience

<!-- --8<-- [start:status] -->
> *Developed with [PaddlePaddle](https://www.paddlepaddle.org.cn/)*

[![Version](https://img.shields.io/pypi/v/paddlesci)](https://pypi.org/project/paddlesci/)
[![Python Version](https://img.shields.io/pypi/pyversions/paddlesci)](https://pypi.org/project/paddlesci/)
[![Doc](https://img.shields.io/readthedocs/paddlescience-docs/latest)](https://paddlescience-docs.readthedocs.io/zh/latest/)
[![Code Style](https://img.shields.io/badge/code_style-black-black)](https://img.shields.io/badge/code_style-black-black)
[![License](https://img.shields.io/github/license/PaddlePaddle/PaddleScience)](./LICENSE)
<!-- --8<-- [end:status] -->

[**PaddleScience使用文档**](https://paddlescience-docs.readthedocs.io/zh/latest/)

<!-- --8<-- [start:description] -->
## 简介

PaddleScience 是一个基于深度学习框架 PaddlePaddle 开发的科学计算套件，利用深度神经网络的学习能力和 PaddlePaddle 框架的自动(高阶)微分机制，解决物理、化学、气象等领域的问题。支持物理机理驱动、数据驱动、数理融合三种求解方式，并提供了基础 API 和详尽文档供用户使用与二次开发。
<!-- --8<-- [end:description] -->

<div align="center">
    <img src="https://paddle-org.bj.bcebos.com/paddlescience/docs/overview/panorama.png" width="100%" height="100%">
</div>

<!-- --8<-- [start:update] -->
## 最近更新

- 添加二维血管案例([LabelFree-DNN-Surrogate](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/labelfree_DNN_surrogate/#4))、空气激波案例([ShockWave](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/shock_wave/))、去噪网络模型([DUCNN](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/DU_CNN))、风电预测模型([Deep Spatial Temporal](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/Deep-Spatio-Temporal))、域分解模型([XPINNs](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/XPINNs))、积分方程求解案例([Volterra Equation](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide/))、分数阶方程求解案例([Fractional Poisson 2D](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fpde/fractional_poisson_2d.py))。
- 针对串联方程和复杂方程场景，`Equation` 模块支持基于 [sympy](https://docs.sympy.org/dev/tutorials/intro-tutorial/intro.html) 的符号计算，并支持和 python 函数混合使用([#507](https://github.com/PaddlePaddle/PaddleScience/pull/507)、[#505](https://github.com/PaddlePaddle/PaddleScience/pull/505))。
- `Geometry` 模块和 `InteriorConstraint`、`InitialConstraint` 支持计算 SDF 微分功能([#539](https://github.com/PaddlePaddle/PaddleScience/pull/539))。
- 添加 **M**ulti**T**ask**L**earning(`ppsci.loss.mtl`) 多任务学习模块，针对多任务优化(如 PINN 方法)进一步提升性能，使用方式：[多任务学习指南](https://paddlescience-docs.readthedocs.io/zh/latest/zh/user_guide/#24)([#493](https://github.com/PaddlePaddle/PaddleScience/pull/505)、[#492](https://github.com/PaddlePaddle/PaddleScience/pull/505))。
<!-- --8<-- [end:update] -->

<!-- --8<-- [start:feature] -->
## 特性

- 支持简单几何和复杂 STL 几何的采样与布尔运算。
- 支持包括 Dirichlet、Neumann、Robin 以及自定义边界条件。
- 支持物理机理驱动、数据驱动、数理融合三种问题求解方式。涵盖流体、结构、气象等领域 20+ 案例。
- 支持结果可视化输出与日志结构化保存。
- 完善的 type hints，用户使用和代码贡献全流程文档，经典案例 AI studio 快速体验，降低使用门槛，提高开发效率。
- 支持基于 sympy 符号计算库的方程表示。
- 更多特性正在开发中...
<!-- --8<-- [end:feature] -->

## 安装使用

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

更多安装方式请参考 [**安装与使用**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/)

## 快速开始

请参考 [**快速开始**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/quickstart/)

## 经典案例

请参考 [**经典案例**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/viv/)

<!-- --8<-- [start:support] -->
## 支持

如使用过程中遇到问题或想提出开发建议，欢迎在 [**Discussion**](https://github.com/PaddlePaddle/PaddleScience/discussions/new?category=general) 提出建议，或者在 [**Issue**](https://github.com/PaddlePaddle/PaddleScience/issues/new/choose) 页面新建 issue。
<!-- --8<-- [end:support] -->

<!-- --8<-- [start:contribution] -->
## 贡献代码

PaddleScience 项目欢迎并依赖开发人员和开源社区中的用户，请参阅 [**贡献指南**](https://paddlescience-docs.readthedocs.io/zh/latest/zh/contribute/)。
<!-- --8<-- [end:contribution] -->

<!-- --8<-- [start:thanks] -->
## 致谢

PaddleScience 的部分模块和案例设计受 [NVIDIA-Modulus](https://github.com/NVIDIA/modulus/tree/main)、[DeepXDE](https://github.com/lululxvi/deepxde/tree/master)、[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop) 等优秀开源套件的启发。
<!-- --8<-- [end:thanks] -->

<!-- --8<-- [start:license] -->
## 证书

[Apache License 2.0](https://github.com/PaddlePaddle/PaddleScience/blob/develop/LICENSE)
<!-- --8<-- [end:license] -->
