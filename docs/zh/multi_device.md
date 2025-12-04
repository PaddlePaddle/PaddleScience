# 多硬件支持

飞桨生态的繁荣离不开开发者和用户的贡献，我们非常欢迎为飞桨的多硬件适配贡献更多的模型。

## 1. 硬件支持列表

当前 PaddleScience 中对于各硬件适配模型的列表汇总如下（不包含仅在 AIStudio 上的案例）

=== "数学(AI for Math)"

    | 问题类型 | 案例名称 | NVIDIA | 海光 | 太初 | 沐曦 |
    |-----|-----|-----|-----|-----|-----|
    | 亥姆霍兹方程 | [SPINN(Helmholtz3D)](./examples/spinn.md) | ✅ | | ✅ | ✅ |
    | 相场方程 | [Allen-Cahn](./examples/allen_cahn.md) | ✅ | | | ✅ |
    | 微分方程 | [拉普拉斯方程](./examples/laplace2d.md) | ✅ | | ✅ | ✅ |
    | 微分方程 | [伯格斯方程](./examples/deephpms.md) | ✅ | | | ✅ |
    | 微分方程 | [非线性偏微分方程](./examples/pirbn.md) | ✅ | | | ✅ |
    | 微分方程 | [洛伦兹方程](./examples/lorenz.md) | ✅ | | ✅ | ✅ |
    | 微分方程 | [若斯叻方程](./examples/rossler.md) | ✅ | | ✅ | ✅ |
    | 算子学习 | [DeepONet](./examples/deeponet.md) | ✅ | | ✅ | ✅ |
    | 积分方程 | [沃尔泰拉积分方程](./examples/volterra_ide.md) | ✅ | | ✅ | ✅ |
    | 光纤怪波 | [Optical rogue wave](./examples/nlsmb.md) | ✅ | | ✅ | ✅ |
    | 域分解 | [XPINN](./examples/xpinns.md) | ✅ | | ✅ | ✅ |
    | 布鲁塞尔扩散系统 | [3D-Brusselator](./examples/brusselator3d.md) | ✅ | | | |
    | 符号回归 | [Transformer4SR](./examples/transformer4sr.md) | ✅ | | | |

=== "技术科学(AI for Technology)"

    | 问题类型 | 案例名称 | NVIDIA | 海光 | 太初 | 沐曦 |
    |-----|-----|-----|-----|-----|-----|
    | 汽车表面阻力预测 | [DrivAerNet](./examples/drivaernet.md) | ✅ | | | ✅ |
    | 一维线性对流问题 | [1D 线性对流](./examples/adv_cvit.md) | ✅ | | | ✅ |
    | 非定常不可压流体 | [2D 方腔浮力驱动流](./examples/ns_cvit.md) | ✅ | | | ✅ |
    | 定常不可压流体 | [Re3200 2D 定常方腔流](./examples/ldc2d_steady.md) | ✅ | | | ✅ |
    | 定常不可压流体 | [2D 达西流](./examples/darcy2d.md) | ✅ | | ✅ | ✅ |
    | 定常不可压流体 | [2D 管道流](./examples/labelfree_DNN_surrogate.md) | ✅ | | | ✅ |
    | 定常不可压流体 | [3D 颅内动脉瘤](./examples/aneurysm.md) | ✅ | | | ✅ |
    | 定常不可压流体 | [任意 2D 几何体绕流](./examples/deepcfd.md) | ✅ | | ✅ | ✅ |
    | 非定常不可压流体 | [2D 非定常方腔流](./examples/ldc2d_unsteady.md) | ✅ | | ✅ | ✅ |
    | 非定常不可压流体 | [Re100 2D 圆柱绕流](./examples/cylinder2d_unsteady.md) | ✅ | | | ✅ |
    | 非定常不可压流体 | [Re100~750 2D 圆柱绕流](./examples/cylinder2d_unsteady_transformer_physx.md) | ✅ | | | ✅ |
    | 可压缩流体 | [2D 空气激波](./examples/shock_wave.md) | ✅ | | | ✅ |
    | 通用流场模拟 | [气动外形设计](./examples/amgnet.md) | ✅ | | | |
    | 流固耦合 | [涡激振动](./examples/viv.md) | ✅ | | | ✅ |
    | 多相流 | [气液两相流](./examples/bubble.md) | ✅ | | | ✅ |
    | 流场高分辨率重构 | [2D 湍流流场重构](./examples/tempoGAN.md) | ✅ | | | ✅ |
    | 求解器耦合 | [CFD-GCN](./examples/cfdgcn.md) | ✅ | | | ✅ |
    | 受力分析 | [1D 欧拉梁变形](./examples/euler_beam.md) | ✅ | | ✅ | ✅ |
    | 受力分析 | [2D 平板变形](./examples/biharmonic2d.md) | ✅ | | | ✅ |
    | 受力分析 | [3D 连接件变形](./examples/bracket.md) | ✅ | | | ✅ |
    | 受力分析 | [结构震动模拟](./examples/phylstm.md) | ✅ | | | ✅ |
    | 受力分析 | [2D 弹塑性结构](./examples/epnn.md) | ✅ | | | ✅ |
    | 受力分析和逆问题 | [3D 汽车控制臂变形](./examples/control_arm.md) | ✅ | | | ✅ |
    | 受力分析和逆问题 | [3D 心脏仿真](./examples/heart.md) | ✅ | | | ✅ |
    | 拓扑优化 | [2D 拓扑优化](./examples/topopt.md) | ✅ | | | ✅ |
    | 热仿真 | [1D 换热器热仿真](./examples/heat_exchanger.md) | ✅ | | | ✅ |
    | 热仿真 | [2D 热仿真](./examples/heat_pinn.md) | ✅ | | | ✅ |
    | 热仿真 | [2D 芯片热仿真](./examples/chip_heat.md) | ✅ | | | |
    | 算子学习 | [NeuralOperator](./examples/neuraloperator.md) | ✅ | | | |
    | 汽车表面阻力预测 | [DrivAerNetPlusPlus](./examples/drivaernetplusplus.md) | ✅ | | | |
    | 求解器耦合 | [NSFNets](./examples/nsfnet.md) | ✅ | | | ✅ |
    | 流场高分辨率重构 | [PhyCRNet](./examples/phycrnet.md) | ✅ | | | |
    | 求解器耦合 | [NSFNet4](./examples/nsfnet4.md) | ✅ | | | |

=== "材料科学(AI for Material)"

    | 问题类型 | 案例名称 | NVIDIA | 海光 | 太初 | 沐曦 |
    |-----|-----|-----|-----|-----|-----|
    | 材料设计 | [散射板设计(反问题)](./examples/hpinns.md) | ✅ | | ✅ | ✅ |
    | 材料设计 | [CGCNN](./examples/cgcnn.md) | ✅ | | ✅ | |

=== "地球科学(AI for Earth Science)"

    | 问题类型 | 案例名称 | NVIDIA | 海光 | 太初 | 沐曦 |
    |-----|-----|-----|-----|-----|-----|
    | 天气预报 | [Extformer-MoE 气象预报](./examples/extformer_moe.md) | ✅ | | ✅ | ✅ |
    | 天气预报 | [FourCastNet 气象预报](./examples/fourcastnet.md) | ✅ | | ✅ | |
    | 天气预报 | [NowCastNet 气象预报](./examples/nowcastnet.md) | ✅ | | | ✅ |
    | 天气预报 | [GraphCast 气象预报](./examples/graphcast.md) | ✅ | | ✅ | ✅ |
    | 天气预报 | [DGMR 气象预报](./examples/dgmr.md) | ✅ | | | |
    | 地震波形反演 | [VelocityGAN 地震波形反演](./examples/velocity_gan.md) | ✅ | | | ✅ |
    | 交通预测 | [TGCN 交通流量预测](./examples/tgcn.md) | ✅ | | | ✅ |
    | 天气预报 | [EarthFormer 气象预报](./examples/earthformer.md) | ✅ | | | ✅ |
    | 交通预测 | [IOPS 交通流量预测](./examples/iops.md) | ✅ | | | ✅ |
    | 天气预报 | [Pang-Weather 气象预报](./examples/pangu_weather.md) | ✅ | | |  ✅|
    | 天气预报 | [FengWu 气象预报](./examples/fengwu.md) | ✅ | | | ✅ |

=== "化学科学 (AI for Chemistry)"

    | 问题类型 | 案例名称 | NVIDIA | 海光 | 太初 | 沐曦 |
    |-----|-----|-----|-----|-----|-----|
    | 化学分子生成 | [Moflow](./examples/moflow.md) | ✅ | | | ✅ |
    | 化学反应预测 | [IFM](./examples/ifm.md) | ✅ | | | ✅ |

## 2. 运行指南

针对 PaddleScience 已支持的硬件，我们为每个硬件提供了一个运行示例，以[1D 欧拉梁变形](./examples/euler_beam.md)为例。

!!! note

    请确保你已经在你的环境中正确安装了计算硬件对应的 PaddlePaddle，否则请参考 [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice)，将你的硬件代码接入到飞桨中。

=== "NVIDIA"

    ``` sh
    # 安装 PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # 若 github clone 速度比较慢，可以使用 gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "模型训练命令"

        ``` sh
        python euler_beam.py
        ```

    === "模型评估命令"

        ``` sh
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "模型导出命令"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "模型推理命令"

        ``` sh
        python euler_beam.py mode=infer
        ```

=== "海光"

    ``` sh
    # 安装 PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # 若 github clone 速度比较慢，可以使用 gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "模型训练命令"

        ``` sh
        python euler_beam.py
        ```

    === "模型评估命令"

        ``` sh
        # 测试自己训练的模型
        python euler_beam.py mode=eval EVAL.pretrained_model_path=$YOUR_MODEL_PATH
        # 测试官方提供的预训练模型
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "模型导出命令"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "模型推理命令"

        ``` sh
        python euler_beam.py mode=infer
        ```

=== "太初"

    ``` sh
    # 安装 PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # 若 github clone 速度比较慢，可以使用 gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "模型训练命令"

        ``` sh
        python euler_beam.py
        ```

    === "模型评估命令"

        ``` sh
        # 测试自己训练的模型
        python euler_beam.py mode=eval EVAL.pretrained_model_path=$YOUR_MODEL_PATH
        # 测试官方提供的预训练模型
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "模型导出命令"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "模型推理命令"

        ``` sh
        python euler_beam.py mode=infer INFER.device=sdaa
        ```

=== "沐曦"

    ``` sh
    # 安装 PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # 若 github clone 速度比较慢，可以使用 gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "模型训练命令"

        ``` sh
        python euler_beam.py
        ```

    === "模型评估命令"

        ``` sh
        # 测试自己训练的模型
        python euler_beam.py mode=eval EVAL.pretrained_model_path=$YOUR_MODEL_PATH
        # 测试官方提供的预训练模型
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "模型导出命令"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "模型推理命令"

        ``` sh
        python euler_beam.py mode=infer
        ```

## 3. 贡献指南

我们在公开的案例文档开头提供了基于 NVIDIA CUDA 训练的参考精度和对应的预训练模型权重，如果需要在指定的硬件上运行，可以参考如下步骤：

1. 如果你的硬件类型尚未接入 PaddlePaddle，则可以参考 [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice) 官方文档，接入飞桨框架。如果你的硬件类型已接入 PaddlePaddle，但尚未添加到 PaddleScience 的硬件支持列表中，请在 [ppsci/utils/config.py](https://github.com/PaddlePaddle/PaddleScience/blob/develop/ppsci/utils/config.py#L215) 和 [deploy/python_infer/base.py](https://github.com/PaddlePaddle/PaddleScience/blob/develop/deploy/python_infer/base.py#L217) 中添加你的硬件类型。

2. 按照案例文档给出的步骤，准备好必要的数据集。

3. 如果模型文档中提供了模型训练命令，则需要在你的硬件上进行全量训练，保存训练日志，记录最佳模型精度以及最佳模型权重，这些内容一般会在训练过程中，自动保存在案例文件夹下。

4. 如果模型文档中提供了模型评估命令，则需要在你的硬件上对第三步所保存的最佳模型进行精度评估，保存评估日志，记录评估精度，这些内容一般会在评估过程中，自动保存在案例文件夹下。

    !!! note

        对于模型全量训练精度，默认要求最佳精度与 NVIDIA CUDA 精度对齐。具体地，如果案例精度指标为相对误差(如 L2 相对误差)，则指标不能超过参考值 ± 0.5%，如果案例精度指标为 MSE/MAE 一类的误差，则与参考值应保持在同一量级。

5. 如果模型文档中提供了模型导出和推理命令，请按照模型导出和推理命令，验证在新硬件上模型导出和推理是否能够正常执行并对齐 CUDA 的推理结果。

6. 上述步骤完成后，可以在 [1. 硬件支持列表](#1) 的表格中，给对应模型添加你的硬件支持信息(✅)，然后提交 PR 到 PaddleScience。你的 PR 应该至少包括以下内容：
    * 在 [2. 运行指南](#2) 中添加基于你的硬件环境使用模型的运行说明文档
    * 训练保存的最佳模型权重文件(`.pdparams` 文件)
    * 训练/评估等运行日志(`.log` 文件)
    * 验证模型精度所用到的软件版本，包括但不限于：
        * PaddlePaddle 版本
        * PaddleCustomDevice 版本(如果有)
    * 验证模型精度所用到的机器环境，包括但不限于：
        * 芯片型号
        * 系统版本
        * 硬件驱动版本
        * 算子库版本等

## 4. 更多文档

更多关于飞桨多硬件适配和使用的相关文档，可以参考：

* [飞桨使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/index_cn.html)
* [飞桨硬件支持](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/index_cn.html)
* [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice)
