# 多硬件支持

飞桨生态的繁荣离不开开发者和用户的贡献，我们非常欢迎为飞桨的多硬件适配贡献更多的模型。

## 1. 硬件支持列表

当前 PaddleScience 中对于各硬件适配模型的列表汇总如下

<p align="center"><b>数学(AI for Math)</b></p>

| 问题类型 | 案例名称 | 昆仑芯 | 海光 | 寒武纪 | 昇腾 | 燧原 | 天数 | 摩尔线程 | 沐曦 | 太初 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 亥姆霍兹方程 | [SPINN(Helmholtz3D)](./examples/spinn.md) | | | | | | | | | |
| 相场方程 | [Allen-Cahn](./examples/allen_cahn.md) | | | | | | | | | |
| 微分方程 | [拉普拉斯方程](./examples/laplace2d.md) | | | | | | | | | |
| 微分方程 | [伯格斯方程](./examples/deephpms.md) | | | | | | | | | |
| 微分方程 | [非线性偏微分方程](./examples/pirbn.md) | | | | | | | | | |
| 微分方程 | [洛伦兹方程](./examples/lorenz.md) | | | | | | | | | |
| 微分方程 | [若斯叻方程](./examples/rossler.md) | | | | | | | | | |
| 算子学习 | [DeepONet](./examples/deeponet.md) | | | | | | | | | |
| 微分方程 | [梯度增强的物理知识融合 PDE 求解](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/gpinn/poisson_1d.py) | | | | | | | | | |
| 积分方程 | [沃尔泰拉积分方程](./examples/volterra_ide.md) | | | | | | | | | |
| 微分方程 | [分数阶微分方程](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fpde/fractional_poisson_2d.py) | | | | | | | | | |
| 光纤怪波 | [Optical rogue wave](./examples/nlsmb.md) | | | | | | | | | |
| 域分解 | [XPINN](./examples/xpinns.md) | | | | | | | | | |
| 布鲁塞尔扩散系统 | [3D-Brusselator](./examples/brusselator3d.md) | | | | | | | | | |
| 符号回归 | [Transformer4SR](./examples/transformer4sr.md) | | | | | | | | | |

<br>
<p align="center"><b>技术科学(AI for Technology)</b></p>

| 问题类型 | 案例名称 | 昆仑芯 | 海光 | 寒武纪 | 昇腾 | 燧原 | 天数 | 摩尔线程 | 沐曦 | 太初 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 汽车表面阻力预测 | [DrivAerNet](./examples/drivaernet.md) | | | | | | | | | |
| 一维线性对流问题 | [1D 线性对流](./examples/adv_cvit.md) | | | | | | | | | |
| 非定常不可压流体 | [2D 方腔浮力驱动流](./examples/ns_cvit.md) | | | | | | | | | |
| 定常不可压流体 | [Re3200 2D 定常方腔流](./examples/ldc2d_steady.md) | | | | | | | | | |
| 定常不可压流体 | [2D 达西流](./examples/darcy2d.md) | | | | | | | | | |
| 定常不可压流体 | [2D 管道流](./examples/labelfree_DNN_surrogate.md) | | | | | | | | | |
| 定常不可压流体 | [3D 颅内动脉瘤](./examples/aneurysm.md) | | | | | | | | | |
| 定常不可压流体 | [任意 2D 几何体绕流](./examples/deepcfd.md) | | | | | | | | | |
| 非定常不可压流体 | [2D 非定常方腔流](./examples/ldc2d_unsteady.md) | | | | | | | | | |
| 非定常不可压流体 | [Re100 2D 圆柱绕流](./examples/cylinder2d_unsteady.md) | | | | | | | | | |
| 非定常不可压流体 | [Re100~750 2D 圆柱绕流](./examples/cylinder2d_unsteady_transformer_physx.md) | | | | | | | | | |
| 可压缩流体 | [2D 空气激波](./examples/shock_wave.md) | | | | | | | | | |
| 飞行器设计 | [MeshGraphNets](https://aistudio.baidu.com/projectdetail/5322713) | | | | | | | | | |
| 飞行器设计 | [火箭发动机真空羽流](https://aistudio.baidu.com/projectdetail/4486133) | | | | | | | | | |
| 飞行器设计 | [Deep-Flow-Prediction](https://aistudio.baidu.com/projectdetail/5671596) | | | | | | | | | |
| 通用流场模拟 | [气动外形设计](./examples/amgnet.md) | | | | | | | | | |
| 流固耦合 | [涡激振动](./examples/viv.md) | | | | | | | | | |
| 多相流 | [气液两相流](./examples/bubble.md) | | | | | | | | | |
| 多相流 | [twophasePINN](https://aistudio.baidu.com/projectdetail/5379212)  | | | | | | | | | |
| 流场高分辨率重构 | [2D 湍流流场重构](./examples/tempoGAN.md) | | | | | | | | | |
| 流场高分辨率重构 | [2D 湍流流场重构](https://aistudio.baidu.com/projectdetail/4493261?contributionType=1) | | | | | | | | | |
| 流场高分辨率重构 | [基于Voronoi嵌入辅助深度学习的稀疏传感器全局场重建](https://aistudio.baidu.com/projectdetail/5807904) | | | | | | | | | |
| 流场预测 | [Catheter](https://aistudio.baidu.com/projectdetail/5379212)  | | | | | | | | | |
| 求解器耦合 | [CFD-GCN](./examples/cfdgcn.md) | | | | | | | | | |
| 受力分析 | [1D 欧拉梁变形](./examples/euler_beam.md) | | | | | | | | | |
| 受力分析 | [2D 平板变形](./examples/biharmonic2d.md) | | | | | | | | | |
| 受力分析 | [3D 连接件变形](./examples/bracket.md) | | | | | | | | | |
| 受力分析 | [结构震动模拟](./examples/phylstm.md) | | | | | | | | | |
| 受力分析 | [2D 弹塑性结构](./examples/epnn.md) | | | | | | | | | |
| 受力分析和逆问题 | [3D 汽车控制臂变形](./examples/control_arm.md) | | | | | | | | | |
| 受力分析和逆问题 | [3D 心脏仿真](./examples/heart.md) | | | | | | | | | |
| 拓扑优化 | [2D 拓扑优化](./examples/topopt.md) | | | | | | | | | |
| 热仿真 | [1D 换热器热仿真](./examples/heat_exchanger.md) | | | | | | | | | |
| 热仿真 | [2D 热仿真](./examples/heat_pinn.md) | | | | | | | | | |
| 热仿真 | [2D 芯片热仿真](./examples/chip_heat.md) | | | | | | | | | |

<br>
<p align="center"><b>材料科学(AI for Material)</b></p>

| 问题类型 | 案例名称 | 昆仑芯 | 海光 | 寒武纪 | 昇腾 | 燧原 | 天数 | 摩尔线程 | 沐曦 | 太初 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 材料设计 | [散射板设计(反问题)](./examples/hpinns.md) | | | | | | | | | |

<br>
<p align="center"><b>地球科学(AI for Earth Science)</b></p>

| 问题类型 | 案例名称 | 昆仑芯 | 海光 | 寒武纪 | 昇腾 | 燧原 | 天数 | 摩尔线程 | 沐曦 | 太初 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 天气预报 | [Extformer-MoE 气象预报](./examples/extformer_moe.md) | | | | | | | | | |
| 天气预报 | [FourCastNet 气象预报](./examples/fourcastnet.md) | | | | | | | | | |
| 天气预报 | [NowCastNet 气象预报](./examples/nowcastnet.md) | | | | | | | | | |
| 天气预报 | [GraphCast 气象预报](./examples/graphcast.md) | | | | | | | | | |
| 大气污染物 | [UNet 污染物扩散](https://aistudio.baidu.com/projectdetail/5663515?channel=0&channelType=0&sUid=438690&shared=1&ts=1698221963752) | | | | | | | | | |
| 天气预报 | [DGMR 气象预报](./examples/dgmr.md) | | | | | | | | | |
| 地震波形反演 | [VelocityGAN 地震波形反演](./examples/velocity_gan.md) | | | | | | | | | |
| 交通预测 | [TGCN 交通流量预测](./examples/tgcn.md) | | | | | | | | | |

## 2. 贡献指南

我们在公开的案例文档开头提供了基于 GPU 训练的参考精度和对应的预训练模型权重，如果需要在指定的硬件上运行，可以参考如下步骤：

1. 在案例开头位置添加一行代码，将飞桨运行设备设置为当前硬件设备

    ``` py hl_lines="3"
    import paddle

    paddle.set_device("your_device_name")

    # 原案例代码
    ```

2. 按照案例文档步骤，准备好数据集，在指定硬件上进行全量训练，保存训练日志，记录最佳模型精度以及最佳模型权重，这些内容一般会在训练过程中，自动保存在案例文件夹下

3. 如果模型文档中提供了模型导出和推理命令，请按照模型导出和推理命令，验证在新硬件上模型导出和推理是否能够正常执行并对齐 GPU 的推理结果

4. 上述步骤完成后，可以在本文档(`docs/zh/multi_device.md`)的表格中，给对应模型在指定硬件上添加支持信息(✅)，然后提交 PR 到 PaddleScience

## 3. 更多文档

更多关于飞桨多硬件适配和使用的相关文档，可以参考：

* [飞桨使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/index_cn.html)
* [飞桨硬件支持](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/index_cn.html)
* [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice)
