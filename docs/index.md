# PaddleScience

--8<--
./README.md:status
--8<--

--8<--
./README.md:description
--8<--

## 📝案例列表

<style>
    table  th{
        background: #C1E6FE;
    }
</style>

<p align="center"><b>数学(AI for Math)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 微分方程 | [拉普拉斯方程](./zh/examples/laplace2d.md) | 机理驱动 | MLP | 无监督学习 | -        | - |
| 微分方程 | [伯格斯方程](./zh/examples/deephpms.md) | 机理驱动 | MLP | 无监督学习 | [Data](https://github.com/maziarraissi/DeepHPMs/tree/master/Data) | [Paper](https://arxiv.org/pdf/1801.06637.pdf) |
| 微分方程 | [洛伦兹方程](./zh/examples/lorenz.md) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
| 微分方程 | [若斯叻方程](./zh/examples/rossler.md) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
| 算子学习 | [DeepONet](./zh/examples/deeponet.md) | 数据驱动 | MLP | 监督学习 | [Data](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html) | [Paper](https://export.arxiv.org/pdf/1910.03193.pdf) |
| 微分方程 | 梯度增强的物理知识融合PDE求解<sup>coming soon</sup> | 机理驱动 | gPINN | 半监督学习 | - |  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782522001438?via%3Dihub) |
| 积分方程 | [沃尔泰拉积分方程](./zh/examples/volterra_ide.md) | 机理驱动 | MLP | 无监督学习 | - | [Project](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py) |

<br>
<p align="center"><b>技术科学(AI for Technology)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 定常不可压流体 | [2D 定常方腔流](./zh/examples/ldc2d_steady.md) | 机理驱动 | MLP | 无监督学习 | - |  |
| 定常不可压流体 | [2D 达西流](./zh/examples/darcy2d.md) | 机理驱动 | MLP | 无监督学习 | - |   |
| 定常不可压流体 | [2D 管道流](./zh/examples/labelfree_DNN_surrogate.md) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/1906.02382) |
| 定常不可压流体 | [3D 血管瘤](./zh/examples/aneurysm.md) | 机理驱动 | MLP | 无监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar) | [Project](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)|
| 定常不可压流体 | [任意 2D 几何体绕流](./zh/examples/deepcfd.md) | 数据驱动 | DeepCFD | 监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [2D 非定常方腔流](./zh/examples/ldc2d_unsteady.md) | 机理驱动 | MLP | 无监督学习 | - | -|
| 非定常不可压流体 | [Re100 2D 圆柱绕流](./zh/examples/cylinder2d_unsteady.md) | 机理驱动 | MLP | 半监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar) | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [Re100~750 2D 圆柱绕流](./zh/examples/cylinder2d_unsteady_transformer_physx.md) | 数据驱动 | Transformer-Physx | 监督学习 | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957)|
| 可压缩流体 | [2D 空气激波](./zh/examples/shock_wave.md) | 机理驱动 | PINN-WE | 无监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/167250) | -|
| 飞行器设计 | [MeshGraphNets](https://aistudio.baidu.com/projectdetail/5322713) | 数据驱动 | GNN | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/184320) | [Paper](https://arxiv.org/abs/2010.03409)|
| 飞行器设计 | [火箭发动机真空羽流](https://aistudio.baidu.com/projectdetail/4486133) | 数据驱动 | CNN | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/167250) | - |
| 飞行器设计 | [Deep-Flow-Prediction](https://aistudio.baidu.com/projectdetail/5671596) | 数据驱动 | TurbNetG | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/197778) | [Paper](https://arxiv.org/abs/1810.08217) |
| 流固耦合 | [涡激振动](./zh/examples/viv.md) | 机理驱动 | MLP | 半监督学习 | [Data](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fsi/VIV_Training_Neta100.mat) | [Paper](https://arxiv.org/abs/2206.03864)|
| 多相流 | [气液两相流](./zh/examples/bubble.md) | 机理驱动 | BubbleNet | 半监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat) | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 多相流 | [twophasePINN](https://aistudio.baidu.com/projectdetail/5379212) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://doi.org/10.1016/j.mlwa.2021.100029)|
| 多相流 | 非高斯渗透率场估计<sup>coming soon</sup> | 机理驱动 | cINN | 监督学习 | - | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 流场高分辨率重构 | [2D 湍流流场重构](./zh/examples/tempoGAN.md) | 数据驱动 | tempoGAN | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://dl.acm.org/doi/10.1145/3197517.3201304)|
| 流场高分辨率重构 | [2D 湍流流场重构](https://aistudio.baidu.com/projectdetail/4493261?contributionType=1) | 数据驱动 | cycleGAN | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://arxiv.org/abs/2007.15324)|
| 流场高分辨率重构 | [基于Voronoi嵌入辅助深度学习的稀疏传感器全局场重建](https://aistudio.baidu.com/projectdetail/5807904) | 数据驱动 | CNN | 监督学习 | [Data1](https://drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c)<br>[Data2](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)<br>[Data3](https://drive.google.com/drive/folders/1xIY_jIu-hNcRY-TTf4oYX1Xg4_fx8ZvD) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 流场高分辨率重构 | 基于扩散的流体超分重构<sup>coming soon</sup> | 数理融合 | DDPM | 监督学习 | - | [Paper](https://www.sciencedirect.com/science/article/pii/S0021999123000670)|
| 受力分析 | [1D 欧拉梁变形](../examples/euler_beam/euler_beam.py) | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | [2D 平板变形](https://aistudio.baidu.com/aistudio/projectdetail/5792325) | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | [3D 连接件变形](./zh/examples/bracket.md) | 机理驱动 | MLP | 无监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar) | [Tutorial](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html) |
| 受力分析 | [结构震动模拟](./zh/examples/phylstm.md) | 机理驱动 | PhyLSTM | 监督学习 | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat) | [Paper](https://arxiv.org/abs/2002.10253) |

<br>
<p align="center"><b>材料科学(AI for Material)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 材料设计 | [散射板设计(反问题)](./zh/examples/hpinns.md) | 数理融合 | 数据驱动 | 监督学习 | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat) | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |
| 材料生成 | 面向对称感知的周期性材料生成<sup>coming soon</sup> | 数据驱动 | SyMat | 监督学习 | - | - |

<br>
<p align="center"><b>地球科学(AI for Earth Science)</b></p>

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 天气预报 | [FourCastNet 气象预报](./zh/examples/fourcastnet.md) | 数据驱动 | FourCastNet | 监督学习 | [ERA5](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 天气预报 | GraphCast 气象预报<sup>coming soon</sup> | 数据驱动 | GraphCastNet* | 监督学习 | - | [Paper](https://arxiv.org/abs/2212.12794) |
| 大气污染物 | [UNet 污染物扩散](https://aistudio.baidu.com/projectdetail/5663515?channel=0&channelType=0&sUid=438690&shared=1&ts=1698221963752) | 数据驱动 | UNet | 监督学习 | [Data](https://aistudio.baidu.com/datasetdetail/198102) | - |

## 🚀快速安装

=== "方式1: 源码安装[推荐]"

    --8<--
    ./README.md:git_install
    --8<--

=== "方式2: pip安装"

    ``` shell
    pip install paddlesci
    ```

=== "[完整安装流程](./zh/install_setup.md)"

    ``` shell
    pip install paddlesci
    ```

    ``` shell
    pip install paddlesci
    ```

--8<--
./README.md:feature
--8<--

--8<--
./README.md:support
--8<--

--8<--
./README.md:contribution
--8<--

--8<--
./README.md:collaboration
--8<--

--8<--
./README.md:thanks
--8<--

- PaddleScience 的部分代码由以下优秀社区开发者贡献（按 [Contributors](https://github.com/PaddlePaddle/PaddleScience/graphs/contributors) 排序）：

    <style>
        .avatar {
            height: 64px;
            width: 64px;
            border: 2px solid rgba(128, 128, 128, 0.308);
            border-radius: 50%;
        }

        .avatar:hover {
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.4);
            transition: 0.4s;
            transform:translateY(-10px);
        }
    </style>
    <a href="https://github.com/Asthestarsfalll"><img class="avatar" src="https://avatars.githubusercontent.com/Asthestarsfalll" alt="avatar" /></a>
    <a href="https://github.com/co63oc"><img class="avatar" src="https://avatars.githubusercontent.com/co63oc" alt="avatar" /></a>
    <a href="https://github.com/MayYouBeProsperous"><img class="avatar" src="https://avatars.githubusercontent.com/MayYouBeProsperous" alt="avatar" /></a>
    <a href="https://github.com/AndPuQing"><img class="avatar" src="https://avatars.githubusercontent.com/AndPuQing" alt="avatar" /></a>
    <a href="https://github.com/lknt"><img class="avatar" src="https://avatars.githubusercontent.com/lknt" alt="avatar" /></a>
    <a href="https://github.com/yangguohao"><img class="avatar" src="https://avatars.githubusercontent.com/yangguohao" alt="avatar" /></a>
    <a href="https://github.com/mrcangye"><img class="avatar" src="https://avatars.githubusercontent.com/mrcangye" alt="avatar" /></a>
    <a href="https://github.com/jjyaoao"><img class="avatar" src="https://avatars.githubusercontent.com/jjyaoao" alt="avatar" /></a>
    <a href="https://github.com/jiamingkong"><img class="avatar" src="https://avatars.githubusercontent.com/jiamingkong" alt="avatar" /></a>
    <a href="https://github.com/Liyulingyue"><img class="avatar" src="https://avatars.githubusercontent.com/Liyulingyue" alt="avatar" /></a>
    <a href="https://github.com/Gxinhu"><img class="avatar" src="https://avatars.githubusercontent.com/Gxinhu" alt="avatar" /></a>
    <a href="https://github.com/XYM-1"><img class="avatar" src="https://avatars.githubusercontent.com/XYM-1" alt="avatar" /></a>
    <a href="https://github.com/xusuyong"><img class="avatar" src="https://avatars.githubusercontent.com/xusuyong" alt="avatar" /></a>
    <a href="https://github.com/NKNaN"><img class="avatar" src="https://avatars.githubusercontent.com/NKNaN" alt="avatar" /></a>
    <a href="https://github.com/ruoyunbai"><img class="avatar" src="https://avatars.githubusercontent.com/ruoyunbai" alt="avatar" /></a>
    <a href="https://github.com/sanbuphy"><img class="avatar" src="https://avatars.githubusercontent.com/sanbuphy" alt="avatar" /></a>
    <a href="https://github.com/ccsuzzh"><img class="avatar" src="https://avatars.githubusercontent.com/ccsuzzh" alt="avatar" /></a>
    <a href="https://github.com/enkilee"><img class="avatar" src="https://avatars.githubusercontent.com/enkilee" alt="avatar" /></a>
    <a href="https://github.com/GreatV"><img class="avatar" src="https://avatars.githubusercontent.com/GreatV" alt="avatar" /></a>

## 🤝合作单位

![cooperation](./images/overview/cooperation_thumbnail.png)

--8<--
./README.md:license
--8<--
