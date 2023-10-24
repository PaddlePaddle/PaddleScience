# PaddleScience

--8<--
./README.md:status
--8<--

--8<--
./README.md:description
--8<--

## 📝案例列表

| 问题类型 | 案例名称 | 优化算法 | 模型类型 | 训练方式 | 数据集 | 参考资料 |
|-----|---------|-----|---------|----|---------|---------|
| 微分方程 | [拉普拉斯方程](./zh/examples/laplace2d.md) | 机理驱动 | MLP | 无监督学习 | -        | - |
| 微分方程 | [伯格斯方程](./zh/examples/deephpms.md) | 机理驱动 | DeepHPMs | 无监督学习 | [Data](https://github.com/maziarraissi/DeepHPMs/tree/master/Data) | [Paper](https://arxiv.org/pdf/1801.06637.pdf) |
| 微分方程 | [洛伦兹方程](./zh/examples/lorenz.md) | 数据驱动 | Transformer-Physx | 监督学习 | - | [Paper](https://arxiv.org/abs/2010.03957) |
| 微分方程 | [若斯叻方程](./zh/examples/rossler.md) | 数据驱动 | Transformer-Physx | 监督学习 | - | [Paper](https://arxiv.org/abs/2010.03957) |
| 算子学习 | [DeepONet](./zh/examples/deeponet.md) | 数据驱动 | MLP | 监督学习 | [Turorial](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html) | [Paper](https://export.arxiv.org/pdf/1910.03193.pdf) |
| 微分方程 | 梯度增强的物理知识融合PDE求解<sup>coming soon</sup> | 机理驱动 | gPINN | 半监督学习 | - |  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782522001438?via%3Dihub) |
| 积分方程 | [沃尔泰拉积分方程](./zh/examples/volterra_ide.md) | 机理驱动 | MLP | 无监督学习 | - | [Project](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py) |
| 定常不可压流体 | [2D 定常方腔流](./zh/examples/ldc2d_steady.md) | 机理驱动 | MLP | 无监督学习 | - |  |
| 定常不可压流体 | [2D 达西流](./zh/examples/darcy2d.md) | 机理驱动 | MLP | 无监督学习 | - |   |
| 定常不可压流体 | [2D 管道流](./zh/examples/labelfree_DNN_surrogate.md) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/1906.02382) |
| 定常不可压流体 | [3D 血管瘤](./zh/examples/aneurysm.md) | 机理驱动 | MLP | 无监督学习 | - | [Project](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)|
| 定常不可压流体 | [任意 2D 几何体绕流](./zh/examples/deepcfd.md) | 数据驱动 | DeepCFD | 有监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [2D 非定常方腔流](./zh/examples/ldc2d_unsteady.md) | 机理驱动 | MLP | 无监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [Re100 2D 圆柱绕流](./zh/examples/cylinder2d_unsteady.md) | 机理驱动 | MLP | 半监督学习 | - | [Paper](https://arxiv.org/abs/2004.08826)|
| 非定常不可压流体 | [Re100~750 2D 圆柱绕流](./zh/examples/cylinder2d_unsteady_transformer_physx.md) | 数据驱动 | Transformer-Physx | 有监督学习 | - | [Paper](https://arxiv.org/abs/2010.03957)|
| 可压缩流体 | [2D 空气激波](./zh/examples/shock_wave.md) | 机理驱动 | PINN-WE | 无监督学习 | - | [Paper](https://arxiv.org/abs/2206.03864)|
| 流固耦合 | [涡激振动](./zh/examples/viv.md) | 机理驱动 | MLP | 半监督学习 | - | [Paper](https://arxiv.org/abs/2206.03864)|
| 多相流 | [气液两相流](./zh/examples/bubble.md) | 机理驱动 | BubbleNet | 半监督学习 | - | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 多相流 | 非高斯渗透率场估计<sup>coming soon</sup> | 机理驱动 | cINN | 有监督学习 | - | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
| 流场高分辨率重构 | 2D 湍流流场重构<sup>coming soon</sup> | 数据驱动 | tempoGAN | 有监督学习 | - | [Paper](https://dl.acm.org/doi/10.1145/3197517.3201304)|
| 流场高分辨率重构 | 基于扩散的流体超分重构<sup>coming soon</sup> | 数理融合 | DDPM | 有监督学习 | - | [Paper](https://www.sciencedirect.com/science/article/pii/S0021999123000670)|
| 受力分析 | 1D 欧拉梁变形 | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | 2D 平板变形<sup>coming soon</sup> | 机理驱动 | MLP | 无监督学习 | - | - |
| 受力分析 | [3D 连接件变形](./zh/examples/bracket.md) | 机理驱动 | MLP | 无监督学习 | - | [Tutorial](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html) |
| 受力分析 | [结构震动模拟](./zh/examples/phylstm.md) | 机理驱动 | PhyLSTM | 有监督学习 | - | [Paper](https://arxiv.org/abs/2002.10253) |
| 材料设计 | [散射板设计(反问题)](./zh/examples/hpinns.md) | 数理融合 | 数据驱动 | 有监督学习 | - | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |
| 材料生成 | 面向对称感知的周期性材料生成<sup>coming soon</sup> | 数据驱动 | SyMat | 有监督学习 | - | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |
| 天气预报 | [FourCastNet 气象预报](./zh/examples/fourcastnet.md) | 数据驱动 | FourCastNet | 有监督学习 | - | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 天气预报 | [GraphCast 气象预报]<sup>coming soon</sup> | 数据驱动 | GraphCastNet* | 有监督学习 | - | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
| 大气污染物 | [UNet 污染物扩散](./https://aistudio.baidu.com/projectdetail/5663515?channelType=0&channel=0) | 数据驱动 | UNet | 有监督学习 | - | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |

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
    <a href="https://github.com/AndPuQing"><img class="avatar" src="https://avatars.githubusercontent.com/AndPuQing" alt="avatar" /></a>
    <a href="https://github.com/MayYouBeProsperous"><img class="avatar" src="https://avatars.githubusercontent.com/MayYouBeProsperous" alt="avatar" /></a>
    <a href="https://github.com/yangguohao"><img class="avatar" src="https://avatars.githubusercontent.com/yangguohao" alt="avatar" /></a>
    <a href="https://github.com/mrcangye"><img class="avatar" src="https://avatars.githubusercontent.com/mrcangye" alt="avatar" /></a>
    <a href="https://github.com/jjyaoao"><img class="avatar" src="https://avatars.githubusercontent.com/jjyaoao" alt="avatar" /></a>
    <a href="https://github.com/jiamingkong"><img class="avatar" src="https://avatars.githubusercontent.com/jiamingkong" alt="avatar" /></a>
    <a href="https://github.com/Liyulingyue"><img class="avatar" src="https://avatars.githubusercontent.com/Liyulingyue" alt="avatar" /></a>
    <a href="https://github.com/XYM-1"><img class="avatar" src="https://avatars.githubusercontent.com/XYM-1" alt="avatar" /></a>
    <a href="https://github.com/xusuyong"><img class="avatar" src="https://avatars.githubusercontent.com/xusuyong" alt="avatar" /></a>
    <a href="https://github.com/NKNaN"><img class="avatar" src="https://avatars.githubusercontent.com/NKNaN" alt="avatar" /></a>
    <a href="https://github.com/ruoyunbai"><img class="avatar" src="https://avatars.githubusercontent.com/ruoyunbai" alt="avatar" /></a>
    <a href="https://github.com/sanbuphy"><img class="avatar" src="https://avatars.githubusercontent.com/sanbuphy" alt="avatar" /></a>
    <a href="https://github.com/ccsuzzh"><img class="avatar" src="https://avatars.githubusercontent.com/ccsuzzh" alt="avatar" /></a>
    <a href="https://github.com/enkilee"><img class="avatar" src="https://avatars.githubusercontent.com/enkilee" alt="avatar" /></a>
    <a href="https://github.com/GreatV"><img class="avatar" src="https://avatars.githubusercontent.com/GreatV" alt="avatar" /></a>

--8<--
./README.md:license
--8<--
