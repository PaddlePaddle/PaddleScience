# FunDiff

<!-- <a href="https://aistudio.baidu.com/projectdetail/7927786" class="md-button md-button--primary" style>AI Studio快速体验</a> -->

!!! warning

    本文档仅复现了 Fundiff 论文中的 turbulence_mass_transfer 任务

!!! note

    请先在 <https://drive.google.com/drive/folders/1GX5uG_3R-yfuP9nMIk0v7ChuEytYwYPW?usp=drive_link> 中下载 tmt.npy 数据集文件

=== "模型训练命令"

    ``` sh
    python main.py -cn fae.yaml

    python main.py -cn diffusion.yaml FAE.pretrained_model_path=/your/fae/pretrained/model/path
    ```

=== "模型评估命令"

    ``` sh
    python main.py -cn diffusion.yaml mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/fundiff/fundiff_turbulence_mass_transfer_dit_pretrained.pdparams
    ```

<!-- === "模型导出命令"

    暂无

=== "模型推理命令"

    暂无 -->

| 预训练模型  | 指标 |
|:--| :--|
| [fundiff_turbulence_mass_transfer_dit_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/fundiff/fundiff_turbulence_mass_transfer_dit_pretrained.pdparams) | Mean relative p error: 0.066<br>Max relative p error: 0.159<br>Min relative p error: 0.029<br>Std relative p error: 0.027<br>Mean relative sdf error: 0.085<br>Max relative sdf error: 0.307<br>Min relative sdf error: 0.022<br>Std relative sdf error: 0.0499 |

## 1. 背景简介

生成模型（尤其是扩散模型和流匹配）的最新进展在图像和视频等离散数据的合成方面取得了显著成功。然而，将这些模型应用于物理应用仍然具有挑战性，因为感兴趣的物理量是由复杂物理定律支配的连续函数。本文介绍了 $\textbf{FunDiff}$，这是一个用于函数空间生成模型的全新框架。FunDiff 将潜在扩散过程与函数自编码器架构相结合，以处理具有不同离散化程度的输入函数，生成可在任意位置求值的连续函数，并无缝地集成物理先验。这些先验通过架构约束或基于物理的损失函数来强制执行，从而确保生成的样本满足基本物理定律。作者从理论上建立了函数空间中密度估计的极小极大最优性保证，表明基于扩散的估计器在合适的正则条件下能够实现最佳收敛速度。结果中展示了 FunDiff 在流体动力学和固体力学等各种应用中的实际有效性。实证结果表明，作者的方法能够生成物理上一致的样本，与目标分布高度一致，并且对噪声数据和低分辨率数据表现出鲁棒性。

## 2. 问题定义

给定已知物理场 $u$ 和 $v$，求解物理场 $p$ 以及 $sdf$。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 训练 FAE

#### 3.1.1 FAE 模型构建

在 FuncDiff 模型中，FAE 模块采用了 Perceiver 的架构，其输入为物理场 $x$ 和查询坐标 $coords$，输出是某个物理场在查询坐标上的值 $u$，因此模型构建代码如下

``` yaml linenums="36" title="fae.yaml"
--8<--
examples/fundiff/conf/fae.yaml:36:63
--8<--
```

``` py linenums="87" title="main.py"
--8<--
examples/fundiff/main.py:87:95
--8<--
```

#### 3.1.2 约束构建

FAE 使用 auto encoder decoder 的训练范式，因此标签即是输入 $u$

``` py linenums="97" title="main.py"
--8<--
examples/fundiff/main.py:97:148
--8<--
```

### 3.2 训练 DiT

#### 3.2.1 DiT 模型构建

DiT 的模型构建如下

``` yaml linenums="66" title="diffusion.yaml"
--8<--
examples/fundiff/conf/diffusion.yaml:66:76
--8<--
```

``` py linenums="186" title="main.py"
--8<--
examples/fundiff/main.py:186:208
--8<--
```

#### 3.2.2 约束构建

在 FuncDiff 模型中，DiT 的训练使用 rectified flow 算法，其对应的数学公式如下：

$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z}, t, {\epsilon}} \left[ \left\| \hat{\mathbf{v}}_\theta(\mathbf{x}, t) - (\mathbf{z} - \mathbf{x}) \right\|^2 \right]
$$

其对应的前向计算实现代码如下

``` py linenums="44" title="main.py"
--8<--
examples/fundiff/main.py:44:85
--8<--
```

整体约束构建如下

``` py linenums="210" title="main.py"
--8<--
examples/fundiff/main.py:210:262
--8<--
```

### 3.3 超参数设定

FAE 使用 100000 步训练步数，0.001 的初始学习率。

``` yaml linenums="65" title="fae.yaml"
--8<--
examples/fundiff/conf/fae.yaml:65:80
--8<--
```

DiT 使用 100000 步训练步数，0.001 的初始学习率。

``` yaml linenums="78" title="diffusion.yaml"
--8<--
examples/fundiff/conf/diffusion.yaml:78:92
--8<--
```

### 3.4 优化器构建

训练过程会调用优化器来更新模型参数，FAE 和 DiT 均选择较为常用的 `Adam` 优化器，并配合使用机器学习中常用的 ExponentialDecay 学习率调整策略。

``` yaml linenums="65" title="fae.yaml"
--8<--
examples/fundiff/conf/fae.yaml:65:80
--8<--
```

``` py linenums="159" title="main.py"
--8<--
examples/fundiff/main.py:159:172
--8<--
```

``` yaml linenums="78" title="diffusion.yaml"
--8<--
examples/fundiff/conf/diffusion.yaml:78:104
--8<--
```

``` py linenums="273" title="main.py"
--8<--
examples/fundiff/main.py:273:286
--8<--
```

### 3.5 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练即可。

``` py linenums="174"
--8<--
examples/fundiff/main.py:174:182
--8<--
```

``` py linenums="288"
--8<--
examples/fundiff/main.py:288:296
--8<--
```

## 4. 完整代码

``` py linenums="1" title="main.py"
--8<--
examples/fundiff/main.py
--8<--
```

## 5. 结果展示

在测试集上进行评估，并对部分结果进行展示

<figure markdown>
  ![result_of_sample_2.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/fundiff/result_of_sample_2.png){ loading=lazy }
</figure>

可以看到对于函数$p(x, coord | u,v)$ 和$sdf(x, coord | u,v)$，模型的预测结果和参考结果基本一致。

## 6. 参考资料

- [FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling](https://arxiv.org/abs/2506.07902v1)
- [fundiff github](https://github.com/sifanexisted/fundiff/tree/main)
