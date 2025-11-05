# Transolver

!!! note

    第一次运行时，会对 mlcfd_data 进行预处理，大约需要一小时，请耐心等待

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip -o mlcfd_data.zip
    unzip mlcfd_data.zip
    python main.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip -o mlcfd_data.zip
    unzip mlcfd_data.zip
    python main.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/transolver/transolver_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python main.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip -o mlcfd_data.zip
    unzip mlcfd_data.zip
    python main.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [transolver_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/transolver/transolver_pretrained.pdparams) | rho_d:, 0.99314<br>c_d: 0.01136<br>relative l2 error of press: 0.07829<br>relative l2 error of velocity: 0.02304<br>press: 4.95888<br>velocity: [0.12163974 0.14851639 0.41583335] 0.26443 |

## 1. 背景简介

Transolver 是一个基于 Transformer 架构的神经算子模型,用于学习偏微分方程(PDE)的解算子。该模型的核心创新在于其物理注意力机制(Physics Attention),能够高效地处理不规则网格上的物理场预测问题。

相比传统的 Transformer 模型,Transolver 具有以下特点:

- **物理感知的注意力机制**: 通过切片(slice)技术将不规则网格点聚合成规则的表示,既保留了空间物理信息,又大幅降低了计算复杂度
- **灵活的几何适应性**: 能够处理任意形状的几何体和非结构化网格
- **高效的计算性能**: 通过切片注意力机制,将计算复杂度从 $O(N^2)$ 降低到 $O(NG)$,其中 $N$ 是网格点数,$G$ 是切片数

本案例使用 Transolver 模型在 ShapeNet Car 数据集上学习汽车外流场的速度场和压力场分布,这是一个典型的计算流体动力学(CFD)代理建模问题。通过学习大量的汽车形状与对应的流场数据,模型可以快速预测新汽车形状的流场分布,从而大幅降低 CFD 仿真的计算成本。

## 2. 问题定义

本案例的目标是建立汽车几何形状到其周围流场(速度场和压力场)的映射关系。具体而言:

- **输入**: 汽车表面及周围空间的网格点坐标 $\mathbf{x} \in \mathbb{R}^{N \times 7}$,其中 $N$ 为网格点数量,7 个维度包括空间坐标、法向量等几何信息
- **输出**: 每个网格点处的速度矢量 $\mathbf{v} \in \mathbb{R}^{N \times 3}$ 和压力标量 $p \in \mathbb{R}^{N \times 1}$

训练目标是最小化预测流场与真实 CFD 仿真结果之间的均方误差,同时确保模型能够准确预测汽车的阻力系数等关键空气动力学参数。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码,用深度学习的方法求解该问题。
为了快速理解 PaddleScience,接下来仅对模型构建、约束构建、优化器构建等关键步骤进行阐述,而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在本问题中,需要建立从网格点坐标 $\mathbf{x}$ 到流场变量 $(\mathbf{v}, p)$ 的映射函数 $f: \mathbb{R}^{N \times 7} \to \mathbb{R}^{N \times 4}$,即:

$$
(\mathbf{v}, p) = f(\mathbf{x})
$$

这里使用 Transolver 模型来表示这个映射函数,用 PaddleScience 代码表示如下:

``` py linenums="25"
--8<--
examples/transolver/main.py:25:27
--8<--
```

为了在计算时准确快速地访问具体变量的值,这里指定网络模型的输入变量名是 `["x"]`,输出变量名是 `["velo_vec", "press"]`,这些命名与后续代码保持一致。

模型的详细配置如下:

``` yaml linenums="43"
--8<--
examples/transolver/conf/shapenet_car.yaml:43:57
--8<--
```

其中:

- `space_dim`: 输入空间维度,设为 7(包括空间坐标、法向量等几何信息)
- `n_layers`: Transformer 层数,设为 8
- `n_hidden`: 隐藏层维度,设为 256
- `n_head`: 多头注意力的头数,设为 8
- `dropout`: Dropout 比率,设为 0
- `act`: 激活函数,使用 `gelu`
- `mlp_ratio`: MLP 层的隐藏层扩展比例,设为 2
- `out_dim`: 输出维度列表,`[3, 1]` 分别对应速度场(3维)和压力场(1维)
- `slice_num`: 切片注意力机制中的切片数量,设为 32,用于降低计算复杂度
- `ref`: 参考网格的分辨率,设为 8
- `unified_pos`: 是否使用统一的位置编码,设为 False

### 3.2 模型架构详解

Transolver 的核心架构包括以下几个关键组件:

#### 3.2.1 预处理层 (Preprocessing)

将输入的几何特征映射到隐藏空间:

$$
h = \text{MLP}(\text{concat}(f, x))
$$

其中 $f$ 是函数特征(如初始场),$x$ 是空间坐标。

#### 3.2.2 物理注意力机制 (Physics Attention)

这是 Transolver 的核心创新,通过以下步骤实现高效的全局信息交互:

1. **切片聚合**: 将 $N$ 个不规则网格点聚合成 $G$ 个切片表示

$$
S = \text{softmax}\left(\frac{W_s(h)}{\tau}\right)^T h
$$

其中 $S \in \mathbb{R}^{G \times D}$ 是切片表示,$\tau$ 是可学习的温度参数。

2. **切片上的自注意力**: 在切片表示上进行标准的 Transformer 注意力计算

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中 $Q = W_q S, K = W_k S, V = W_v S$。

3. **反聚合**: 将切片表示映射回原始网格点

$$
h' = S \cdot \text{Attn}(Q, K, V) \cdot W
$$

通过这种机制,计算复杂度从 $O(N^2)$ 降低到 $O(NG + G^2)$,当 $G \ll N$ 时可显著提升效率。

#### 3.2.3 前馈网络 (MLP)

每个 Transformer 块后接一个前馈网络:

$$
\text{FFN}(h) = W_2 \cdot \text{GELU}(W_1 h + b_1) + b_2
$$

#### 3.2.4 层归一化与残差连接

每个子层都使用层归一化(LayerNorm)和残差连接:

$$
\begin{aligned}
h^{(l+1)} &= \text{Attn}(\text{LN}(h^{(l)})) + h^{(l)} \\
h^{(l+1)} &= \text{FFN}(\text{LN}(h^{(l+1)})) + h^{(l+1)}
\end{aligned}
$$

### 3.3 数据加载

本案例使用 ShapeNet Car 数据集,包含不同汽车形状的 CFD 仿真结果。数据加载代码如下:

``` py linenums="29"
--8<--
examples/transolver/main.py:29:43
--8<--
```

数据集配置:

``` yaml linenums="31"
--8<--
examples/transolver/conf/shapenet_car.yaml:31:40
--8<--
```

其中:

- `data_dir`: 原始数据目录
- `save_dir`: 预处理后数据保存目录
- `val_fold_id`: 用于交叉验证的折数 ID
- `preprocessed`: 是否使用预处理数据
- `r`: 数据下采样比例

### 3.4 约束构建

本案例使用监督学习约束,通过最小化预测流场与真实流场的误差来训练模型:

``` py linenums="45"
--8<--
examples/transolver/main.py:45:76
--8<--
```

损失函数包含两部分:

1. 速度场的均方误差: $\mathcal{L}_{velo} = \text{MSE}(\mathbf{v}_{pred}, \mathbf{v}_{true})$
2. 表面压力场的均方误差: $\mathcal{L}_{press} = \text{MSE}(p_{pred}|_{surf}, p_{true}|_{surf})$

总损失为: $\mathcal{L} = \mathcal{L}_{velo} + w_{press} \cdot \mathcal{L}_{press}$,其中 $w_{press}$ 为压力损失权重。

### 3.5 优化器构建

使用 Adam 优化器配合指数衰减学习率策略:

``` py linenums="78"
--8<--
examples/transolver/main.py:78:89
--8<--
```

学习率配置:

``` yaml linenums="63"
--8<--
examples/transolver/conf/shapenet_car.yaml:63:67
--8<--
```

其中包含:

- 预热阶段(`warmup_epoch`): 前 60 个 epoch 学习率从 $\frac{lr_{max}}{25}$ 逐渐增加到 $lr_{max}$
- 衰减阶段: 之后每个 step 学习率按 $lr = lr_{max} \times \gamma^{step}$ 衰减

### 3.6 评估器构建

在训练过程中使用验证集评估模型性能:

``` py linenums="91"
--8<--
examples/transolver/main.py:91:129
--8<--
```

评估指标包括:

- 速度场的均方误差
- 表面压力场的均方误差

### 3.7 模型训练与评估

完成上述设置后,将实例化的对象传递给 `ppsci.solver.Solver`,然后启动训练和评估:

``` py linenums="140"
--8<--
examples/transolver/main.py:140:142
--8<--
```

### 3.8 模型评估

评估阶段除了计算常规的误差指标外,还会计算阻力系数的预测误差和斯皮尔曼相关系数:

``` py linenums="145"
--8<--
examples/transolver/main.py:145:241
--8<--
```

评估指标包括:

- **相对 L2 误差**: 衡量预测场与真实场的整体偏差
- **均方根误差 (RMSE)**: 评估逐点预测精度
- **阻力系数误差**: 评估关键空气动力学参数的预测精度
- **斯皮尔曼相关系数 ($\rho_d$)**: 评估阻力系数排序的相关性

## 4. 完整代码

``` py linenums="1" title="main.py"
--8<--
examples/transolver/main.py
--8<--
```

## 5. 结果展示

模型训练完成后,可以在验证集上评估预测性能。主要评估指标包括:

- **速度场相对 L2 误差**: 衡量速度场预测的整体精度
- **压力场相对 L2 误差**: 衡量压力场预测的整体精度
- **阻力系数相对误差**: 评估关键气动参数的预测准确性
- **斯皮尔曼相关系数**: 评估不同汽车形状阻力系数排序的准确性

通过训练,Transolver 模型能够以较高精度预测汽车周围的流场分布,相比传统 CFD 仿真可以大幅降低计算时间,为汽车气动外形优化提供快速的代理模型。

## 6. 参考资料

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)
- [Transolver GitHub Repository](https://github.com/thuml/Transolver)
