# Graph Networks for Physics Discovery (GN)

<a href="https://aistudio.baidu.com/projectdetail/9557878" class="md-button md-button--primary" style>AI Studio快速体验</a>

开始训练、评估前，请先确保已安装依赖 `scipy`、`scikit-learn`、`celluloid`、`pgl`，请运行安装命令：
```bash
pip install scipy scikit-learn celluloid pgl
```

=== "模型训练命令"

    ``` sh
    python gn.py mode=train MODEL.arch=OGN DATA.type=spring
    ```
=== "模型评估命令"

    ``` sh
    python gn.py mode=eval MODEL.arch=OGN EVAL.pretrained_model_path="./outputs_symbolic_gcn/.../checkpoints/latest"
    ```
=== "模型推理命令"

    ``` sh
    python gn.py mode=infer MODEL.arch=OGN INFER.pretrained_model_path="./outputs_symbolic_gcn/.../checkpoints/latest"
    ```
=== "模型导出命令"

    ``` sh
    python gn.py mode=export MODEL.arch=OGN INFER.pretrained_model_path="./outputs_symbolic_gcn/.../checkpoints/latest"
    ```

## 1. 背景简介

深度学习方法在物理系统建模中展现出强大能力，但传统神经网络往往是黑盒模型，难以提供物理解释。图神经网络（GNNs）通过其固有的归纳偏置，特别适合建模粒子系统中的相互作用。然而，如何从训练好的GNN中提取可解释的物理定律仍然是一个挑战。

基于论文 **[Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)** 的思想，我们开发了一个完整的物理发现框架。该框架通过在GNN训练过程中引入强归纳偏置（如L1正则化鼓励稀疏表示），然后对学习到的模型内部组件应用符号回归，从而提取显式的物理关系。这种方法不仅能够重新发现已知的力定律和哈密顿量，还能从复杂数据中发现新的解析公式。

本案例使用图网络（Graph Networks, GN）对多种物理系统（如弹簧系统、引力系统、电荷系统等）的动力学进行建模，并通过符号回归提取潜在的物理定律。

## 2. 模型原理

本章节基于论文 **[Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)** (NeurIPS 2020) 的核心思想进行介绍。

### 2.1 核心问题

**传统困境**：深度学习模型（如全连接网络、CNN）虽然强大，但缺乏物理解释性。我们能否既利用深度学习的拟合能力，又能提取可解释的物理定律？

**本文方案**：
1. 使用具有**归纳偏置**的架构（图网络）学习物理系统
2. 在训练中引入**稀疏性约束**，迫使模型学习简单表示
3. 对学到的内部函数进行**符号回归**，提取解析表达式
4. 用符号表达式替换神经网络组件，获得可解释模型

### 2.2 图网络架构

图网络（GN）是天然适合物理建模的架构，因为：
- **排列不变性**：粒子顺序不影响结果（符合物理对称性）
- **局部作用**：粒子间通过边交互（符合力的成对性）
- **可扩展性**：可以处理不同数量的粒子

GN的核心思想是将物理系统建模为图结构：
- **节点**：代表物理实体（如粒子），包含状态 xᵢ = [位置, 速度, 质量, 电荷]
- **边**：代表实体间的相互作用，通过消息传递机制计算
- **全局属性**：可选的系统级属性（如总能量）

GN内部包含三个可分离的函数组件：
- **边函数（消息函数）φₑ(xᵢ, xⱼ)**：计算从节点j到节点i的消息向量
- **节点函数 φᵥ(xᵢ, Σmᵢⱼ)**：根据聚合的消息更新节点状态
- **全局函数 φᵤ(Σxᵢ)**：计算系统的全局属性（如总能量）

**关键**：这些函数组件（φₑ, φᵥ, φᵤ）天然对应物理中的基本概念：
- φₑ 对应**成对作用力**（如引力、库仑力）
- φᵥ 对应**运动方程**（如牛顿第二定律）
- φᵤ 对应**能量函数**（如哈密顿量）

因此，如果我们能对这些函数进行符号回归，就能直接提取物理定律！

### 2.3 物理发现框架

我们的物理发现框架包含以下步骤：

1. **设计具有可分离内部结构的深度学习模型**：使用图网络作为核心归纳偏置，特别适合粒子系统建模
2. **端到端训练模型**：使用可用数据训练GN模型
3. **对模型内部函数进行符号回归**：对φₑ、φᵥ、φᵤ等组件拟合符号表达式
4. **用符号表达式替换神经网络组件**：创建完全解析的物理模型

### 2.4 稀疏表示与正则化

**核心理论**：如果真实物理系统可以用线性潜在空间完美描述，且我们用足够大的消息维度训练GNN，那么学到的消息向量将是真实力向量的**线性变换**。

例如，对于引力系统 `F = -G·m₁m₂/r²·r̂`，消息向量可能学到：
```
message = [c₁·dx, c₂·dy, c₃·m₁m₂/r², c₄·..., 0, 0, ...]
```
其中只有少数几个通道是活跃的，其余为零。

为了鼓励这种稀疏性，我们在训练中使用正则化策略：

| 正则化方法 | 适用模型 | 作用机制 | 配置参数 |
|-----------|---------|---------|---------|
| **L1正则化** | OGN | 对节点特征施加L1惩罚，迫使网络学习稀疏激活 | `l1_strength: 1e-2` |
| **KL正则化** | VarOGN | 使后验分布接近稀疏先验，自然产生稀疏性 | `kl_weight: 1e-3` |
| **瓶颈架构** | 所有 | 限制消息维度（msg_dim），迫使信息压缩 | `msg_dim: 100` |

**为什么稀疏性重要**：
1. **符号回归可行性**：稀疏表示意味着只有少数变量参与，符号回归搜索空间大幅减小
2. **物理解释性**：活跃的通道对应真实的物理量组合
3. **泛化能力**：简单模型通常泛化更好（奥卡姆剃刀原则）

## 3. 实现细节

### 3.1 数据集介绍

我们的数据生成器支持多种物理系统，包括：

| 系统类型 | 势能函数 | 物理意义 |
|---------|---------|---------|
| `r2` | `-m₁m₂/r` | 引力/库仑力（吸引）|
| `r1` | `m₁m₂·log(r)` | 2D引力（涡旋）|
| `spring` | `(r-1)²` | 胡克定律（平衡长度=1）|
| `damped` | `(r-1)² + damping·v·x/n` | 带阻尼的弹簧 |
| `string` | `(r-1)² + y·q` | 弦 + 重力 |
| `charge` | `q₁q₂/r` | 库仑力（排斥/吸引）|
| `superposition` | `q₁q₂/r - m₁m₂/r` | 电磁+引力叠加 |
| `discontinuous` | 分段函数 | 模拟非光滑力（如碰撞）|
| `string_ball` | 弹簧 + 球体排斥 | 复杂约束系统 |

数据生成器使用高精度ODE求解器（`scipy.integrate.odeint`）生成轨迹数据，确保物理准确性。

### 3.2 数据生成器详解

数据生成器（`simulate.py`）使用 PaddlePaddle 自动微分计算力，核心流程如下：

1. **势能函数定义**：为每种物理系统定义势能函数 U(r)
2. **自动微分计算力**：`F = -∇U`，使用 `paddle.grad` 计算势能梯度
3. **ODE求解**：使用 `scipy.integrate.odeint` 求解 `d²x/dt² = F/m`
4. **数据格式**：
   - 输入：[x, y, vx, vy, charge, mass]（2D系统）
   - 标签：加速度 [ax, ay] 或导数 [dq/dt, dp/dt]

### 3.3 模型构建

我们实现了三种核心模型架构：

#### OGN (Object-based Graph Network)
- **物理基础**：牛顿力学 (F=ma)
- **输出**：加速度 a = F/m
- **网络结构**：
  - 消息函数（Message Function）：`φₑ: (xᵢ, xⱼ) → mᵢⱼ`，将源节点和目标节点特征映射到消息向量（维度100）
  - 节点更新函数（Node Function）：`φᵥ: (xᵢ, Σmᵢⱼ) → aᵢ`，聚合所有消息并输出加速度
- **适用场景**：通用动力学系统（弹簧、引力、电荷等）
- **正则化**：L1正则化鼓励消息向量稀疏性，使符号回归更容易

#### VarOGN (Variational ODE Graph Network)
- **物理基础**：变分推断框架
- **输出**：加速度均值和方差（不确定性量化）
- **网络结构**：
  - 消息函数输出 μ 和 logσ²
  - 训练时采样：`m = μ + ε·exp(logσ²/2)`，其中 ε ~ N(0,1)
  - 推理时使用均值：`m = μ`
- **适用场景**：噪声数据/不确定性建模/鲁棒性评估
- **正则化**：KL散度正则化使后验分布接近先验分布

#### HGN (Hamiltonian Graph Network)
- **物理基础**：哈密顿力学（能量守恒原理）
- **输出**：[dq/dt, dp/dt]（广义坐标和广义动量的时间导数）
- **网络结构**：
  - 成对能量函数：`Eᵢⱼ = φₑ(xᵢ, xⱼ)`
  - 自身能量函数：`Eᵢ = φᵥ(xᵢ)`
  - 哈密顿量：`H = Σᵢ Eᵢ + Σᵢⱼ Eᵢⱼ`
  - 哈密顿方程：`dq/dt = ∂H/∂p`, `dp/dt = -∂H/∂q`
- **适用场景**：保守系统（无耗散）、长时间预测
- **特点**：通过自动微分实现哈密顿方程，天然保证能量守恒（在无耗散系统中）

模型通过配置文件中的 `MODEL.arch` 参数动态选择：

```yaml
MODEL:
  arch: "OGN"  # 可选: "OGN", "VarOGN", "HGN"
  input_keys: ["node_features", "edge_index"]
  output_keys: ["acceleration"]  # HGN: ["derivative"]
  n_f: auto  # 自动设置为 dimension * 2 + 2
  msg_dim: 100  # 消息维度（OGN/VarOGN）
  ndim: auto  # 自动设置为 dimension
  hidden: 300  # 隐藏层维度
```

**输入输出格式**：

| 模型 | 输入格式 | 输出格式 | 说明 |
|------|---------|---------|------|
| OGN | `node_features`: [batch, n_nodes, n_f]<br>`edge_index`: [batch, 2, n_edges] | `acceleration`: [batch, n_nodes, ndim] | 直接预测加速度 |
| VarOGN | 同上 | `acceleration`: [batch, n_nodes, ndim] | 预测加速度（均值） |
| HGN | 同上 | `derivative`: [batch, n_nodes, 2*ndim] | 前半部分为dq/dt，后半部分为dp/dt |

其中 `n_f = 2*ndim + 2`（位置、速度、电荷、质量）。

### 3.4 约束构建

我们使用PPSCI的`SupervisedConstraint`构建监督约束。数据流程如下：

1. **数据集准备**：
   ```python
   train_dataset = OGNDataset(cfg, mode="train")
   train_data, train_labels = train_dataset.get_data_and_labels()
   ```

2. **约束构建**：
   ```python
   train_constraint = ppsci.constraint.SupervisedConstraint(
       {
           "dataset": {
               "name": "NamedArrayDataset",
               "input": train_data,
               "label": train_labels
           },
           "batch_size": cfg.TRAIN.batch_size,
           "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": True},
       },
       loss=loss_fn,
       name="train_constraint",
   )
   ```

3. **损失函数**：根据模型类型自动调整
   - **OGN/VarOGN**：MAE或MSE损失 + L1正则化（作用于节点特征）
   - **HGN**：仅对加速度部分（derivative的后半部分）计算损失

**自动化配置**：
- `batch_size = auto`：自动计算为 `int(64 * (4 / num_nodes)²)`
- 训练/验证集划分：默认 0.8/0.2
- 数据下采样：`downsample_factor = 5`，减少计算量

### 3.5 优化器构建

训练使用Adam优化器配合OneCycleLR学习率调度：

```python
# 动态计算每个epoch的批次数
batch_per_epoch = int(1000*10 / (cfg.TRAIN.batch_size/32.0))

# OneCycleLR学习率调度器
lr_scheduler = paddle.optimizer.lr.OneCycleLR(
    max_learning_rate=cfg.TRAIN.lr_scheduler.max_learning_rate,
    total_steps=cfg.TRAIN.epochs * batch_per_epoch,
    divide_factor=cfg.TRAIN.lr_scheduler.final_div_factor
)

optimizer = paddle.optimizer.Adam(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=cfg.TRAIN.optimizer.weight_decay
)
```

关键超参数：

```yaml
TRAIN:
  epochs: 1000
  batch_size: auto  # 自动计算
  optimizer:
    learning_rate: 1e-3
    weight_decay: 1e-8
  lr_scheduler:
    name: "OneCycleLR"
    max_learning_rate: 1e-3
    final_div_factor: 1e5  # 最终学习率 = max_lr / final_div_factor
  loss:
    type: "MAE"  # 可选 MAE 或 MSE
```

**学习率调度策略**：
- OneCycleLR：学习率先从 max_lr/divide_factor 增加到 max_lr，再逐渐降低到 max_lr/final_div_factor
- 适合快速收敛且避免过拟合

### 3.6 符号回归准备

训练GNN的目的是为符号回归做准备。根据论文思想，我们鼓励稀疏的潜在表示：

**理论基础**：
- 如果真实物理系统有完美的线性潜在空间描述，训练后的GNN消息向量将是真实力向量的线性变换
- L1正则化迫使网络学习稀疏表示，使得消息向量中只有少数维度是活跃的
- 这些活跃的维度通常对应真实的物理量（如距离、质量乘积等）

**实践流程**：

1. **训练稀疏GNN**：
   ```yaml
   MODEL:
     msg_dim: 100  # 消息维度
     regularization_type: "l1"
     l1_strength: 1e-2
   ```

2. **提取消息数据**：训练过程中可以记录消息向量（需要修改代码添加钩子）
   ```python
   messages = model.msg_fnc(edge_features)  # [num_edges, msg_dim]
   # 保存消息及对应的物理特征（dx, dy, r, m1, m2等）
   ```

3. **符号回归**：使用工具如 [PySR](https://github.com/MilesCranmer/PySR) 拟合符号表达式
   ```python
   from pysr import PySRRegressor
   
   # 选择最活跃的消息通道
   active_channels = np.argsort(np.std(messages, axis=0))[-5:]
   
   # 对每个活跃通道进行符号回归
   for ch in active_channels:
       model = PySRRegressor(
           niterations=40,
           binary_operators=["+", "*", "/", "-"],
           unary_operators=["square", "sqrt", "neg"],
       )
       model.fit(physical_features, messages[:, ch])
       print(f"Channel {ch}: {model.sympy()}")
   ```

4. **发现物理定律**：
   - 对于弹簧系统：可能发现 `m ~ (r-1)` （胡克定律）
   - 对于引力系统：可能发现 `m ~ -m1*m2/r²` （万有引力定律）
   - 对于电荷系统：可能发现 `m ~ q1*q2/r²` （库仑定律）

## 4. 完整代码

完整实现包含以下文件：

```python


class OGNDataset:
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.data = None
        self.labels = None
        self.input_keys = tuple(cfg.MODEL.input_keys)
        self.label_keys = tuple(cfg.MODEL.output_keys)
        self._prepare_data()

    def _prepare_data(self):
        # Get time step size for physics system
        dt = 1e-2
        for sim_set in self.cfg.DATATYPE_TYPE:
            if sim_set['sim'] == self.cfg.DATA.type:
                dt = sim_set['dt'][0]
                break

        # Generate simulation data
        sim_dataset = SimulationDataset(
            sim=self.cfg.DATA.type,
            n=self.cfg.DATA.num_nodes,
            dim=self.cfg.DATA.dimension,
            dt=dt,
            nt=self.cfg.DATA.time_steps,
            seed=self.cfg.seed
        )
        sim_dataset.simulate(self.cfg.DATA.num_samples)
        
        # Compute target data based on model architecture
        if self.cfg.MODEL.arch == "HGN":
            raw_target_data = sim_dataset.get_derivative()  # [v, a] for HGN
        else:
            raw_target_data = sim_dataset.get_acceleration()  # Acceleration for OGN/VarOGN
        
        # Downsample data
        downsample_factor = self.cfg.DATA.downsample_factor
        input_data = np.concatenate([sim_dataset.data[:, i] for i in range(0, sim_dataset.data.shape[1], downsample_factor)])
        target_data = np.concatenate([raw_target_data[:, i] for i in range(0, sim_dataset.data.shape[1], downsample_factor)])

        # Split train/validation set
        from sklearn.model_selection import train_test_split
        test_size = 1.0 - getattr(self.cfg.TRAIN, 'train_split', 0.8)
        
        if self.mode == "train":
            input_data, _, target_data, _ = train_test_split(
                input_data, target_data, test_size=test_size, shuffle=False
            )
        elif self.mode == "eval":
            _, input_data, _, target_data = train_test_split(
                input_data, target_data, test_size=test_size, shuffle=False
            )
        
        # Generate edge indices
        edge_index = get_edge_index(self.cfg.DATA.num_nodes, self.cfg.DATA.type)

        self.data = {
            "node_features": input_data.astype(np.float32),
            "edge_index": np.tile(edge_index.T.numpy(), (len(input_data), 1, 1))
        }
        self.labels = {
            list(self.cfg.MODEL.output_keys)[0]: target_data.astype(np.float32)
        }

    def __len__(self):
        return len(self.data["node_features"])

    def __getitem__(self, idx):
        return {
            "input": {
                "node_features": self.data["node_features"][idx],
                "edge_index": self.data["edge_index"][idx]
            },
            "label": {
                list(self.cfg.MODEL.output_keys)[0]: self.labels[list(self.cfg.MODEL.output_keys)[0]][idx]
            }
        }
        
    def get_data_and_labels(self):
        return self.data, self.labels
        
    def __call__(self):
        return self


def create_loss_function(cfg):
    def loss_function(input_dict, label_dict, model_output):
        # Get loss type from config (MAE or MSE)
        loss_type = getattr(cfg.TRAIN.loss, 'type', 'MAE')
        
        # Get output key
        output_key = list(cfg.MODEL.output_keys)[0] if cfg.MODEL.arch != "HGN" else "derivative"
        
        # Compute base loss
        if cfg.MODEL.arch == "HGN":
            output_key = "derivative"
            pred_accel = model_output[output_key][:, cfg.DATA.dimension:]
            target_accel = label_dict[output_key][:, cfg.DATA.dimension:]
            if loss_type == "MSE":
                base_loss = paddle.mean(paddle.square(pred_accel - target_accel))
            else:  # MAE
                base_loss = paddle.mean(paddle.abs(pred_accel - target_accel))
        else:
            pred = model_output[output_key]
            target = label_dict[output_key]
            if loss_type == "MSE":
                base_loss = paddle.mean(paddle.square(pred - target))
            else:  # MAE
                base_loss = paddle.mean(paddle.abs(pred - target))

        # Add regularization if configured
        if cfg.MODEL.regularization_type == "l1" and cfg.MODEL.arch in ["OGN", "VarOGN"]:
            reg_loss = cfg.MODEL.l1_strength * paddle.mean(paddle.abs(input_dict["node_features"]))
            return base_loss + reg_loss

        return base_loss
    
    return loss_function


def train(cfg):
    # Set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.TRAIN.batch_size == "auto":
        cfg.TRAIN.batch_size = int(64 * (4 / cfg.DATA.num_nodes) ** 2)
    
    # Update output keys for HGN
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create datasets
    logger.message(f"Creating {cfg.DATA.type} dataset using simulate.py...")
    train_dataset = OGNDataset(cfg, mode="train")
    eval_dataset = OGNDataset(cfg, mode="eval")
    
    # Get training and evaluation data
    train_data, train_labels = train_dataset.get_data_and_labels()
    eval_data, eval_labels = eval_dataset.get_data_and_labels()
    
    # Create model based on architecture
    logger.message(f"Creating model: {cfg.MODEL.arch}")
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create loss function and constraints
    loss_fn = create_loss_function(cfg)
    train_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": train_data,
                "label": train_labels
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": True},
        },
        loss=loss_fn,
        name="train_constraint",
    )
    
    eval_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_labels
            },
            "batch_size": cfg.EVAL.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": False},
        },
        loss=loss_fn,
        name="eval_constraint",
    )
    
    constraint = {
        train_constraint.name: train_constraint,
        eval_constraint.name: eval_constraint,
    }
    
    # Calculate iterations per epoch
    batch_per_epoch = int(1000*10 / (cfg.TRAIN.batch_size/32.0))
    
    # Create optimizer and learning rate scheduler
    if cfg.TRAIN.lr_scheduler.name == "OneCycleLR":
        lr_scheduler = paddle.optimizer.lr.OneCycleLR(
            max_learning_rate=cfg.TRAIN.lr_scheduler.max_learning_rate,
            total_steps=cfg.TRAIN.epochs*batch_per_epoch,
            divide_factor=cfg.TRAIN.lr_scheduler.final_div_factor
        )
    else:
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=cfg.TRAIN.optimizer.learning_rate,
            gamma=0.9
        )
    
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=cfg.TRAIN.optimizer.weight_decay
    )
    
    # Create PaddleScience Solver and start training
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        cfg=cfg,
    )
    solver.train()


def evaluate(cfg: DictConfig):
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create model based on architecture
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create evaluation dataset
    eval_dataset = OGNDataset(cfg, mode="eval")
    eval_data, eval_labels = eval_dataset.get_data_and_labels()
    
    # Create loss function and constraint
    loss_fn = create_loss_function(cfg)
    eval_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_labels
            },
            "batch_size": cfg.EVAL.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": False},
        },
        loss=loss_fn,
        name="eval_constraint",
    )
    
    constraint = {
        eval_constraint.name: eval_constraint,
    }
    
    # Create PaddleScience Solver
    solver = ppsci.solver.Solver(model, constraint, cfg=cfg)
    
    # Run evaluation
    logger.message("Starting evaluation...")
    eval_result = solver.eval()
    logger.message(f"Evaluation completed. Result: {eval_result}")


def inference(cfg: DictConfig):
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create model based on architecture
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create PaddleScience Solver
    solver = ppsci.solver.Solver(model, cfg=cfg)

    # Generate test data
    logger.message("Generating inference data using simulate.py...")
    dt = 1e-2
    for sim_set in cfg.DATATYPE_TYPE:
        if sim_set['sim'] == cfg.DATA.type:
            dt = sim_set['dt'][0]
            break
    
    sim_dataset = SimulationDataset(
        sim=cfg.DATA.type,
        n=cfg.DATA.num_nodes,
        dim=cfg.DATA.dimension,
        dt=dt,
        nt=100,
        seed=cfg.seed
    )
    sim_dataset.simulate(1)

    # Prepare input data (first time step)
    node_features = sim_dataset.data[0, 0]
    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)
    
    input_dict = {
        "node_features": paddle.to_tensor(node_features, dtype='float32').unsqueeze(0),
        "edge_index": edge_index.unsqueeze(0)
    }

    # Run inference
    logger.message("Running inference...")
    output_dict = solver.predict(input_dict, return_numpy=True)

    # Display results
    if cfg.MODEL.arch == "HGN":
        dq_dt = output_dict["derivative"][0, :cfg.DATA.dimension]
        dv_dt = output_dict["derivative"][0, cfg.DATA.dimension:]
        logger.message(f"HGN inference - dq/dt: {dq_dt}, dv/dt (acceleration): {dv_dt}")
    else:
        acceleration = output_dict[list(cfg.MODEL.output_keys)[0]][0]
        logger.message(f"{cfg.MODEL.arch} inference - acceleration: {acceleration}")


def export(cfg: DictConfig):
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create model based on architecture
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create PaddleScience Solver
    solver = ppsci.solver.Solver(model, cfg=cfg)
    
    # Define input specifications for export
    from paddle.static import InputSpec
    input_spec = [
        {
            "node_features": InputSpec([None, cfg.DATA.num_nodes, cfg.MODEL.n_f], "float32", "node_features"),
            "edge_index": InputSpec([None, 2, cfg.DATA.num_nodes * (cfg.DATA.num_nodes - 1)], "int64", "edge_index")
        }
    ]
    
    # Export model to static graph
    logger.message(f"Exporting model to {cfg.INFER.export_path}")
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)
    logger.message("Model export completed!")


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    elif cfg.mode == "export":
        export(cfg)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
```


## 5. 预期结果


## 6. 参考资料

- **论文**：[Discovering Symbolic Models from Deep Learning with Inductive Biases (NeurIPS 2020)](https://arxiv.org/abs/2006.11287)
- **作者**：Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, Shirley Ho
- **代码仓库**：[github.com/MilesCranmer/symbolic_deep_learning](https://github.com/MilesCranmer/symbolic_deep_learning)
- **符号回归工具**：[PySR - Python Symbolic Regression](https://github.com/MilesCranmer/PySR)
