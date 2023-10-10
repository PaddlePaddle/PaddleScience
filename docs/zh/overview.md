# PaddleScience 模块介绍

PaddleScience 在代码结构上划分为 12 个模块。从一般深度学习工作流的角度来看，这 12 个模块分别负责构建输入数据、构建神经网络模型、构建损失函数、构建优化器，训练、评估、可视化等功能。从科学计算角度来看，部分模块承担了不同于 CV、NLP 任务的功能，比如用于物理机理驱动的 Equation 模块，定义方程公式和辅助高阶微分计算；用于涉及几何场景采样的 Geometry 模块，定义简单、复杂几何形状并在其内部、边界采样构造数据；Constraint 模块将不同的优化目标视为一种“约束”，使得套件能用一套训练代码统一物理机理驱动、数据驱动、数理融合三种不同的求解流程。

## 1. 整体工作流

<figure markdown>
  ![workflow](../images/overview/workflow.jpg){ loading=lazy style="height:80%;width:80%"}
</figure>

上图是 PaddleScience 的 workflow 示意图（以基于几何的问题求解为例），流程描述如下

1. Geometry 负责构建几何并在几何上采样，完成数据构建；
2. 用 Model 模块接受输入，得到模型输出；
3. 科学计算任务具有特殊性，模型输出往往并不是前向计算的终点，还需要进一步按照 Equation，计算出方程公式所需的变量；
4. 计算损失函数，并利用框架的自动微分机制，求出所有参数的梯度；
5. 上述的优化目标可以施加在几何的不同区域上，比如interior、boundary区域，因此上图中的 Constraint 可以有多个；
6. 将所有 Constraint 贡献的梯度累加，并用于更新模型参数；
7. 训练过程中如果开启了评估和可视化功能，则会按一定频率自动对当前模型进行评估和预测结果可视化；
8. Solver 是整个套件运行的全局调度模块，负责将上述过程按用户指定的轮数和频率重复运行。

## 2. 模块简介

### 2.1 [Arch](./api/arch.md)

Arch 模块负责各种神经网络模型的组网、参数初始化、前向计算等功能，内置了多种模型供用户使用。

### 2.2 [AutoDiff](./api/autodiff.md)

AutoDiff 模块负责计算高阶微分功能，内置基于 Paddle 自动微分机制的全局单例 `jacobian`、`hessian` 供用户使用。

### 2.3 [Constraint](./api/constraint.md)

<figure markdown>
  ![constraint](../images/overview/constraint.jpg){ loading=lazy style="height:50%;width:50%"}
</figure>

为了在套件中统一物理信息驱动、数据驱动、数理融合三种求解方式，我们将数据构造、输入到输出的计算过程、损失函数等必要接口在其定义完毕之后，统一记录在 Constraint 这一模块中，有了这些接口，Constraint 就能表示不同的训练目标，如：

- `InteriorConstraint` 定义了在给定的几何区域内部，按照给定输入到输出的计算过程，利用损失函数优化模型参数，使得模型输出满足给定的条件；
- `BoundaryConstraint` 定义了在给定的几何区域边界上，按照给定输入到输出的计算过程，利用损失函数优化模型参数，使得模型输出满足给定的条件；
- `SupervisedConstraint` 定义了在给定的监督数据（相当于CV、NLP中的监督训练）上，按照给定输入到输出的计算过程，利用损失函数优化模型参数，使得模型输出满足给定的条件。
- ...

这一模块有两个主要作用，一是在代码流程上统一了物理信息驱动、数据驱动两个不同的优化范式（前者类似监督训练方式，后者类似无监督训练方式），二是使得套件能应用在数理融合的场景中，只需分别构造不同的 Constraint 并让它们共同参与训练即可。

### 2.4 Data

Data 模块负责数据的读取、包装和预处理，由以下 3 个子模块分别负责。

#### 2.4.1 [Dataset](./api/data/dataset.md)

Dataset 模块存放各种数据集的包装类，将数据包装成统一的结构再用于读取。

#### 2.4.2 [Transform](./api/data/process/transform.md)

Transform 模块存放各种基于单样本的数据预处理方法，包括数据平移 `Translate` 和数据缩放 `Scale`，一般与 `data.dataset` 结合使用。

#### 2.4.3 [BatchTransform](./api/data/process/batch_transform.md)

BatchTransform 模块存放各种基于批样本的数据预处理方法。

### 2.5 [Equation](./api/equation.md)

<figure markdown>
  ![equation](../images/overview/equation.jpg){ loading=lazy style="height:80%;width:80%"}
</figure>

Equation 模块负责定义各种常见方程的计算函数，如 `NavierStokes` 表示 N-S 方程，`Vibration` 表示振动方程，每个方程内部含有相关变量的计算函数。

### 2.6 [Geometry](./api/geometry.md)

<figure markdown>
  ![geometry](../images/overview/geometry.jpg#center){ loading=lazy style="height:50%;width:50%" }
</figure>

Geometry 模块负责定义各种常见的几何形状，如 `Interval` 线段几何、`Rectangle` 矩形几何、`Sphere` 球面几何。

### 2.7 [Loss](./api/loss.md)

Loss 模块负责定义各种损失函数，在模型前向与方程计算完毕后，将结果与参考值计算损失值，供后续的梯度优化算法使用；

同时该模块也提供了常见的多任务学习优化功能，包括 PCGrad、AGDA，进一步提升多个 loss 共同优化（如 PINN 方法）的精度。

### 2.8 [Optimizer](./api/optimizer.md)

Optimizer 模块负责定义各种优化器，如 `SGD`、`Adam`、`LBFGS`。

### 2.9 [Solver](./api/solver.md)

Solver 模块负责定义求解器，作为训练、评估、推理、可视化的启动和管理引擎。

### 2.10 [Utils]

Utils 模块内部存放了一些适用于多种场景下的工具类、函数，例如在 `reader.py` 下的数据读取函数，在 `logger.py` 下的日志打印函数，以及在 `expression.py` 下的方程计算类。

根据其功能细分为以下 8 个子模块

| 子模块名称 | 子模块功能 |
| :-- | :-- |
| [ppsci.utils.checker](./api/utils/checker.md)| ppsci 安装功能检查相关 |
| [ppsci.utils.expression](./api/utils/expression.md)| python 函数/sympy 表达式前向计算相关 |
| [ppsci.utils.initializer](./api/utils/initializer.md)| 常用参数初始化方法 |
| [ppsci.utils.logger](./api/utils/logger.md)| 日志打印模块 |
| [ppsci.utils.misc](./api/utils/misc.md)| 存放通用函数 |
| [ppsci.utils.reader](./api/utils/reader.md)| 文件读取模块 |
| [ppsci.utils.save_load](./api/utils/save_load.md)| 模型参数保存与加载 |
| [ppsci.utils.symbolic](./api/utils/symbolic.md)| sympy 符号计算功能相关 |

### 2.11 [Validate](./api/validate.md)

Validator 模块负责定义各种评估器，用于模型训练完一个 epoch 后自动在指定数据上进行评估（可选，默认不开启训练时评估）并得到评估指标。

### 2.12 [Visualize](./api/visualize.md)

Visualizer 模块负责定义各种可视化器，用于模型评估完后在指定数据上进行预测（可选，默认不开启训练时可视化）并将结果保存成可视化的文件。
