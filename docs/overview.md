# PaddleScience 介绍

PaddleScience 是一个基于深度学习框架 PaddlePaddle 的科学计算套件，支持物理信息驱动、数据驱动、数理融合三种求解方式，已支持 8+ 各领域的案例，并提供了基础 API 供用户使用与二次开发。

## 1. 运行流程

![TODO](workflow.jpg)

## 2. 模块简介

### 2.1 [Arch](./zh/api/arch.md)

Arch 模块负责各种神经网络模型的组网、参数初始化、前向计算等功能，内置了多种模型供用户使用。

### 2.2 [AutoDiff](./zh/api/autodiff.md)

AutoDiff 模块负责计算高阶（1、2阶）微分功能，内置全局单例模式对象 `jacobian`、`hessian` 供用户使用。

### 2.3 [Constraint](./zh/api/constraint.md)

为了在套件中统一物理信息驱动、数据驱动、数理融合三种求解方式，我们将数据构造、输入到输出的计算过程、损失函数等功能记录到了 Constraint 这一模块中，有了这些对象，Constraint 就能表示不同的训练目标，如

- `InteriorConstraint` 定义了在给定的几何区域内部，按照给定输入到输出的计算过程，利用损失函数优化模型参数，使得模型输出满足给定的条件；
- `BoundaryConstraint` 定义了在给定的几何区域边界上，按照给定输入到输出的计算过程，利用损失函数优化模型参数，使得模型输出满足给定的条件；
- `SupervisedConstraint` 定义了在给定的监督数据（相当于CV、NLP中的监督训练）上，按照给定输入到输出的计算过程，利用损失函数优化模型参数，使得模型输出满足给定的条件。
- ...

这一模块有两个主要作用，一是在代码流程上统一了物理信息驱动、数据驱动两个不同的优化范式（前者类似监督训练方式，后者类似无监督训练方式），二是使得套件能应用在数理融合的场景中，只需分别构造不同的 Constraint 并让它们共同参与训练即可。

### 2.4 Data

Data 模块负责数据的读取、包装以及预处理，由 3 个子模块分别负责。

#### 2.4.1 [Dataset](./zh/api/data/dataset.md)

Dataset 模块存放各种数据集的包装类，将数据包装成统一的结构再用于读取。

#### 2.4.2 [Transform](./zh/api/data/process/transform.md)

Transform 模块存放各种数据预处理方法，包括数据平移 `Translate` 和数据缩放 `Scale`，一般与 `data.dataset` 结合使用。

#### 2.4.3 [BatchTransform](./zh/api/data/process/batch_transform.md)

BatchTransform 模块存放各种基于批数据的预处理方法。

### 2.5 [Equation](./zh/api/equation.md)

Equation 模块负责定义各种常见方程的计算函数，如 `NavierStokes` 表示 N-S 方程，`Vibration` 表示振动方程，每个方程内部含有相关变量的计算函数。

### 2.6 [Geometry](./zh/api/geometry.md)

Geometry 模块负责定义各种常见的几何形状，如 `Interval` 线段几何、`Rectangle` 矩形几何、`Sphere` 球面几何。

### 2.7 [Loss](./zh/api/loss.md)

Loss 模块负责定义各种损失函数，在模型前向与方程计算完毕后，将结果与参考值计算损失值，供后续的梯度优化算法使用。

### 2.8 [Optimizer](./zh/api/optimizer.md)

Optimizer 模块负责定义各种优化器，如 `SGD`、`Adam`、`LBFGS`。

### 2.9 [Solver](./zh/api/solver.md)

Solver 模块负责定义求解器，作为训练、评估、推理、可视化的启动和管理引擎。

### 2.10 [Utils](./zh/api/utils.md)

Utils 模块内部存放了一些适用于多种场景下的函数，如在 `reader.py` 下的多种数据读取函数，以及在 `logger.py` 下的日志打印函数。

### 2.11 [Validate](./zh/api/validate.md)

Validator 模块负责定义各种评估器，用于模型训练完一个 epoch 后自动在指定数据上进行评估（可选，默认不开启训练时评估）并得到评估指标。

### 2.12 [Visualize](./zh/api/visualize.md)

Visualizer 模块负责定义各种可视化器，用于模型评估完后在指定数据上进行预测（可选，默认不开启训练时可视化）并将结果保存成可视化的文件。
