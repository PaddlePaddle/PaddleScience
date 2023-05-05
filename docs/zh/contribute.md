# 贡献指南

本文档介绍如何基于 PaddleScience 套件进行代码开发并最终贡献到 PaddleScience 套件中

## 1. 准备工作

1. 将 PaddleScience fork 到自己的仓库
2. 克隆自己仓库里的 PaddleScience 到本地，并进入该目录

    ``` shell
    git clone https://github.com/your_username/PaddleScience.git
    cd PaddleScience
    ```

3. 安装必要的依赖包

    ``` shell
    pip install -r requirements.txt
    # 安装较慢时可以加上-i选项，提升下载速度
    # pip install -r requirements.txt -i https://pypi.douban.com/simple/
    ```

4. 基于 develop 分支，新建一个新分支(假设新分支名字为dev_model)

    ``` shell
    git checkout -b "dev_model"
    ```

5. 添加 PaddleScience 目录到系统环境变量 PYTHONPATH 中

    ``` shell
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```

## 2. 编写代码

完成上述准备工作后，就可以基于 PaddleScience 提供的 API 开始编写自己的案例代码了，接下来开始详细介绍
这一过程。

### 2.1 导入必要的包

PaddleScience 所提供的 API 全部在 `ppsci` 模块下，因此在代码文件的开头首先需要导入 `ppsci` 这个顶
层模块以及日志打印模块，然后再根据您自己的需要，导入其他必要的模块。

``` python
import ppsci
from ppsci.utils import logger

# 导入其他必要的模块
# import ...
```

### 2.2 设置运行环境

在运行 python 的主体代码之前，我们同样需要设置一些必要的运行环境，比如固定随机种子、设置模型/日志保存
目录、初始化日志打印模块。

``` python
if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    output_dir = "./ldc2d_steady_Re10"
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")
```

### 2.3 构建模型

#### 2.3.2 构建已有模型

PaddleScience 内置了一些常见的模型，如 `MLP` 模型等，如果您想使用这些内置的模型，可以直接调用
[`ppsci.arch`](./api/arch.md) 下的 API，并填入模型实例化所需的参数，即可快速构建模型。

``` python
# create a MLP model
model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 9, 50, "tanh")
```

#### 2.3.1 构建新的模型

当 PaddleScienc 内置的模型无法满足您的需求时，您也可以通过新增模型文件并编写模型代码的方式，使用您自
定义的模型，步骤如下：

1. 在 `ppsci/arch` 下新建模型文件，以 `new_model.py` 为例
2. 在 `new_model.py` 文件中导入 PaddleScience 的模型基类所在模块 `base`，并让创建的新模型类（以
`Class NewModel` 为例）从 `base.Arch` 继承。

    ``` python title="ppsci/arch/new_model.py"
    from ppsci.arch import base

    class NewModel(base.Arch):
        def __init__(self, ...):
            ...
            # init

        def forward(self, ...):
            if self._input_transform is not None:
                x = self._input_transform(x)

            y = self.concat_to_tensor(x, self.input_keys, axis=-1)
            y = self.forward_tensor(y)
            y = self.split_to_dict(y, self.output_keys, axis=-1)

            if self._output_transform is not None:
                y = self._output_transform(y)
            return y
    ```

3. 编写 `__init__` 代码，用于模型创建时的初始化；然后再编写 `forward` 代码，用于定义模型接受输入、
计算输出这一前向过程。

4. 在 `ppsci/arch/__init__.py` 中导入编写的新模型类，并添加到 `__all__` 中

    ``` python title="ppsci/arch/__init__.py"
    from ppsci.arch.new_model import NewModel

    __all__ = [
        ...,
        ...,
        "NewModel",
    ]
    ```

完成上述新模型代码编写的工作之后，我们就能像 PaddleScience 内置模型一样，以
`ppsci.arch.NewModel` 的方式，调用我们编写的新模型类，并用于创建模型实例。

### 2.4.2 构建方程

如果您的案例问题中涉及到方程计算，那么可以选择使用 PaddleScience 内置的方程，或者编写自己的方程。

#### 2.4.1 构建已有方程

PaddleScience 内置了一些常见的方程，如 `NavierStokes` 方程等，如果您想使用这些内置的方程，可以直接
调用 [`ppsci.equation`](./api/equation.md) 下的 API，并填入方程实例化所需的参数，即可快速构建模型。

``` python
# create a Vibration equation
viv_equation = ppsci.equation.Vibration(2, -4, 0)
```

#### 2.4.2 构建新的方程

当 PaddleScienc 内置的方程无法满足您的需求时，您也可以通过新增方程文件并编写方程代码的方式，使用您自
定义的方程，步骤如下：

1. 在 `ppsci/equation/pde` 下新建方程文件（如果您的方程并不是 PDE 方程，那么需要新建一个方程类文件
夹，比如在 `ppsci/equation` 下新建 `ode` 文件夹，再将您的方程文件放在 `ode` 文件夹下，此处以 PDE 类的方程 `new_pde.py` 为例）

2. 在 `new_pde.py` 文件中导入 PaddleScience 的方程基类所在模块 `base`，并让创建的新方程类
（以`Class NewPDE` 为例）从 `base.PDE` 继承

    ``` python title="ppsci/equation/pde/new_pde.py"
    from ppsci.equation.pde import base

    class NewPDE(base.PDE):
        def __init__(self, ...):
            ...
            # init
    ```

3. 编写 `__init__` 代码，用于方程创建时的初始化

4. 在 `ppsci/equation/__init__.py` 中导入编写的新方程类，并添加到 `__all__` 中

    ``` python title="ppsci/equation/__init__.py"
    from ppsci.equation.pde.new_pde import NewPDE

    __all__ = [
        ...,
        ...,
        "NewPDE",
    ]
    ```

完成上述新方程代码编写的工作之后，我们就能像 PaddleScience 内置方程一样，以
`ppsci.equation.NewPDE` 的方式，调用我们编写的新方程类，并用于创建方程实例。

在方程构建完毕后之后，我们需要将所有方程包装为到一个字典中

``` python
equation = {
    "ViV": viv_equation,
}
```

### 2.5 构建约束条件

无论是 PINNs 方法还是数据驱动方法，它们总是需要一些数据来指导网络模型的训练，而利用数据指导网络训练的
这一过程，在 PaddleScience 中由 `Constraint`（约束） 负责。

#### 2.5.1 构建已有约束

PaddleScience 内置了一些常见的约束，如 `SupervisedConstraint` 监督约束等，如果您想使用这些内置的
约束，可以直接调用 [`ppsci.constraint`](./api/constraint.md) 下的 API，并填入约束实例化所需的参
数，即可快速构建约束条件。

``` python
# create a SupervisedConstraint
sup_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    {"eta": lambda out: out["eta"], **equation["VIV"].equations},
    name="Sup",
)
```

#### 2.5.1 构建新的约束

当 PaddleScienc 内置的约束无法满足您的需求时，您也可以通过新增约束文件并编写约束代码的方式，使用您自
定义的约束，步骤如下：

1. 在 `ppsci/constraint` 下新建约束文件（此处以约束 `new_constraint.py` 为例）

2. 在 `new_constraint.py` 文件中导入 PaddleScience 的约束基类所在模块 `base`，并让创建的新约束
类（以 `Class NewConstraint` 为例）从 `base.PDE` 继承

    ``` python title="ppsci/constraint/new_constraint.py"
    from ppsci.constraint import base

    class NewConstraint(base.Constraint):
        def __init__(self, ...):
            ...
            # init
    ```

3. 编写 `__init__` 代码，用于约束创建时的初始化

4. 在 `ppsci/constraint/__init__.py` 中导入编写的新约束类，并添加到 `__all__` 中

    ``` python title="ppsci/constraint/__init__.py"
    from ppsci.constraint.new_constraint import NewConstraint

    __all__ = [
        ...,
        ...,
        "NewConstraint",
    ]
    ```

完成上述新约束代码编写的工作之后，我们就能像 PaddleScience 内置约束一样，以
`ppsci.constraint.NewConstraint` 的方式，调用我们编写的新约束类，并用于创建约束实例。

### 2.6 定义超参数

在模型开始训练前，需要制定一些训练相关的超参数，如训练轮数、学习率等，如下所示

``` python
# set training hyper-parameters
epochs = 10000
learning_rate = 0.001
```

### 2.7 构建优化器

模型训练除了用于计算的模型本身，还需要一个用于更新模型参数的优化器，如下所示

``` python
# set optimizer
optimizer = ppsci.optimizer.Adam(0.001)(model)
```

### 2.8 构建评估器

#### 2.8.1 构建已有评估器

PaddleScience 内置了一些常见的评估器，如 `SupervisedValidator` 评估器等，如果您想使用这些内置的评
估器，可以直接调用 [`ppsci.validate`](./api/validate.md) 下的 API，并填入评估器实例化所需的参数，
即可快速构建评估器。

``` python
# create a SupervisedValidator
eta_mse_validator = ppsci.validate.SupervisedValidator(
    valida_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    {"eta": lambda out: out["eta"], **equation["VIV"].equations},
    metric={"MSE": ppsci.metric.MSE()},
    name="eta_mse",
)
```

#### 2.8.2 构建新的评估器

当 PaddleScienc 内置的评估器无法满足您的需求时，您也可以通过新增评估器文件并编写评估器代码的方式，使
用您自定义的评估器，步骤如下：

1. 在 `ppsci/validate` 下新建评估器文件（此处以 `new_validator.py` 为例）

2. 在 `new_validator.py` 文件中导入 PaddleScience 的评估器基类所在模块 `base`，并让创建的新评估
器类（以 `Class NewValidator` 为例）从 `base.Validator` 继承

    ``` python title="ppsci/validate/new_validator.py"
    from ppsci.validate import base

    class NewValidator(base.Validator):
        def __init__(self, ...):
            ...
            # init
    ```

3. 编写 `__init__` 代码，用于评估器创建时的初始化

4. 在 `ppsci/validate/__init__.py` 中导入编写的新评估器类，并添加到 `__all__` 中

    ``` python title="ppsci/validate/__init__.py"
    from ppsci.validate.new_validator import NewValidator

    __all__ = [
        ...,
        ...,
        "NewValidator",
    ]
    ```

完成上述新评估器代码编写的工作之后，我们就能像 PaddleScience 内置评估器一样，以
`ppsci.equation.NewValidator` 的方式，调用我们编写的新评估器类，并用于创建评估器实例。

#### 2.10 构建可视化器

PaddleScience 内置了一些常见的可视化器，如 `VisualizerVtu` 可视化器等，如果您想使用这些内置的可视
化器，可以直接调用 [`ppsci.visulizer`](./api/visualize.md) 下的 API，并填入可视化器实例化所需的
参数，即可快速构建模型。

``` python
# manually collate input data for visualization,
# interior+boundary
vis_points = {}
for key in vis_interior_points:
    vis_points[key] = np.concatenate(
        (vis_interior_points[key], vis_boundary_points[key])
    )

visualizer = {
    "visulzie_u_v": ppsci.visualize.VisualizerVtu(
        vis_points,
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
        prefix="result_u_v",
    )
}
```

如需新增可视化器，步骤与其他模块的新增方法相同，此处不再赘述。

### 2.9 构建Solver

[`Solver`](./api/solver.md) 是 PaddleScience 负责调用训练、评估、可视化的类，在训练开始前，需要把构建好的模型、约束、优化
器等实例传给 Solver，再调用它的内置方法进行训练、评估、可视化。

``` python
# initialize solver
solver = ppsci.solver.Solver(
    model,
    constraint,
    output_dir,
    optimizer,
    lr_scheduler,
    epochs,
    iters_per_epoch,
    eval_during_train=True,
    eval_freq=eval_freq,
    equation=equation,
    validator=validator,
    visualizer=visualizer,
)
```

### 2.10 训练

PaddleScience 模型的训练只需调用一行代码

``` python
# train model
solver.train()
```

### 2.11 评估

PaddleScience 模型的评估只需调用一行代码

``` python
# train model
solver.eval()
```

### 2.12 可视化

PaddleScience 模型的可视化只需调用一行代码

``` python
# train model
solver.visualize()
```

## 3. 整理代码并提交

### 3.1 整理代码

在完成范例编写并训练完成后，确认结果无误，就可以把代码进行整理，使用 git 命令将所有新增、修改的代码文件
以及必要的文档、图片等更新到自己的仓库的 `dev_model` 分支上

### 3.2 提交 pull request

在 github 网页端切换到 `dev_model` 分支，并点击 "Contribute"，再点击 "Open pull request" 按钮，
将你的代码作为合入请求贡献到 PaddleScience。
