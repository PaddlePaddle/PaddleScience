# 开发指南

本文档介绍如何基于 PaddleScience 套件进行代码开发并最终贡献到 PaddleScience 套件中。

PaddleScience 相关的论文复现、API 开发任务开始之前需提交 RFC 文档，请参考：[PaddleScience RFC Template](https://github.com/PaddlePaddle/community/blob/master/rfcs/Science/template.md)

## 1. 准备工作

1. 将 PaddleScience fork 到**自己的仓库**
2. 克隆**自己仓库**里的 PaddleScience 到本地，并进入该目录

    ``` sh
    git clone -b develop https://github.com/USER_NAME/PaddleScience.git
    cd PaddleScience
    ```

    上方 `clone` 命令中的 `USER_NAME` 字段请填入的自己的用户名。

3. 安装必要的依赖包

    ``` sh
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

4. 基于当前所在的 `develop` 分支，新建一个分支(假设新分支名字为 `dev_model`)

    ``` sh
    git checkout -b "dev_model"
    ```

5. 添加 PaddleScience 目录到系统环境变量 `PYTHONPATH` 中

    ``` sh
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```

6. 执行以下代码，验证安装的 PaddleScience 基础功能是否正常

    ``` sh
    python -c "import ppsci; ppsci.run_check()"
    ```

    如果出现 PaddleScience is installed successfully.✨ 🍰 ✨，则说明安装验证成功。

## 2. 编写代码

完成上述准备工作后，就可以基于 PaddleScience 开始开发自己的案例或者功能了。

假设新建的案例代码文件路径为：`PaddleScience/examples/demo/demo.py`，接下来开始详细介绍这一流程

### 2.1 导入必要的包

PaddleScience 所提供的 API 全部在 `ppsci.*` 模块下，因此在 `demo.py` 的开头首先需要导入 `ppsci` 这个顶层模块，接着导入日志打印模块 `logger`，方便打印日志时自动记录日志到本地文件中，最后再根据您自己的需要，导入其他必要的模块。

``` py title="examples/demo/demo.py"
import ppsci
from ppsci.utils import logger

# 导入其他必要的模块
# import ...
```

### 2.2 设置运行环境

在运行 `demo.py` 之前，需要进行一些必要的运行环境设置，如固定随机种子(保证实验可复现性)、设置输出目录并初始化日志打印模块(保存重要实验数据)。

``` py title="examples/demo/demo.py"
if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_example"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
```

完成上述步骤之后，`demo.py` 已经搭好了必要框架。接下来介绍如何基于自己具体的需求，对 `ppsci.*` 下的其他模块进行开发或者复用，以最终在 `demo.py` 中使用。

### 2.3 构建模型

#### 2.3.1 构建已有模型

PaddleScience 内置了一些常见的模型，如 `MLP` 模型，如果您想使用这些内置的模型，可以直接调用 [`ppsci.arch.*`](./api/arch.md) 下的 API，并填入模型实例化所需的参数，即可快速构建模型。

``` py title="examples/demo/demo.py"
# create a MLP model
model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 9, 50, "tanh")
```

上述代码实例化了一个 `MLP` 全连接模型，其输入数据为两个字段：`"x"`、`"y"`，输出数据为三个字段：`"u"`、`"v"`、`"w"`；模型具有 $9$ 层隐藏层，每层的神经元个数为 $50$ 个，每层隐藏层使用的激活函数均为 $\tanh$ 双曲正切函数。

#### 2.3.2 构建新的模型

当 PaddleScience 内置的模型无法满足您的需求时，您就可以通过新增模型文件并编写模型代码的方式，使用您自定义的模型，步骤如下：

1. 在 `ppsci/arch/` 文件夹下新建模型结构文件，以 `new_model.py` 为例。
2. 在 `new_model.py` 文件中导入 PaddleScience 的模型基类所在的模块 `base`，并从 `base.Arch` 派生出您想创建的新模型类（以
`Class NewModel` 为例）。

    ``` py title="ppsci/arch/new_model.py"
    from ppsci.arch import base

    class NewModel(base.Arch):
        def __init__(self, ...):
            ...
            # initialization

        def forward(self, ...):
            ...
            # forward
    ```

3. 编写 `NewModel.__init__` 方法，其被用于模型创建时的初始化操作，包括模型层、参数变量初始化；然后再编写 `NewModel.forward` 方法，其定义了模型从接受输入、计算输出这一过程。以 `MLP.__init__` 和 `MLP.forward` 为例，如下所示。

    === "MLP.\_\_init\_\_"

        ``` py
        --8<--
        ppsci/arch/mlp.py:86:151
        --8<--
        ```

    === "MLP.forward"

        ``` py
        --8<--
        ppsci/arch/mlp.py:153:180
        --8<--
        ```

4. 在 `ppsci/arch/__init__.py` 中导入编写的新模型类 `NewModel`，并添加到 `__all__` 中

    ``` py title="ppsci/arch/__init__.py" hl_lines="3 8"
    ...
    ...
    from ppsci.arch.new_model import NewModel

    __all__ = [
        ...,
        ...,
        "NewModel",
    ]
    ```

完成上述新模型代码编写的工作之后，在 `demo.py` 中，就能通过调用 `ppsci.arch.NewModel`，实例化刚才编写的模型，如下所示。

``` py title="examples/demo/demo.py"
model = ppsci.arch.NewModel(...)
```

### 2.4 构建方程

如果您的案例问题中涉及到方程计算，那么可以选择使用 PaddleScience 内置的方程，或者编写自己的方程。

#### 2.4.1 构建已有方程

PaddleScience 内置了一些常见的方程，如 `NavierStokes` 方程，如果您想使用这些内置的方程，可以直接
调用 [`ppsci.equation.*`](./api/equation.md) 下的 API，并填入方程实例化所需的参数，即可快速构建方程。

``` py title="examples/demo/demo.py"
# create a Vibration equation
viv_equation = ppsci.equation.Vibration(2, -4, 0)
```

#### 2.4.2 构建新的方程

当 PaddleScience 内置的方程无法满足您的需求时，您也可以通过新增方程文件并编写方程代码的方式，使用您自定义的方程。

假设需要计算的方程公式如下所示。

$$
\begin{cases}
    \begin{align}
        \dfrac{\partial u}{\partial x} + \dfrac{\partial u}{\partial y} &= u + 1, \tag{1} \\
        \dfrac{\partial v}{\partial x} + \dfrac{\partial v}{\partial y} &= v. \tag{2}
    \end{align}
\end{cases}
$$

> 其中 $x$, $y$ 为模型输入，表示$x$、$y$轴坐标；$u=u(x,y)$、$v=v(x,y)$ 是模型输出，表示 $(x,y)$ 处的 $x$、$y$ 轴方向速度。

首先我们需要将上述方程进行适当移项，将含有变量、函数的项移动到等式左侧，含有常数的项移动到等式右侧，方便后续转换成程序代码，如下所示。

$$
\begin{cases}
    \begin{align}
        \dfrac{\partial u}{\partial x} +  \dfrac{\partial u}{\partial y} - u &= 1, \tag{3}\\
        \dfrac{\partial v}{\partial x} +  \dfrac{\partial v}{\partial y} - v &= 0. \tag{4}
    \end{align}
\end{cases}
$$

然后就可以将上述移项后的方程组根据以下步骤转换成对应的程序代码。

1. 在 `ppsci/equation/pde/` 下新建方程文件。如果您的方程并不是 PDE 方程，那么需要新建一个方程类文件夹，比如在 `ppsci/equation/` 下新建 `ode` 文件夹，再将您的方程文件放在 `ode` 文件夹下。此处以PDE类的方程 `new_pde.py` 为例。

2. 在 `new_pde.py` 文件中导入 PaddleScience 的方程基类所在模块 `base`，并从 `base.PDE` 派生 `Class NewPDE`。

    ``` py title="ppsci/equation/pde/new_pde.py"
    from ppsci.equation.pde import base

    class NewPDE(base.PDE):
    ```

3. 编写 `__init__` 代码，用于方程创建时的初始化，在其中定义必要的变量和公式计算过程。PaddleScience 支持使用 sympy 符号计算库创建方程和直接使用 python 函数编写方程，两种方式如下所示。

    === "sympy expression"

        ``` py title="ppsci/equation/pde/new_pde.py"
        from ppsci.equation.pde import base

        class NewPDE(base.PDE):
            def __init__(self):
                x, y = self.create_symbols("x y") # 创建自变量 x, y
                u = self.create_function("u", (x, y))  # 创建关于自变量 (x, y) 的函数 u(x,y)
                v = self.create_function("v", (x, y))  # 创建关于自变量 (x, y) 的函数 v(x,y)

                expr1 = u.diff(x) + u.diff(y) - u  # 对应等式(3)左侧表达式
                expr2 = v.diff(x) + v.diff(y) - v  # 对应等式(4)左侧表达式

                self.add_equation("expr1", expr1)  # 将expr1 的 sympy 表达式对象添加到 NewPDE 对象的公式集合中
                self.add_equation("expr2", expr2)  # 将expr2 的 sympy 表达式对象添加到 NewPDE 对象的公式集合中
        ```

    === "python function"

        ``` py title="ppsci/equation/pde/new_pde.py"
        from ppsci.autodiff import jacobian

        from ppsci.equation.pde import base

        class NewPDE(base.PDE):
            def __init__(self):
                def expr1_compute_func(out):
                    x, y = out["x"], out["y"]  # 从 out 数据字典中取出自变量 x, y 的数据值
                    u = out["u"]  # 从 out 数据字典中取出因变量 u 的函数值

                    expr1 = jacobian(u, x) + jacobian(u, y) - u  # 对应等式(3)左侧表达式计算过程
                    return expr1  # 返回计算结果值

                def expr2_compute_func(out):
                    x, y = out["x"], out["y"]  # 从 out 数据字典中取出自变量 x, y 的数据值
                    v = out["v"]  # 从 out 数据字典中取出因变量 v 的函数值

                    expr2 = jacobian(v, x) + jacobian(v, y) - v  # 对应等式(4)左侧表达式计算过程
                    return expr2

                self.add_equation("expr1", expr1_compute_func)  # 将 expr1 的计算函数添加到 NewPDE 对象的公式集合中
                self.add_equation("expr2", expr2_compute_func)  # 将 expr2 的计算函数添加到 NewPDE 对象的公式集合中
        ```

4. 在 `ppsci/equation/__init__.py` 中导入编写的新方程类，并添加到 `__all__` 中

    ``` py title="ppsci/equation/__init__.py" hl_lines="3 8"
    ...
    ...
    from ppsci.equation.pde.new_pde import NewPDE

    __all__ = [
        ...,
        ...,
        "NewPDE",
    ]
    ```

完成上述新方程代码编写的工作之后，我们就能像 PaddleScience 内置方程一样，以 `ppsci.equation.NewPDE` 的方式，调用我们编写的新方程类，并用于创建方程实例。

在方程构建完毕后之后，我们需要将所有方程包装为到一个字典中

``` py title="examples/demo/demo.py"
new_pde = ppsci.equation.NewPDE(...)
equation = {..., "newpde": new_pde}
```

### 2.5 构建几何模块[可选]

模型训练、验证时所用的输入、标签数据的来源，根据具体案例场景的不同而变化。大部分基于 PINN 的案例，其数据来自几何形状内部、表面采样得到的坐标点、法向量、SDF 值；而基于数据驱动的方法，其输入、标签数据大多数来自于外部文件，或通过 numpy 等第三方库构造的存放在内存中的数据。本章节主要对第一种情况所需的几何模块进行介绍，第二种情况则不一定需要几何模块，其构造方式可以参考 [#2.6 构建约束条件](#2.6)。

#### 2.5.1 构建已有几何

PaddleScience 内置了几类常用的几何形状，包括简单几何、复杂几何，如下所示。

| 几何调用方式 | 含义 |
| -- | -- |
|`ppsci.geometry.Interval`| 1 维线段几何|
|`ppsci.geometry.Disk`| 2 维圆面几何|
|`ppsci.geometry.Polygon`| 2 维多边形几何|
|`ppsci.geometry.Rectangle` | 2 维矩形几何|
|`ppsci.geometry.Triangle` | 2 维三角形几何|
|`ppsci.geometry.Cuboid`  | 3 维立方体几何|
|`ppsci.geometry.Sphere`   | 3 维圆球几何|
|`ppsci.geometry.Mesh`    | 3 维 Mesh 几何|
|`ppsci.geometry.PointCloud`     | 点云几何|
|`ppsci.geometry.TimeDomain`      | 1 维时间几何(常用于瞬态问题)|
|`ppsci.geometry.TimeXGeometry`        | 1 + N 维带有时间的几何(常用于瞬态问题)|

以计算域为 2 维矩形几何为例，实例化一个 x 轴边长为2，y 轴边长为 1，且左下角为点 (-1,-3) 的矩形几何代码如下：

``` py title="examples/demo/demo.py"
LEN_X, LEN_Y = 2, 1  # 定义矩形边长
rect = ppsci.geometry.Rectangle([-1, -3], [-1 + LEN_X, -3 + LEN_Y])  # 通过左下角、右上角对角线坐标构造矩形
```

其余的几何体构造方法类似，参考 API 文档的 [ppsci.geometry](./api/geometry.md) 部分即可。

#### 2.5.2 构建新的几何

下面以构建一个新的几何体 —— 2 维椭圆（无旋转）为例进行介绍。

1. 首先我们需要在二维几何的代码文件 `ppsci/geometry/geometry_2d.py` 中新建椭圆类 `Ellipse`，并制定其直接父类为 `geometry.Geometry` 几何基类。
然后根据椭圆的代数表示公式：$\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} = 1$，可以发现表示一个椭圆需要记录其圆心坐标 $(x_0,y_0)$、$x$ 轴半径 $a$、$y$ 轴半径 $b$。因此该椭圆类的代码如下所示。

    ``` py title="ppsci/geometry/geometry_2d.py"
    class Ellipse(geometry.Geometry):
        def __init__(self, x0: float, y0: float, a: float, b: float)
            self.center = np.array((x0, y0), dtype=paddle.get_default_dtype())
            self.a = a
            self.b = b
    ```

2. 为椭圆类编写必要的基础方法，如下所示。

    - 判断给定点集是否在椭圆内部

        ``` py title="ppsci/geometry/geometry_2d.py"
        def is_inside(self, x):
            return ((x / self.center) ** 2).sum(axis=1) < 1
        ```

    - 判断给定点集是否在椭圆边界上

        ``` py title="ppsci/geometry/geometry_2d.py"
        def on_boundary(self, x):
            return np.isclose(((x / self.center) ** 2).sum(axis=1), 1)
        ```

    - 在椭圆内部点随机采样(此处使用“拒绝采样法”实现)

        ``` py title="ppsci/geometry/geometry_2d.py"
        def random_points(self, n, random="pseudo"):
            res_n = n
            result = []
            max_radius = self.center.max()
            while (res_n < n):
                rng = sampler.sample(n, 2, random)
                r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
                x = np.sqrt(r) * np.cos(theta)
                y = np.sqrt(r) * np.sin(theta)
                candidate = max_radius * np.stack((x, y), axis=1) + self.center
                candidate = candidate[self.is_inside(candidate)]
                if len(candidate) > res_n:
                    candidate = candidate[: res_n]

                result.append(candidate)
                res_n -= len(candidate)
            result = np.concatenate(result, axis=0)
            return result
        ```

    - 在椭圆边界随机采样(此处基于椭圆参数方程实现)

        ``` py title="ppsci/geometry/geometry_2d.py"
        def random_boundary_points(self, n, random="pseudo"):
            theta = 2 * np.pi * sampler.sample(n, 1, random)
            X = np.concatenate((self.a * np.cos(theta),self.b * np.sin(theta)), axis=1)
            return X + self.center
        ```

3. 在 `ppsci/geometry/__init__.py` 中加入椭圆类 `Ellipse`，如下所示。

    ``` py title="ppsci/geometry/__init__.py" hl_lines="3 8"
    ...
    ...
    from ppsci.geometry.geometry_2d import Ellipse

    __all__ = [
        ...,
        ...,
        "Ellipse",
    ]
    ```

完成上述实现之后，我们就能以如下方式实例化椭圆类。同样地，建议将所有几何类实例包装在一个字典中，方便后续索引。

``` py title="examples/demo/demo.py"
ellipse = ppsci.geometry.Ellipse(0, 0, 2, 1)
geom = {..., "ellipse": ellipse}
```

### 2.6 构建约束条件

无论是 PINNs 方法还是数据驱动方法，它们总是需要利用数据来指导网络模型的训练，而这一过程在 PaddleScience 中由 `Constraint`（约束）模块负责。

#### 2.6.1 构建已有约束

PaddleScience 内置了一些常见的约束，如下所示。

|约束名称|功能|
|--|--|
|`ppsci.constraint.BoundaryConstraint`|边界约束|
|`ppsci.constraint.InitialConstraint` |内部点初值约束|
|`ppsci.constraint.IntegralConstraint` |边界积分约束|
|`ppsci.constraint.InteriorConstraint`|内部点约束|
|`ppsci.constraint.PeriodicConstraint`   |边界周期约束|
|`ppsci.constraint.SupervisedConstraint` |监督数据约束|

如果您想使用这些内置的约束，可以直接调用 [`ppsci.constraint.*`](./api/constraint.md) 下的 API，并填入约束实例化所需的参数，即可快速构建约束条件。

``` py title="examples/demo/demo.py"
# create a SupervisedConstraint
sup_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    {"eta": lambda out: out["eta"], **equation["VIV"].equations},
    name="Sup",
)
```

约束的参数填写方式，请参考对应的 API 文档参数说明和样例代码。

#### 2.6.2 构建新的约束

当 PaddleScience 内置的约束无法满足您的需求时，您也可以通过新增约束文件并编写约束代码的方式，使用您自
定义的约束，步骤如下：

1. 在 `ppsci/constraint` 下新建约束文件（此处以约束 `new_constraint.py` 为例）

2. 在 `new_constraint.py` 文件中导入 PaddleScience 的约束基类所在模块 `base`，并让创建的新约束
类（以 `Class NewConstraint` 为例）从 `base.PDE` 继承

    ``` py title="ppsci/constraint/new_constraint.py"
    from ppsci.constraint import base

    class NewConstraint(base.Constraint):
    ```

3. 编写 `__init__` 方法，用于约束创建时的初始化。

    ``` py title="ppsci/constraint/new_constraint.py"
    from ppsci.constraint import base

    class NewConstraint(base.Constraint):
        def __init__(self, ...):
            ...
            # initialization
    ```

4. 在 `ppsci/constraint/__init__.py` 中导入编写的新约束类，并添加到 `__all__` 中

    ``` py title="ppsci/constraint/__init__.py" hl_lines="3 8"
    ...
    ...
    from ppsci.constraint.new_constraint import NewConstraint

    __all__ = [
        ...,
        ...,
        "NewConstraint",
    ]
    ```

完成上述新约束代码编写的工作之后，我们就能像 PaddleScience 内置约束一样，以 `ppsci.constraint.NewConstraint` 的方式，调用我们编写的新约束类，并用于创建约束实例。

``` py title="examples/demo/demo.py"
new_constraint = ppsci.constraint.NewConstraint(...)
constraint = {..., new_constraint.name: new_constraint}
```

### 2.7 定义超参数

在模型开始训练前，需要定义一些训练相关的超参数，如训练轮数、学习率等，如下所示。

``` py title="examples/demo/demo.py"
EPOCHS = 10000
LEARNING_RATE = 0.001
```

### 2.8 构建优化器

模型训练时除了模型本身，还需要定义一个用于更新模型参数的优化器，如下所示。

``` py title="examples/demo/demo.py"
optimizer = ppsci.optimizer.Adam(0.001)(model)
```

### 2.9 构建评估器[可选]

#### 2.9.1 构建已有评估器

PaddleScience 内置了一些常见的评估器，如下所示。

|评估器名称|功能|
|--|--|
|`ppsci.validator.GeometryValidator`|几何评估器|
|`ppsci.validator.SupervisedValidator` |监督数据评估器|

如果您想使用这些内置的评估器，可以直接调用 [`ppsci.validate.*`](./api/validate.md) 下的 API，并填入评估器实例化所需的参数，即可快速构建评估器。

``` py title="examples/demo/demo.py"
# create a SupervisedValidator
eta_mse_validator = ppsci.validate.SupervisedValidator(
    valid_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    {"eta": lambda out: out["eta"], **equation["VIV"].equations},
    metric={"MSE": ppsci.metric.MSE()},
    name="eta_mse",
)
```

#### 2.9.2 构建新的评估器

当 PaddleScience 内置的评估器无法满足您的需求时，您也可以通过新增评估器文件并编写评估器代码的方式，使
用您自定义的评估器，步骤如下：

1. 在 `ppsci/validate` 下新建评估器文件（此处以 `new_validator.py` 为例）。

2. 在 `new_validator.py` 文件中导入 PaddleScience 的评估器基类所在模块 `base`，并让创建的新评估器类（以 `Class NewValidator` 为例）从 `base.Validator` 继承。

    ``` py title="ppsci/validate/new_validator.py"
    from ppsci.validate import base

    class NewValidator(base.Validator):
    ```

3. 编写 `__init__` 代码，用于评估器创建时的初始化

    ``` py title="ppsci/validate/new_validator.py"
    from ppsci.validate import base

    class NewValidator(base.Validator):
        def __init__(self, ...):
            ...
            # initialization
    ```

4. 在 `ppsci/validate/__init__.py` 中导入编写的新评估器类，并添加到 `__all__` 中。

    ``` py title="ppsci/validate/__init__.py" hl_lines="3 8"
    ...
    ...
    from ppsci.validate.new_validator import NewValidator

    __all__ = [
        ...,
        ...,
        "NewValidator",
    ]
    ```

完成上述新评估器代码编写的工作之后，我们就能像 PaddleScience 内置评估器一样，以 `ppsci.validate.NewValidator` 的方式，调用我们编写的新评估器类，并用于创建评估器实例。同样地，在评估器构建完毕后之后，建议将所有评估器包装到一个字典中方便后续索引。

``` py title="examples/demo/demo.py"
new_validator = ppsci.validate.NewValidator(...)
validator = {..., new_validator.name: new_validator}
```

### 2.10 构建可视化器[可选]

PaddleScience 内置了一些常见的可视化器，如 `VisualizerVtu` 可视化器等，如果您想使用这些内置的可视
化器，可以直接调用 [`ppsci.visualizer.*`](./api/visualize.md) 下的 API，并填入可视化器实例化所需的
参数，即可快速构建模型。

``` py title="examples/demo/demo.py"
# manually collate input data for visualization,
# interior+boundary
vis_points = {}
for key in vis_interior_points:
    vis_points[key] = np.concatenate(
        (vis_interior_points[key], vis_boundary_points[key])
    )

visualizer = {
    "visualize_u_v": ppsci.visualize.VisualizerVtu(
        vis_points,
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
        prefix="result_u_v",
    )
}
```

如需新增可视化器，步骤与其他模块的新增方法类似，此处不再赘述。

### 2.11 构建Solver

[`Solver`](./api/solver.md) 是 PaddleScience 负责调用训练、评估、可视化的全局管理类。在训练开始前，需要把构建好的模型、约束、优化器等实例传给 `Solver` 以实例化，再调用它的内置方法进行训练、评估、可视化。

``` py title="examples/demo/demo.py"
# initialize solver
solver = ppsci.solver.Solver(
    model,
    constraint,
    output_dir,
    optimizer,
    lr_scheduler,
    EPOCHS,
    iters_per_epoch,
    eval_during_train=True,
    eval_freq=eval_freq,
    equation=equation,
    validator=validator,
    visualizer=visualizer,
)
```

### 2.12 编写配置文件[重要]

经过上述步骤的开发，案例代码的主要部分已经完成。
当我们想基于这份代码运行一些调优实验，从而得到更好的结果，或更好地对运行参数设置进行管理，
则可以利用 PaddleScience 提供的配置管理系统，将实验运行参数从代码中分离出来，写到 `yaml` 格式的配置文件中，从而更好的管理、记录、调优实验。

以 `bracket` 案例代码为例，在运行时我们需要在适当的位置设置方程参数、STL 文件路径、训练轮数、`batch_size`、随机种子、学习率等超参数，如下所示

``` shell
...
# specify parameters
LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
MU_C = 0.01 * MU
LAMBDA_ = LAMBDA_ / MU_C
MU = MU / MU_C
SIGMA_NORMALIZATION = CHARACTERISTIC_LENGTH / (
    CHARACTERISTIC_DISPLACEMENT * MU_C
)
T = -4.0e4 * SIGMA_NORMALIZATION
...
...
# set geometry
support = ppsci.geometry.Mesh(SUPPORT_PATH)
bracket = ppsci.geometry.Mesh(BRACKET_PATH)
aux_lower = ppsci.geometry.Mesh(AUX_LOWER_PATH)
aux_upper = ppsci.geometry.Mesh(AUX_UPPER_PATH)
cylinder_hole = ppsci.geometry.Mesh(CYLINDER_HOLE_PATH)
cylinder_lower = ppsci.geometry.Mesh(CYLINDER_LOWER_PATH)
cylinder_upper = ppsci.geometry.Mesh(CYLINDER_UPPER_PATH)
...
...
# initialize solver
solver = ppsci.solver.Solver(
    model,
    constraint,
    output_dir,
    optimizer,
    lr_scheduler,
    epochs,
    iters_per_epoch,
    save_freq=save_freq,
    log_freq=log_freq,
    eval_during_train=eval_during_train,
    eval_freq=eval_freq,
    seed=seed,
    equation=equation,
    geom=geom,
    validator=validator,
    visualizer=visualizer,
    checkpoint_path=checkpoint_path,
    eval_with_no_grad=eval_with_no_grad,
)
...
```

这些参数在实验过程中随时可能作为变量而被手动调整，在调整过程中如何避免频繁修改源代码导致试验记录混乱、保障记录完整可追溯便是一大问题，因此 PaddleScience 提供了基于 hydra + omegaconf 的
配置文件管理系统来解决这一问题。

将已有的代码修改成配置文件控制的方式非常简单，只需要将必要的参数写到 `yaml` 文件中，然后通过 hydra 在程序运行时读取、解析该文件，通过其内容控制实验运行即可。具体包含以下几个步骤。

1. 假设代码文件为 `demo.py`，则需在代码文件所在目录下新建 `conf` 文件夹，并在 `conf` 下新建与 `demo.py` 同名的 `demo.yaml` 文件。
2. 参考 `examples/bracket/conf/bracket.yaml`，将 `demo.py` 中必要的超参数按照其语义填写到 `demo.yaml` 的各个层级的配置中，如 `mode`、`output_dir`、`seed`、方程参数、文件路径等参数为通用参数，直接填写在一级层级；而模型结构参数、训练轮数等参数则只与模型、训练相关，只需分别填写在 `MODEL`、`TRAIN` 层级下即可（`EVAL` 层级同理）。
3. 将已有的 `train` 和 `eval` 函数修改为接受一个参数 `cfg`（`cfg` 即为读取进来的 `yaml` 文件里的内容，并以字典的形式存储），并将其内部的超参数统一改为通过 `cfg.xxx` 获取而非原先的直接设置为数字或字符串。
4. 新建一个 `main` 函数（同样接受且只接受一个 `cfg` 参数），它负责根据 `cfg.mode` 来调用 `train` 或 `eval` 函数。最后给 `main` 函数加上装饰器 `@hydra.main(version_base=None, config_path="./conf", config_name="demo.yaml")`。
5. 在主程序的启动入口 `if __name__ == "__main__":` 中启动 `main()` 即可。

全部填写完毕后，`demo.yaml` 大致如下所示。

``` yaml hl_lines="4 26-29 32-35 38-44 58-72 75-97 100-104"
hydra:
  run:
    # dynamic output directory according to running time and override name
    dir: outputs_bracket/${now:%Y-%m-%d}/${now:%H-%M-%S}/${hydra.job.override_dirname} # (1)
  job:
    name: ${mode} # name of logfile
    chdir: false # keep current working direcotry unchaned
    config:
      override_dirname:
        exclude_keys:
          - TRAIN.checkpoint_path
          - TRAIN.pretrained_model_path
          - EVAL.pretrained_model_path
          - mode
          - output_dir
          - log_freq
  callbacks:
    init_callback:
      _target_: ppsci.utils.callbacks.InitCallback
  sweep:
    # output directory for multirun
    dir: ${hydra.run.dir}
    subdir: ./

# general settings
mode: train # running mode: train/eval (2)
seed: 2023 # (3)
output_dir: ${hydra:run.dir} # (4)
log_freq: 20 # (5)

# set working condition (6)
NU: 0.3
E: 100.0e9
CHARACTERISTIC_LENGTH: 1.0
CHARACTERISTIC_DISPLACEMENT: 1.0e-4

# set geometry file path (7)
SUPPORT_PATH: ./stl/support.stl
BRACKET_PATH: ./stl/bracket.stl
AUX_LOWER_PATH: ./stl/aux_lower.stl
AUX_UPPER_PATH: ./stl/aux_upper.stl
CYLINDER_HOLE_PATH: ./stl/cylinder_hole.stl
CYLINDER_LOWER_PATH: ./stl/cylinder_lower.stl
CYLINDER_UPPER_PATH: ./stl/cylinder_upper.stl

# set evaluate data path (8)
DEFORMATION_X_PATH: ./data/deformation_x.txt
DEFORMATION_Y_PATH: ./data/deformation_y.txt
DEFORMATION_Z_PATH: ./data/deformation_z.txt
NORMAL_X_PATH: ./data/normal_x.txt
NORMAL_Y_PATH: ./data/normal_y.txt
NORMAL_Z_PATH: ./data/normal_z.txt
SHEAR_XY_PATH: ./data/shear_xy.txt
SHEAR_XZ_PATH: ./data/shear_xz.txt
SHEAR_YZ_PATH: ./data/shear_yz.txt

# model settings (9)
MODEL:
  disp_net:
    input_keys: ["x", "y", "z"]
    output_keys: ["u", "v", "w"]
    num_layers: 6
    hidden_size: 512
    activation: "silu"
    weight_norm: true
  stress_net:
    input_keys: ["x", "y", "z"]
    output_keys: ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"]
    num_layers: 6
    hidden_size: 512
    activation: "silu"
    weight_norm: true

# training settings
TRAIN: # (10)
  epochs: 2000 # (11)
  iters_per_epoch: 1000 # (12)
  save_freq: 20 # (13)
  eval_during_train: true # (14)
  eval_freq: 20 # (15)
  lr_scheduler: # (16)
    epochs: ${TRAIN.epochs}
    iters_per_epoch: ${TRAIN.iters_per_epoch}
    learning_rate: 0.001
    gamma: 0.95
    decay_steps: 15000
    by_epoch: false
  batch_size: # (17)
    bc_back: 1024
    bc_front: 128
    bc_surface: 4096
    support_interior: 2048
    bracket_interior: 1024
  weight: # (18)
    bc_back: {"u": 10, "v": 10, "w": 10}
  pretrained_model_path: null # (19)
  checkpoint_path: null # (20)

# evaluation settings
EVAL: # (21)
  pretrained_model_path: null # (22)
  eval_with_no_grad: true # (23)
  batch_size: # (24)
    sup_validator: XXX
```

### 2.13 训练

PaddleScience 模型的训练只需调用一行代码。

``` py title="examples/demo/demo.py"
solver.train()
```

### 2.14 评估

PaddleScience 模型的评估只需调用一行代码。

``` py title="examples/demo/demo.py"
solver.eval()
```

### 2.15 可视化[可选]

若 `Solver` 实例化时传入了 `visualizer` 参数，则 PaddleScience 模型的可视化只需调用一行代码。

``` py title="examples/demo/demo.py"
solver.visualize()
```

!!! tip "可视化方案"

    对于一些复杂的案例，`Visualizer` 的编写成本并不低，并且不是任何数据类型都可以进行方便的可视化。因此可以在训练完成之后，手动构建用于预测的数据字典，再使用 `solver.predict` 得到模型预测结果，最后利用 `matplotlib` 等第三方库，对预测结果进行可视化并保存。

## 3. 编写文档

除了案例代码，PaddleScience 同时存放了对应案例的详细文档，使用 Markdown + [Mkdocs-Material](https://squidfunk.github.io/mkdocs-material/) 进行编写和渲染，撰写文档步骤如下。

### 3.1 安装必要依赖包

文档撰写过程中需进行即时渲染，预览文档内容以检查撰写的内容是否有误。因此需要按照如下命令，安装 mkdocs 相关依赖包。

``` sh
pip install -r docs/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 撰写文档内容

PaddleScience 文档基于 [Mkdocs-Material](https://squidfunk.github.io/mkdocs-material/)、[PyMdown](https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/) 等插件进行编写，其在 Markdown 语法基础上支持了多种扩展性功能，能极大地提升文档的美观程度和阅读体验。建议参考超链接内的文档内容，选择合适的功能辅助文档撰写。

### 3.3 使用 markdownlint 格式化文档[可选]

如果您使用的开发环境为 VSCode，则推荐安装 [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) 扩展。安装完毕后在编写完的文档内：点击右键-->格式化文档即可。

### 3.4 预览文档

在 `PaddleScience/` 目录下执行以下命令，等待构建完成后，点击显示的链接进入本地网页预览文档内容。

``` sh
mkdocs serve
```

``` log
# ====== 终端打印信息如下 ======
# INFO     -  Building documentation...
# INFO     -  Cleaning site directory
# INFO     -  Documentation built in 20.95 seconds
# INFO     -  [07:39:35] Watching paths for changes: 'docs', 'mkdocs.yml'
# INFO     -  [07:39:35] Serving on http://127.0.0.1:8000/PaddlePaddle/PaddleScience/
# INFO     -  [07:39:41] Browser connected: http://127.0.0.1:58903/PaddlePaddle/PaddleScience/
# INFO     -  [07:40:41] Browser connected: http://127.0.0.1:58903/PaddlePaddle/PaddleScience/zh/development/
```

!!! tip "手动指定服务地址和端口号"

    若默认端口号 8000 被占用，则可以手动指定服务部署的地址和端口，示例如下。

    ``` sh
    # 指定 127.0.0.1 为地址，8687 为端口号
    mkdocs serve -a 127.0.0.1:8687
    ```

## 4. 整理代码并提交

### 4.1 安装 pre-commit

PaddleScience 是一个开源的代码库，由多人共同参与开发，因此为了保持最终合入的代码风格整洁、一致，
PaddleScience 使用了包括 [isort](https://github.com/PyCQA/isort#installing-isort)、[black](https://github.com/psf/black) 等自动化代码检查、格式化插件，
让 commit 的代码遵循 python [PEP8](https://pep8.org/) 代码风格规范。

因此在 commit 您的代码之前，请务必先执行以下命令安装 `pre-commit`。

``` sh
pip install pre-commit
pre-commit install
```

关于 pre-commit 的详情请参考 [Paddle 代码风格检查指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/codestyle_check_guide_cn.html)

### 4.2 整理代码

在完成范例编写与训练后，确认结果无误，就可以整理代码。
使用 git 命令将所有新增、修改的代码文件以及必要的文档、图片等一并上传到自己仓库的 `dev_model` 分支上。

### 4.3 提交 pull request

在 github 网页端切换到 `dev_model` 分支，并点击 "Contribute"，再点击 "Open pull request" 按钮，
将含有您的代码、文档、图片等内容的 `dev_model` 分支作为合入请求贡献到 PaddleScience。
