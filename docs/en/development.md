# Development Guide

This document describes how to develop code based on the PaddleScience suite and ultimately contribute to the PaddleScience suite.

Before starting PaddleScience-related paper reproduction and API development tasks, you need to submit an RFC document. Please refer to: [PaddleScience RFC Template](https://github.com/PaddlePaddle/community/blob/master/rfcs/Science/template.md)

## 1. Preparation

1. Fork PaddleScience to **your own repository** on the web page.
2. Clone PaddleScience from **your own repository** to local and enter the directory.

    ``` sh
    git clone -b develop https://github.com/USER_NAME/PaddleScience.git
    cd PaddleScience
    ```

    Please fill in your github username in the `USER_NAME` field in the `clone` command above.

3. Install necessary dependent packages.

    ``` sh
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

4. Create a new branch based on the current `develop` branch (assume the new branch name is `dev_model`).

    ``` sh
    git checkout -b "dev_model"
    ```

5. Add the PaddleScience directory to the system environment variable `PYTHONPATH`.

    ``` sh
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```

6. Execute the following code to verify whether the basic functions of installed PaddleScience are normal.

    ``` sh
    python -c "import ppsci; ppsci.run_check()"
    ```

    If PaddleScience is installed successfully.✨ 🍰 ✨ appears, the installation verification is successful.

7. Install pre-commit [Important]

    PaddleScience is an open source code base developed by many people. Therefore, in order to keep the final merged code style clean and consistent,
    PaddleScience uses automated code checking and formatting plugins including [isort](https://github.com/PyCQA/isort#installing-isort), [black](https://github.com/psf/black), etc.,
    to make the committed code follow the python [PEP8](https://pep8.org/) code style specification.

    Therefore, before committing your code, please be sure to execute the following command in the `PaddleScience/` directory to install `pre-commit`, otherwise the submitted PR will be detected by code-style that the code is not formatted and cannot be merged.

    ``` sh
    python -m pip install pre-commit
    pre-commit install
    ```

    If you have already committed the code, you can manually execute the pre-commit command after installing the above pre-commit to format the code: `pre-commit run --files your/committed/code/file/or/folder`, then manually `git add` the modified files, and then `git commit`.

    For details on pre-commit, please refer to [Paddle Code Style Check Guide](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/codestyle_check_guide_cn.html).

## 2. Write Code

After completing the above preparation work, you can start developing your own case or function based on PaddleScience.

Assume that the path of the new case code file is: `PaddleScience/examples/demo/demo.py`. Next, we will introduce this process in detail.

### 2.1 Import Necessary Packages

All APIs provided by PaddleScience are under the `ppsci.*` module. Therefore, at the beginning of `demo.py`, you first need to import the `ppsci` top-level module, then import the log printing module `logger` to facilitate automatically recording logs to local files when printing logs, and finally import other necessary modules according to your own needs.

``` py title="examples/demo/demo.py"
import ppsci
from ppsci.utils import logger

# Import other necessary modules
# import ...
```

### 2.2 Set Running Environment

Before running `demo.py`, some necessary running environment settings are required, such as fixing random seeds (guaranteeing experiment reproducibility), setting output directories and initializing log printing modules (saving important experimental data).

``` py title="examples/demo/demo.py"
if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_example"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
```

After completing the above steps, `demo.py` has set up the necessary framework. Next, we will introduce how to develop or reuse other modules under `ppsci.*` based on your specific needs, so as to finally use them in `demo.py`.

### 2.3 Build Model

#### 2.3.1 Build Existing Model

PaddleScience has built-in some common models, such as the `MLP` model. If you want to use these built-in models, you can directly call the API under [`ppsci.arch.*`](./api/arch.md) and fill in the parameters required for model instantiation to quickly build the model.

``` py title="examples/demo/demo.py"
# create a MLP model
model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 9, 50, "tanh")
```

The above code instantiates an `MLP` fully connected model. Its input data has two fields: `"x"`, `"y"`, and output data has three fields: `"u"`, `"v"`, `"w"`; the model has $9$ hidden layers, each layer has $50$ neurons, and the activation function used by each hidden layer is the $\tanh$ hyperbolic tangent function.

#### 2.3.2 Build New Model

When the built-in models of PaddleScience cannot meet your needs, you can use your custom model by adding a model file and writing model code. The steps are as follows:

1. Create a new model structure file under the `ppsci/arch/` folder, taking `new_model.py` as an example.
2. In the `new_model.py` file, import the module `base` where the model base class of PaddleScience is located, and derive the new model class you want to create from `base.Arch` (taking `Class NewModel` as an example).

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

3. Write the `NewModel.__init__` method, which is used for initialization operations during model creation, including initialization of model layers and parameter variables; then write the `NewModel.forward` method, which defines the process of the model accepting input and calculating output. Taking `MLP.__init__` and `MLP.forward` as examples, as shown below.

    === "MLP.\_\_init\_\_"

        ``` py
        --8<--
        ppsci/arch/mlp.py:139:279
        --8<--
        ```

    === "MLP.forward"

        ``` py
        --8<--
        ppsci/arch/mlp.py:298:315
        --8<--
        ```

4. Import the written new model class `NewModel` in `ppsci/arch/__init__.py` and add it to `__all__`.

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

After completing the work of writing new model code above, in `demo.py`, you can instantiate the model just written by calling `ppsci.arch.NewModel`, as shown below.

``` py title="examples/demo/demo.py"
model = ppsci.arch.NewModel(...)
```

### 2.4 Build Equation

If your case problem involves equation calculation, you can choose to use PaddleScience's built-in equations or write your own equations.

#### 2.4.1 Build Existing Equation

PaddleScience has built-in some common equations, such as `NavierStokes` equation. If you want to use these built-in equations, you can directly call the API under [`ppsci.equation.*`](./api/equation.md) and fill in the parameters required for equation instantiation to quickly build the equation.

``` py title="examples/demo/demo.py"
# create a Vibration equation
viv_equation = ppsci.equation.Vibration(2, -4, 0)
```

#### 2.4.2 Build New Equation

When the built-in equations of PaddleScience cannot meet your needs, you can also use your custom equation by adding an equation file and writing equation code.

Assume the equation formula to be calculated is as follows.

$$
\begin{cases}
    \begin{align}
        \dfrac{\partial u}{\partial x} + \dfrac{\partial u}{\partial y} &= u + 1, \tag{1} \\
        \dfrac{\partial v}{\partial x} + \dfrac{\partial v}{\partial y} &= v. \tag{2}
    \end{align}
\end{cases}
$$

> Where $x$, $y$ are model inputs, representing x and y axis coordinates; $u=u(x,y)$, $v=v(x,y)$ are model outputs, representing x and y axis direction velocities at $(x,y)$.

First, we need to appropriately transpose the above equations, moving terms containing variables and functions to the left side of the equation, and terms containing constants to the right side of the equation, to facilitate subsequent conversion into program code, as shown below.

$$
\begin{cases}
    \begin{align}
        \dfrac{\partial u}{\partial x} +  \dfrac{\partial u}{\partial y} - u &= 1, \tag{3}\\
        \dfrac{\partial v}{\partial x} +  \dfrac{\partial v}{\partial y} - v &= 0. \tag{4}
    \end{align}
\end{cases}
$$

Then the above transposed equation system can be converted into corresponding program code according to the following steps.

1. Create a new equation file under `ppsci/equation/pde/`. If your equation is not a PDE equation, you need to create a new equation class folder, for example, create an `ode` folder under `ppsci/equation/`, and then place your equation file under the `ode` folder. Here take the PDE class equation `new_pde.py` as an example.

2. In the `new_pde.py` file, import the module `base` where the equation base class of PaddleScience is located, and derive `Class NewPDE` from `base.PDE`.

    ``` py title="ppsci/equation/pde/new_pde.py"
    from ppsci.equation.pde import base

    class NewPDE(base.PDE):
    ```

3. Write `__init__` code for initialization during equation creation, in which necessary variables and formula calculation processes are defined. PaddleScience supports creating equations using the sympy symbolic calculation library and writing equations directly using python functions. The two methods are as follows.

    === "sympy expression"

        ``` py title="ppsci/equation/pde/new_pde.py"
        from ppsci.equation.pde import base

        class NewPDE(base.PDE):
            def __init__(self):
                x, y = self.create_symbols("x y") # Create independent variables x, y
                u = self.create_function("u", (x, y))  # Create function u(x,y) regarding independent variables (x, y)
                v = self.create_function("v", (x, y))  # Create function v(x,y) regarding independent variables (x, y)

                expr1 = u.diff(x) + u.diff(y) - u  # Expression on the left side of equation (3)
                expr2 = v.diff(x) + v.diff(y) - v  # Expression on the left side of equation (4)

                self.add_equation("expr1", expr1)  # Add sympy expression object of expr1 to the formula collection of NewPDE object
                self.add_equation("expr2", expr2)  # Add sympy expression object of expr2 to the formula collection of NewPDE object
        ```

    === "python function"

        ``` py title="ppsci/equation/pde/new_pde.py"
        from ppsci.autodiff import jacobian

        from ppsci.equation.pde import base

        class NewPDE(base.PDE):
            def __init__(self):
                def expr1_compute_func(out):
                    x, y = out["x"], out["y"]  # Extract data values of independent variables x, y from out data dictionary
                    u = out["u"]  # Extract function value of dependent variable u from out data dictionary

                    expr1 = jacobian(u, x) + jacobian(u, y) - u  # Calculation process of expression on the left side of equation (3)
                    return expr1  # Return calculation result value

                def expr2_compute_func(out):
                    x, y = out["x"], out["y"]  # Extract data values of independent variables x, y from out data dictionary
                    v = out["v"]  # Extract function value of dependent variable v from out data dictionary

                    expr2 = jacobian(v, x) + jacobian(v, y) - v  # Calculation process of expression on the left side of equation (4)
                    return expr2

                self.add_equation("expr1", expr1_compute_func)  # Add calculation function of expr1 to the formula collection of NewPDE object
                self.add_equation("expr2", expr2_compute_func)  # Add calculation function of expr2 to the formula collection of NewPDE object
        ```

4. Import the written new equation class `NewPDE` in `ppsci/equation/__init__.py` and add it to `__all__`.

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

After completing the work of writing new equation code above, we can call the new equation class we wrote and use it to create an equation instance in the manner of `ppsci.equation.NewPDE`, just like PaddleScience built-in equations.

After the equation construction is completed, we need to wrap all equations into a dictionary.

``` py title="examples/demo/demo.py"
new_pde = ppsci.equation.NewPDE(...)
equation = {..., "newpde": new_pde}
```

### 2.5 Build Geometry Module [Optional]

The source of input and label data used during model training and validation varies according to specific case scenarios. Most PINN-based cases have data from coordinate points, normal vectors, and SDF values sampled inside or on the surface of geometric shapes; while for data-driven methods, most of their input and label data come from external files, or data stored in memory constructed by third-party libraries such as numpy. This chapter mainly introduces the geometry module required for the first case. The second case does not necessarily require a geometry module, and its construction method can refer to [#2.6 Build Constraint Condition](#26-build-constraint-condition).

#### 2.5.1 Build Existing Geometry

PaddleScience has built-in several types of common geometric shapes, including simple geometries and complex geometries, as shown below.

| Geometry Call Method | Meaning |
| -- | -- |
|`ppsci.geometry.Interval`| 1D Line Segment Geometry|
|`ppsci.geometry.Disk`| 2D Disk Geometry|
|`ppsci.geometry.Polygon`| 2D Polygon Geometry|
|`ppsci.geometry.Rectangle` | 2D Rectangle Geometry|
|`ppsci.geometry.Triangle` | 2D Triangle Geometry|
|`ppsci.geometry.Cuboid`  | 3D Cuboid Geometry|
|`ppsci.geometry.Sphere`   | 3D Sphere Geometry|
|`ppsci.geometry.Mesh`    | 3D Mesh Geometry|
|`ppsci.geometry.PointCloud`     | Point Cloud Geometry|
|`ppsci.geometry.TimeDomain`      | 1D Time Geometry (commonly used for transient problems)|
|`ppsci.geometry.TimeXGeometry`        | 1 + N D Geometry with Time (commonly used for transient problems)|

Taking the calculation domain as a 2D rectangle geometry as an example, the code to instantiate a rectangle geometry with x-axis side length of 2, y-axis side length of 1, and bottom-left corner at point (-1, -3) is as follows:

``` py title="examples/demo/demo.py"
LEN_X, LEN_Y = 2, 1  # Define rectangle side lengths
rect = ppsci.geometry.Rectangle([-1, -3], [-1 + LEN_X, -3 + LEN_Y])  # Construct rectangle by bottom-left and top-right diagonal coordinates
```

Other geometry construction methods are similar, please refer to the [ppsci.geometry](./api/geometry.md) part of the API document.

#### 2.5.2 Build New Geometry

The following introduces how to build a new geometry —— 2D ellipse (no rotation) as an example.

1. First, we need to create a new ellipse class `Ellipse` in the code file `ppsci/geometry/geometry_2d.py` of 2D geometry, and designate its direct parent class as `geometry.Geometry` geometry base class.
Then according to the algebraic representation formula of ellipse: $\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} = 1$, it can be found that representing an ellipse requires recording its center coordinates $(x_0,y_0)$, x-axis radius $a$, and y-axis radius $b$. Therefore, the code of this ellipse class is as follows.

    ``` py title="ppsci/geometry/geometry_2d.py"
    class Ellipse(geometry.Geometry):
        def __init__(self, x0: float, y0: float, a: float, b: float)
            self.center = np.array((x0, y0), dtype=paddle.get_default_dtype())
            self.a = a
            self.b = b
    ```

2. Write necessary basic methods for the ellipse class, as shown below.

    - Determine whether the given point set is inside the ellipse

        ``` py title="ppsci/geometry/geometry_2d.py"
        def is_inside(self, x):
            return ((x / self.center) ** 2).sum(axis=1) < 1
        ```

    - Determine whether the given point set is on the boundary of the ellipse

        ``` py title="ppsci/geometry/geometry_2d.py"
        def on_boundary(self, x):
            return np.isclose(((x / self.center) ** 2).sum(axis=1), 1)
        ```

    - Randomly sample inside the ellipse (implemented using "Rejection Sampling Method" here)

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

    - Randomly sample on the boundary of the ellipse (implemented based on ellipse parameter equation here)

        ``` py title="ppsci/geometry/geometry_2d.py"
        def random_boundary_points(self, n, random="pseudo"):
            theta = 2 * np.pi * sampler.sample(n, 1, random)
            X = np.concatenate((self.a * np.cos(theta),self.b * np.sin(theta)), axis=1)
            return X + self.center
        ```

3. Add the ellipse class `Ellipse` in `ppsci/geometry/__init__.py`, as shown below.

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

After completing the above implementation, we can instantiate the ellipse class in the following way. Similarly, it is recommended to wrap all geometry class instances in a dictionary for subsequent indexing.

``` py title="examples/demo/demo.py"
ellipse = ppsci.geometry.Ellipse(0, 0, 2, 1)
geom = {..., "ellipse": ellipse}
```

### 2.6 Build Constraint Condition

Whether it is the PINNs method or the data-driven method, they always need to use data to guide the training of the network model, and this process is responsible by the `Constraint` module in PaddleScience.

#### 2.6.1 Build Existing Constraint

PaddleScience has built-in some common constraints, as shown below.

|Constraint Name|Function|
|--|--|
|`ppsci.constraint.BoundaryConstraint`|Boundary Constraint|
|`ppsci.constraint.InitialConstraint` |Internal Point Initial Value Constraint|
|`ppsci.constraint.IntegralConstraint` |Boundary Integral Constraint|
|`ppsci.constraint.InteriorConstraint`|Internal Point Constraint|
|`ppsci.constraint.PeriodicConstraint`   |Boundary Periodic Constraint|
|`ppsci.constraint.SupervisedConstraint` |Supervised Data Constraint|

If you want to use these built-in constraints, you can directly call the API under [`ppsci.constraint.*`](./api/constraint.md) and fill in the parameters required for constraint instantiation to quickly build constraint conditions.

``` py title="examples/demo/demo.py"
# create a SupervisedConstraint
sup_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    {"eta": lambda out: out["eta"], **equation["VIV"].equations},
    name="Sup",
)
```

For the parameter filling method of constraints, please refer to the corresponding API document parameter description and sample code.

#### 2.6.2 Build New Constraint

When the built-in constraints of PaddleScience cannot meet your needs, you can also use your custom constraint by adding a constraint file and writing constraint code. The steps are as follows:

1. Create a new constraint file under `ppsci/constraint` (here take constraint `new_constraint.py` as an example)

2. In the `new_constraint.py` file, import the module `base` where the constraint base class of PaddleScience is located, and let the created new constraint class (take `Class NewConstraint` as an example) inherit from `base.PDE`.

    ``` py title="ppsci/constraint/new_constraint.py"
    from ppsci.constraint import base

    class NewConstraint(base.Constraint):
    ```

3. Write the `__init__` method for initialization during constraint creation.

    ``` py title="ppsci/constraint/new_constraint.py"
    from ppsci.constraint import base

    class NewConstraint(base.Constraint):
        def __init__(self, ...):
            ...
            # initialization
    ```

4. Import the written new constraint class `NewConstraint` in `ppsci/constraint/__init__.py` and add it to `__all__`.

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

After completing the work of writing new constraint code above, we can call the new constraint class we wrote and use it to create a constraint instance in the manner of `ppsci.constraint.NewConstraint`, just like PaddleScience built-in constraints.

``` py title="examples/demo/demo.py"
new_constraint = ppsci.constraint.NewConstraint(...)
constraint = {..., new_constraint.name: new_constraint}
```

### 2.7 Define Hyperparameters

Before the model starts training, some training-related hyperparameters need to be defined, such as training rounds, learning rate, etc., as shown below.

``` py title="examples/demo/demo.py"
EPOCHS = 10000
LEARNING_RATE = 0.001
```

### 2.8 Build Optimizer

In addition to the model itself, an optimizer for updating model parameters needs to be defined during model training, as shown below.

``` py title="examples/demo/demo.py"
optimizer = ppsci.optimizer.Adam(0.001)(model)
```

### 2.9 Build Validator [Optional]

#### 2.9.1 Build Existing Validator

PaddleScience has built-in some common validators, as shown below.

|Validator Name|Function|
|--|--|
|`ppsci.validator.GeometryValidator`|Geometry Validator|
|`ppsci.validator.SupervisedValidator` |Supervised Data Validator|

If you want to use these built-in validators, you can directly call the API under [`ppsci.validate.*`](./api/validate.md) and fill in the parameters required for validator instantiation to quickly build a validator.

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

#### 2.9.2 Build New Validator

When the built-in validators of PaddleScience cannot meet your needs, you can also use your custom validator by adding a validator file and writing validator code. The steps are as follows:

1. Create a new validator file under `ppsci/validate` (here take `new_validator.py` as an example).

2. In the `new_validator.py` file, import the module `base` where the validator base class of PaddleScience is located, and let the created new validator class (take `Class NewValidator` as an example) inherit from `base.Validator`.

    ``` py title="ppsci/validate/new_validator.py"
    from ppsci.validate import base

    class NewValidator(base.Validator):
    ```

3. Write `__init__` code for initialization during validator creation.

    ``` py title="ppsci/validate/new_validator.py"
    from ppsci.validate import base

    class NewValidator(base.Validator):
        def __init__(self, ...):
            ...
            # initialization
    ```

4. Import the written new validator class `NewValidator` in `ppsci/validate/__init__.py` and add it to `__all__`.

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

After completing the work of writing new validator code above, we can call the new validator class we wrote and use it to create a validator instance in the manner of `ppsci.validate.NewValidator`, just like PaddleScience built-in validators. Similarly, after the validator is built, it is recommended to wrap all validators in a dictionary for convenient subsequent indexing.

``` py title="examples/demo/demo.py"
new_validator = ppsci.validate.NewValidator(...)
validator = {..., new_validator.name: new_validator}
```

### 2.10 Build Visualizer [Optional]

PaddleScience has built-in some common visualizers, such as `VisualizerVtu` visualizer, etc. If you want to use these built-in visualizers, you can directly call the API under [`ppsci.visualizer.*`](./api/visualize.md) and fill in the parameters required for visualizer instantiation to quickly build the model.

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

If you need to add a new visualizer, the steps are similar to the addition method of other modules, and will not be repeated here.

### 2.11 Build Solver

[`Solver`](./api/solver.md) is the global management class responsible for calling training, evaluation, and visualization in PaddleScience. Before training starts, you need to pass the built model, constraint, optimizer and other instances to `Solver` for instantiation, and then call its built-in methods for training, evaluation, and visualization.

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

### 2.12 Write Configuration File [Important]

??? info "Long content, click to expand"

    After the development of the above steps, the main part of the case code has been completed.
    When we want to run some tuning experiments based on this code to get better results, or better manage the running parameter settings,
    we can use the configuration management system provided by PaddleScience to separate the experimental running parameters from the code and write them into the configuration file in `yaml` format, so as to better manage, record, and tune experiments.

    Taking `viv` case code as an example, at runtime we need to set equation parameters, STL file path, training rounds, `batch_size`, random seed, learning rate and other hyperparameters in appropriate positions, as shown below.

    ``` py
    ...
    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.VIV_DATA_PATH,
            "input_keys": ("t_f",),
            "label_keys": ("eta", "f"),
            "weight_dict": {"eta": 100},
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }
    ...
    ...
    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()
    ...
    ```

    These parameters may be manually adjusted as variables at any time during the experiment. How to avoid frequent modification of source code leading to confusion of experimental records and ensure complete and traceable records during the adjustment process is a major problem. Therefore, PaddleScience provides a configuration file management system based on hydra + omegaconf to solve this problem.

    It is very simple to modify the existing code to the way of configuration file control. You only need to write the necessary parameters into the `yaml` file, then read and parse the file through hydra when the program is running, and control the experiment running through its content. Taking the `viv` case as an example, it specifically includes the following steps.

    1. You need to create a `conf` folder under the directory where the code file `viv.py` is located, and create a `viv.yaml` file with the same name as `viv.py` under `conf`, as shown below.

        ``` sh hl_lines="3-4"
        PaddleScience/examples/fsi/
        ├── viv.py
        └── conf
            └── viv.yaml
        ```

    2. Fill in the necessary hyperparameters in `viv.py` into the configuration of each level of `viv.yaml` according to their semantics. For example, general parameters `mode`, `output_dir`, `seed`, equation parameters, file paths, etc., are directly filled in the first level; while model structure parameters, training rounds, etc., which are only related to model and training, only need to be filled in `MODEL` and `TRAIN` levels respectively (`EVAL` level is the same).
    3. Modify the existing `train` and `evaluate` functions to accept a parameter `cfg` (`cfg` is the content in the read `yaml` file and stored in the form of a dictionary), and uniformly change the internal hyperparameters to be obtained through `cfg.xxx` instead of the original direct setting as numbers or strings, as shown below.

        ``` py
        from omegaconf import DictConfig

        def train(cfg: DictConfig):
            # Training code...

        def evaluate(cfg: DictConfig):
            # Evaluation code...
        ```

    4. Create a new `main` function (which also accepts and only accepts one `cfg` parameter). It is responsible for calling the `train` or `evaluate` function according to `cfg.mode`, and add the decorator `@hydra.main(version_base=None, config_path="./conf", config_name="viv.yaml")` to the `main` function, as shown below.

        ``` py
        @hydra.main(version_base=None, config_path="./conf", config_name="viv.yaml")
        def main(cfg: DictConfig):
            if cfg.mode == "train":
                train(cfg)
            elif cfg.mode == "eval":
                evaluate(cfg)
            else:
                raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")

        ```

    5. Start `main()` in the main program startup entry `if __name__ == "__main__":`, as shown below.

        ``` py
        if __name__ == "__main__":
            main()
        ```

    After all transformations are completed, `viv.py` and `viv.yaml` are as shown below.

    === "examples/fsi/viv.py"

        ``` py linenums="1"
        --8<--
        examples/fsi/viv.py
        --8<--
        ```

    === "examples/fsi/conf/viv.yaml"

        ``` yaml linenums="1"
        defaults: # (1)
          - ppsci_default # (2)
          - TRAIN: train_default # (3)
          - TRAIN/ema: ema_default # (4)
          - TRAIN/swa: swa_default # (5)
          - EVAL: eval_default # (6)
          - INFER: infer_default # (7)
          - _self_ # (8)

        hydra:
          run:
            # dynamic output directory according to running time and override name
            dir: outputs_VIV/${now:%Y-%m-%d}/${now:%H-%M-%S}/${hydra.job.override_dirname} # (9)
          job:
            name: ${mode} # name of logfile
            chdir: false # keep current working directory unchanged
          callbacks:
            init_callback: # (10)
              _target_: ppsci.utils.callbacks.InitCallback # (11)
          sweep:
            # output directory for multirun
            dir: ${hydra.run.dir}
            subdir: ./

        # general settings
        mode: train # running mode: train/eval # (12)
        seed: 42 # (13)
        output_dir: ${hydra:run.dir} # (14)
        log_freq: 20 # (15)
        use_tbd: false # (16)

        VIV_DATA_PATH: "./VIV_Training_Neta100.mat" # (17)

        # model settings
        MODEL: # (18)
          input_keys: ["t_f"] # (19)
          output_keys: ["eta"] # (20)
          num_layers: 5 # (21)
          hidden_size: 50 # (22)
          activation: "tanh" # (23)

        # training settings
        TRAIN: # (24)
          epochs: 100000 # (25)
          iters_per_epoch: 1 # (26)
          save_freq: 10000 # (27)
          eval_during_train: true # (28)
          eval_freq: 1000 # (29)
          batch_size: 100 # (30)
          lr_scheduler: # (31)
            epochs: ${TRAIN.epochs} # (32)
            iters_per_epoch: ${TRAIN.iters_per_epoch} # (33)
            learning_rate: 0.001 # (34)
            step_size: 20000 # (35)
            gamma: 0.9 # (36)
          pretrained_model_path: null # (37)
          checkpoint_path: null # (38)

        # evaluation settings
        EVAL: # (39)
          pretrained_model_path: null # (40)
          batch_size: 32 # (41)

        # inference settings
        INFER: # (42)
          pretrained_model_path: "https://paddle-org.bj.bcebos.com/paddlescience/models/viv/viv_pretrained.pdparams" # (43)
          export_path: ./inference/viv # (44)
          pdmodel_path: ${INFER.export_path}.json # (45)
          pdiparams_path: ${INFER.export_path}.pdiparams # (46)
          input_keys: ${MODEL.input_keys} # (47)
          output_keys: ["eta", "f"] # (48)
          device: gpu # (49)
          engine: native # (50)
          precision: fp32 # (51)
          onnx_path: ${INFER.export_path}.onnx # (52)
          ir_optim: true # (53)
          min_subgraph_size: 10 # (54)
          gpu_mem: 4000 # (55)
          gpu_id: 0 # (56)
          max_batch_size: 64 # (57)
          num_cpu_threads: 4 # (58)
          batch_size: 16 # (59)
        ```

        1. `defaults:` - Beginning of defining a series of default configuration items.
        2. `- ppsci_default` - Use default configuration named `ppsci_default`.
        3. `- TRAIN: train_default` - Apply `train_default` default configuration under `TRAIN` namespace.
        4. `- TRAIN/ema: ema_default` - Apply `ema_default` configuration under `ema` sub-namespace of `TRAIN`.
        5. `- TRAIN/swa: swa_default` - Apply `swa_default` configuration under `swa` sub-namespace of `TRAIN`.
        6. `- EVAL: eval_default` - Apply `eval_default` default configuration under `EVAL` namespace.
        7. `- INFER: infer_default` - Apply `infer_default` default configuration under `INFER` namespace.
        8. `- _self_` - Indicates that the configuration of the current file itself will be included, used to overwrite or add default settings.
        9. `dir: outputs_VIV/...` - Set dynamic output directory based on current time and override name.
        10. `init_callback:` - Define settings for initialization callback.
        11. `_target_: ppsci.utils.callbacks.InitCallback` - Specify the concrete implementation class or function of the initialization callback.
        12. `mode: train` - Set current running mode to training mode.
        13. `seed: 42` - Set global random seed to 42 to ensure experimental reproducibility.
        14. `output_dir: ${hydra:run.dir}` - Set output directory, same as Hydra configured running directory.
        15. `log_freq: 20` - Set log recording frequency, record every 20 times.
        16. `use_tbd: false` - Turn off tensorboard function.
        17. `VIV_DATA_PATH: "./VIV_Training_Neta100.mat"` - Specify VIV data file path (can be other dataset or hyperparameter values).
        18. `MODEL:` - Start defining model related settings.
        19. `input_keys: ["t_f"]` - Set model input keys to `t_f`.
        20. `output_keys: ["eta"]` - Set model output keys to `eta`.
        21. `num_layers: 5` - Set model layers to 5.
        22. `hidden_size: 50` - Set model hidden layer size to 50.
        23. `activation: "tanh"` - Use hyperbolic tangent function as activation function.
        24. `TRAIN:` - Start defining training related settings.
        25. `epochs: 100000` - Set total training rounds to 100000.
        26. `iters_per_epoch: 1` - Each round contains 1 iteration.
        27. `save_freq: 10000` - Save model or checkpoint every 10000 iterations.
        28. `eval_during_train: true` - Enable evaluation during training.
        29. `eval_freq: 1000` - Perform evaluation every 1000 iterations.
        30. `batch_size: 100` - Set training batch size to 100.
        31. `lr_scheduler:` - Start defining learning rate scheduler settings.
        32. `epochs: ${TRAIN.epochs}` - Rounds used by learning rate scheduler are the same as training rounds.
        33. `iters_per_epoch: ${TRAIN.iters_per_epoch}` - Iterations per round used by learning rate scheduler are the same as training.
        34. `learning_rate: 0.001` - Set initial learning rate to 0.001.
        35. `step_size: 20000` - Adjust learning rate every 20000 steps.
        36. `gamma: 0.9` - Learning rate adjustment ratio is 0.9.
        37. `pretrained_model_path: null` - Initial pre-trained model path, can be actual path or url, default is null meaning no pre-trained model is used.
        38. `checkpoint_path: null` - Checkpoint path, used to specify the checkpoint loaded by the model, default is null meaning no checkpoint is loaded.
        39. `EVAL:` - Start defining evaluation related settings.
        40. `pretrained_model_path: null` - Pre-trained model path used during evaluation, default is null meaning no pre-trained model is used.
        41. `batch_size: 32` - Batch size during evaluation.
        42. `INFER:` - Start defining inference related settings.
        43. `pretrained_model_path: "https://..."` - Specify pre-trained model path, can be actual path or url, used for inference.
        44. `export_path: ./inference/viv` - Set model export path, i.e., the saving location of model files required for inference.
        45. `pdmodel_path: ${INFER.export_path}.json` - Specify Paddle model structure file path, suffix can be `.json` or `.pdmodel`.
        46. `pdiparams_path: ${INFER.export_path}.pdiparams` - Specify Paddle model parameter file path.
        47. `input_keys: ${MODEL.input_keys}` - Input keys used during inference are the same as when model is defined.
        48. `output_keys: ["eta", "f"]` - Set output keys during inference, including "eta" and "f".
        49. `device: gpu` - Specify device used for inference as GPU.
        50. `engine: native` - Set inference engine to native (may refer to default engine using specific library or framework).
        51. `precision: fp32` - Set inference precision to 32-bit floating point number.
        52. `onnx_path: ${INFER.export_path}.onnx` - If supported, specify ONNX model file path.
        53. `ir_optim: true` - Enable Intermediate Representation optimization.
        54. `min_subgraph_size: 10` - Set minimum size for subgraph optimization to improve inference performance.
        55. `gpu_mem: 4000` - Specify amount of memory (unit may be MB) available for GPU during inference.
        56. `gpu_id: 0` - Specify which GPU to use for inference, here is the first GPU.
        57. `max_batch_size: 64` - Set maximum batch size supported during inference.
        58. `num_cpu_threads: 4` - Specify number of CPU threads used for inference.
        59. `batch_size: 16` - Set actual batch size during inference to 16.

### 2.13 Training

The training of the PaddleScience model only requires calling one line of code.

``` py title="examples/demo/demo.py"
solver.train()
```

### 2.14 Evaluation

The evaluation of the PaddleScience model only requires calling one line of code.

``` py title="examples/demo/demo.py"
solver.eval()
```

### 2.15 Visualization [Optional]

If the `visualizer` parameter is passed when instantiating `Solver`, the visualization of the PaddleScience model only requires calling one line of code.

``` py title="examples/demo/demo.py"
solver.visualize()
```

!!! tip "Visualization Solution"

    For some complex cases, the cost of writing `Visualizer` is not low, and not any data type can be easily visualized. Therefore, after training is completed, you can manually construct a data dictionary for prediction, then use `solver.predict` to get model prediction results, and finally use third-party libraries such as `matplotlib` to visualize and save the prediction results.

## 3. Write Documentation

In addition to case code, PaddleScience also stores detailed documentation for corresponding cases, written and rendered using Markdown + [Mkdocs](https://www.mkdocs.org/) + [Mkdocs-Material](https://squidfunk.github.io/mkdocs-material/). The steps for writing documentation are as follows.

### 3.1 Install Necessary Dependent Packages

Instant rendering is required during the documentation writing process to preview the document content to check for errors. Therefore, you need to install mkdocs related dependent packages according to the following command.

``` sh
python -m pip install -r docs/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 Write Document Content

PaddleScience documentation is written based on plugins such as [Mkdocs-Material](https://squidfunk.github.io/mkdocs-material/), [PyMdown](https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/). Based on Markdown syntax, it supports a variety of extensible functions, which can greatly improve the aesthetics and reading experience of the document. It is recommended to refer to the document content in the hyperlink and select appropriate functions to assist in document writing.

### 3.3 Use markdownlint to Format Document [Optional]

If your development environment is VSCode, it is recommended to install the [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) extension. After installation, inside the written document: right click --> Format Document.

### 3.4 Preview Document

Assuming the location of the written document is `PaddleScience/docs/zh/examples/your_exmaple.md`, in order to display it in the left directory of PaddleScience official website [Classic Cases](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/examples/allen_cahn/), you need to modify `PaddleScience/mkdocs.yml`. Add the relative path of `your_exmaple.md` to the list under `- Classic Cases:` following other cases, as shown in the highlighted line below.

``` yaml hl_lines="23" title="PaddleScience/mkdocs.yml"
...
--8<--
mkdocs.yml:38:58
--8<--
        - EXAMPLE_NAME: docs/zh/examples/your_exmaple.md
...
```

Then execute the following command in the `PaddleScience/` directory. After waiting for the build to complete, click the displayed link to enter the local webpage to preview the document content.

``` sh
mkdocs serve
```

``` log
# ====== Terminal printing information is as follows ======
# INFO     -  Building documentation...
# INFO     -  Cleaning site directory
# INFO     -  Documentation built in 20.95 seconds
# INFO     -  [07:39:35] Watching paths for changes: 'docs', 'mkdocs.yml'
# INFO     -  [07:39:35] Serving on http://127.0.0.1:8000/PaddlePaddle/PaddleScience/
# INFO     -  [07:39:41] Browser connected: http://127.0.0.1:58903/PaddlePaddle/PaddleScience/
# INFO     -  [07:40:41] Browser connected: http://127.0.0.1:58903/PaddlePaddle/PaddleScience/zh/development/
```

!!! tip "Manually Specify Service Address and Port Number"

    If the default port number 8000 is occupied, you can manually specify the address and port for service deployment. Examples are as follows.

    ``` sh
    # Specify 127.0.0.1 as address and 8687 as port number
    mkdocs serve -a 127.0.0.1:8687
    ```

## 4. Organize Code and Submit

### 4.1 Organize Code

After completing example writing and training, and confirming that the results are correct, you can organize the code.
Use git commands to submit all new and modified code files as well as necessary documents, pictures, etc. to the local `dev_model` branch together.

### 4.2 Sync Upstream Code

During the development process, the upstream code may be updated, so you need to execute the following command to pull the latest upstream code, merge it into the current code, and synchronize with the latest upstream code.

``` sh
git remote add upstream https://github.com/PaddlePaddle/PaddleScience.git
git fetch upstream develop:upstream_develop
git merge upstream_develop
```

If a conflict occurs, you need to resolve the conflict, then use `git add` and `git commit -m "merge code of upstream"` commands to commit the code to the local repository, and finally execute `git push origin dev_model` to push the code to your remote repository.

### 4.3 Submit Pull Request

Switch to the `dev_model` branch on the github webpage, click "Contribute", then click the "Open pull request" button,
and contribute the `dev_model` branch containing your code, documents, pictures and other content to PaddleScience as a merge request.
