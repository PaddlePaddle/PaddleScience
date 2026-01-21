# 2D-Cylinder (2D Flow Around a Cylinder)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6160381?contributionType=1&sUid=438690&shared=1&ts=1683961158552" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar -o cylinder2d_unsteady_Re100_dataset.tar
    # unzip it
    tar -xvf cylinder2d_unsteady_Re100_dataset.tar
    python cylinder2d_unsteady_Re100.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar -o cylinder2d_unsteady_Re100_dataset.tar
    # unzip it
    tar -xvf cylinder2d_unsteady_Re100_dataset.tar
    python cylinder2d_unsteady_Re100.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python cylinder2d_unsteady_Re100.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar -o cylinder2d_unsteady_Re100_dataset.tar
    # unzip it
    tar -xvf cylinder2d_unsteady_Re100_dataset.tar
    python cylinder2d_unsteady_Re100.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [cylinder2d_unsteady_Re100_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_pretrained.pdparams) | loss(Residual): 0.00398<br>MSE.continuity(Residual): 0.00126<br>MSE.momentum_x(Residual): 0.00151<br>MSE.momentum_y(Residual): 0.00120 |

## 1. Background Introduction

The flow around a cylinder is a fundamental problem with applications in diverse fields. In industrial design, it is used to simulate and optimize fluid dynamics in equipment such as wind turbines, as well as the aerodynamic performance of automobiles and aircraft. In environmental engineering, it aids in predicting river flooding and modeling pollutant dispersion. Furthermore, it holds significant practical value in broader engineering contexts, including fluid dynamics, heat transfer, and aerodynamics.

2D flow around a cylinder describes the flow pattern of low-speed steady flow past a two-dimensional cylinder, governed primarily by the Reynolds number ($Re$). When $Re \le 1$ (the Stokes region), inertial forces are negligible compared to viscous forces; streamlines are symmetrical upstream and downstream, and the drag coefficient is approximately inversely proportional to $Re$ (ranging from 10 to 60). As $Re$ increases, this symmetry is gradually lost.

## 2. Problem Definition

Mass conservation:

$$
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
$$

$x$ momentum conservation:

$$
\dfrac{\partial u}{\partial t} + u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} + \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2})
$$

$y$ momentum conservation:

$$
\dfrac{\partial v}{\partial t} + u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} + \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2})
$$

**Let:**

$t^* = \dfrac{L}{U_0}$

$x^*=y^* = L$

$u^*=v^* = U_0$

$p^* = \rho {U_0}^2$

**Define:**

Dimensionless time $\tau = \dfrac{t}{t^*}$

Dimensionless coordinate $x: X = \dfrac{x}{x^*}$; Dimensionless coordinate $y: Y = \dfrac{y}{y^*}$

Dimensionless velocity $x: U = \dfrac{u}{u^*}$; Dimensionless velocity $y: V = \dfrac{v}{u^*}$

Dimensionless pressure $P = \dfrac{p}{p^*}$

Reynolds number $Re = \dfrac{L U_0}{\nu}$

The following dimensionless Navier-Stokes equations apply to the interior of the fluid domain:

Mass conservation:

$$
\dfrac{\partial U}{\partial X} + \dfrac{\partial U}{\partial Y} = 0
$$

$x$ momentum conservation:

$$
\dfrac{\partial U}{\partial \tau} + U\dfrac{\partial U}{\partial X} + V\dfrac{\partial U}{\partial Y} = -\dfrac{\partial P}{\partial X} + \dfrac{1}{Re}(\dfrac{\partial ^2 U}{\partial X^2} + \dfrac{\partial ^2 U}{\partial Y^2})
$$

$y$ momentum conservation:

$$
\dfrac{\partial V}{\partial \tau} + U\dfrac{\partial V}{\partial X} + V\dfrac{\partial V}{\partial Y} = -\dfrac{\partial P}{\partial Y} + \dfrac{1}{Re}(\dfrac{\partial ^2 V}{\partial X^2} + \dfrac{\partial ^2 V}{\partial Y^2})
$$

For the fluid domain boundary and the inner circumference boundary of the fluid domain, Dirichlet boundary conditions need to be applied:

Fluid domain inlet boundary:

$$
u=1, v=0
$$

Circumference boundary:

$$
u=0, v=0
$$

Fluid domain outlet boundary:

$$
p=0
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

Before starting to build the code, please download the dataset required for training and evaluation according to the following command

``` sh
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar
tar -xf cylinder2d_unsteady_Re100_dataset.tar
```

### 3.1 Model Construction

In the 2D-Cylinder problem, for each spatiotemporal coordinate $(t, x, y)$, there are three unknown quantities to solve: lateral velocity $u$, longitudinal velocity $v$, and pressure $p$. We employ a Multilayer Perceptron (MLP) to approximate the mapping function $f: \mathbb{R}^3 \to \mathbb{R}^3$ from $(t, x, y)$ to $(u, v, p)$, such that:

$$
u, v, p = f(t, x, y)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows

``` py linenums="33"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:33:34
--8<--
```

To ensure accurate and efficient variable access during computation, we define the model's input keys as `["t", "x", "y"]` and output keys as `["u", "v", "p"]`, maintaining consistency with the code.

We instantiate the MLP model with 9 hidden layers, 50 neurons per layer, and the `tanh` activation function.

### 3.2 Equation Construction

Since this problem involves the 2D transient Navier-Stokes equations, we can directly use the built-in `NavierStokes` class in PaddleScience.

``` py linenums="36"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:36:39
--8<--
```

When instantiating the `NavierStokes` class, necessary parameters need to be specified: dynamic viscosity $\nu=0.02$, fluid density $\rho=1.0$.

### 3.3 Computational Domain Construction

The computational domain is defined by point clouds stored in CSV files. We combine the built-in `PointCloud` geometry and `TimeDomain` to construct a spatiotemporal `TimeXGeometry` domain.

``` py linenums="41"
# set timestamps
train_timestamps = np.linspace(
    cfg.TIME_START, cfg.TIME_END, cfg.NUM_TIMESTAMPS, endpoint=True
).astype("float32")
train_timestamps = np.random.choice(train_timestamps, cfg.TRAIN_NUM_TIMESTAMPS)
train_timestamps.sort()
t0 = np.array([cfg.TIME_START], dtype="float32")

val_timestamps = np.linspace(
    cfg.TIME_START, cfg.TIME_END, cfg.NUM_TIMESTAMPS, endpoint=True
).astype("float32")

logger.message(f"train_timestamps: {train_timestamps.tolist()}")
logger.message(f"val_timestamps: {val_timestamps.tolist()}")

# set time-geometry
geom = {
    "time_rect": ppsci.geometry.TimeXGeometry(
        ppsci.geometry.TimeDomain(
            cfg.TIME_START,
            cfg.TIME_END,
            timestamps=np.concatenate((t0, train_timestamps), axis=0),
        ),
        ppsci.geometry.PointCloud(
            reader.load_csv_file(
                "./datasets/domain_train.csv",
                ("x", "y"),
                alias_dict={"x": "Points:0", "y": "Points:1"},
            ),
            ("x", "y"),
        ),
    ),
    "time_rect_eval": ppsci.geometry.PointCloud(
        reader.load_csv_file(
            "./datasets/domain_eval.csv",
            ("t", "x", "y"),
        ),
        ("t", "x", "y"),
    ),
}
```

1. The evaluation data points already contain timestamp information, so there is no need to combine with `TimeDomain` into `TimeXGeometry`, just use `PointCloud` to read in the data.

???+ tip "Tip"

    `PointCloud` and `TimeDomain` are two `Geometry` derived classes that can be used independently.

    If the input data only comes from point cloud geometry, you can directly use `ppsci.geometry.PointCloud(...)` to create a spatial geometric domain object;

    If the input data only comes from a one-dimensional time domain, you can directly use `ppsci.geometry.TimeDomain(...)` to construct a time domain object.

### 3.4 Constraint Construction

According to the dimensionless formulas and boundary conditions obtained in [2. Problem Definition](#2-problem-definition), corresponding to the three constraint conditions guiding model training in the computational domain, namely:

1. Dimensionless Navier-Stokes equation constraint applied to internal points of the fluid domain (after simple term shifting)

    $$
    \dfrac{\partial U}{\partial X} + \dfrac{\partial U}{\partial Y} = 0
    $$

    $$
    \dfrac{\partial U}{\partial \tau} + U\dfrac{\partial U}{\partial X} + V\dfrac{\partial U}{\partial Y} + \dfrac{\partial P}{\partial X} - \dfrac{1}{Re}(\dfrac{\partial ^2 U}{\partial X^2} + \dfrac{\partial ^2 U}{\partial Y^2}) = 0
    $$

    $$
    \dfrac{\partial V}{\partial \tau} + U\dfrac{\partial V}{\partial X} + V\dfrac{\partial V}{\partial Y} + \dfrac{\partial P}{\partial Y} - \dfrac{1}{Re}(\dfrac{\partial ^2 V}{\partial X^2} + \dfrac{\partial ^2 V}{\partial Y^2}) = 0
    $$

    In order to facilitate obtaining intermediate variables, the `NavierStokes` class internally names the results on the left side of the above formula as `continuity`, `momentum_x`, `momentum_y` respectively.

2. Dirichlet boundary condition constraints applied to the fluid domain inlet, internal circumference, and fluid domain outlet

    Fluid domain inlet boundary:

    $$
    u=1, v=0
    $$

    Fluid domain outlet boundary:

    $$
    p=0
    $$

    Circumference boundary:

    $$
    u=0, v=0
    $$

3. Initial value condition constraint applied to internal points of the fluid domain at the initial moment:

    $$
    u=u_{t0}, v=v_{t0}, p=p_{t0}
    $$

Next, use `InteriorConstraint` and `SupervisedConstraint` built in PaddleScience to construct the above two constraints.

Before defining constraints, you need to specify the number of sampling points for each constraint, indicating the number of sampled data for each constraint in its corresponding computational domain, as well as general sampling configuration.

``` py linenums="82"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:82:84
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on internal points of the fluid domain as an example, the code is as follows:

``` py linenums="86"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:86:98
--8<--
```

- **Equation Expression**: Specifies how to calculate the constraint target. We use `equation["NavierStokes"].equations` from Section 3.2.
- **Target Values**: The target values for the constraint variables. We aim to minimize the residuals of `continuity`, `momentum_x`, and `momentum_y` to 0.
- **Computational Domain**: The domain where the constraint applies. We use `geom["time_rect"]` from Section 3.3.
- **Sampling Configuration**: Specifies how data is sampled. We use full data points (`IterableNamedArrayDataset`, `iters_per_epoch=1`) with a `batch_size` of 9420 * 30 (9420 spatial points × 30 timestamps).
- **Loss Function**: We use the Mean Squared Error (MSE) with `reduction="mean"` to average the loss across all points.
- **Name**: A unique name for the constraint, set here as "EQ".

#### 3.4.2 Boundary Constraint

We also construct Dirichlet boundary constraints for the inflow, outflow, and cylinder circumference boundaries. Taking `bc_inlet_cylinder` as an example, we use `SupervisedConstraint` since the boundary data is provided in a CSV file. The `dataloader_cfg` is configured as follows:

- **Dataset Class**: `IterableCSVDataset` for loading full data.
- **File Path**: `./datasets/domain_inlet_cylinder.csv`.
- **Input Keys**: `("x", "y")` to be read from the file.
- **Label Keys**: `("u", "v")` to be read from the file.
- **Alias Dict**: Maps file column names to standard keys, e.g., `{"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"}`.
- **Weight Dict**: Assigns weights to labels. We amplify "u" and "v" weights to 10: `{"u": 10, "v": 10}`.
- **Timestamps**: Specifies the time information, set here to `train_timestamps`.

We use the MSE loss function with `reduction="mean"` and name the constraint "BC_inlet_cylinder".

The remaining `bc_outlet` is constructed according to the same principle, the code is as follows:

``` py linenums="99"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:99:127
--8<--
```

#### 3.4.3 Initial Value Constraint

For points in the fluid domain at time $t=t_0$, we also need to apply initial value constraints to $u$, $v$, $p$. The code is as follows:

``` py linenums="128"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:128:142
--8<--
```

#### 3.4.4 Supervised Constraint

In this case, a certain number of supervision points are added inside the fluid domain to ensure the final convergence of the model, so a supervised constraint needs to be added finally. The data also comes from CSV files. The code is as follows:

``` py linenums="143"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:143:157
--8<--
```

After the differential equation constraint, boundary constraint, initial value constraint, and supervised constraint are constructed, encapsulate them into a dictionary with the names we just named as keys for subsequent access.

``` py linenums="159"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:159:166
--8<--
```

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, we use 40,000 training epochs, evaluation interval is 400 epochs, and learning rate is set to 0.001.

``` yaml linenums="60"
--8<--
examples/cylinder/2d_unsteady/conf/cylinder2d_unsteady.yaml:60:65
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="168"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:168:169
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="171"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:171:189
--8<--
```

The equation setting is the same as the setting of [Constraint Construction](#34-constraint-construction), indicating how to calculate the target variables to be evaluated;

Here we set the label value to 0 for the three target variables `momentum_x`, `continuity`, `momentum_y`;

The computational domain is the same as the setting of [Constraint Construction](#34-constraint-construction), indicating evaluation on the specified computational domain;

The sampling point configuration needs to specify the total number of evaluation points `total_size`. Here we set it to 9662 \* 50 (9420 points in the fluid domain + 161 fluid domain inflow boundary points + 81 fluid domain outflow boundary points, a total of 50 evaluation moments);

For evaluation metric `metric`, select `ppsci.metric.MSE`;

Other configurations are similar to the settings of [Constraint Construction](#34-constraint-construction).

### 3.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, we can choose a suitable visualizer to visualize the output result.

The output data in this article is a two-dimensional point set in an area. The coordinates of each moment $t$ are $(x^t_i, y^t_i)$, and the corresponding value is $(u^t_i, v^t_i, p^t_i)$. Therefore, we only need to save the evaluated output data as 50 **vtu format** files according to time, and finally open them with visualization software to view. The code is as follows:

``` py linenums="191"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:191:204
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="206"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:206:228
--8<--
```

## 4. Complete Code

``` py linenums="1" title="cylinder2d_unsteady_Re100.py"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py
--8<--
```

## 5. Result Display

The prediction results are displayed below. The horizontal axis represents the x-direction, and the vertical axis represents the y-direction, with fluid flowing from left to right. The figure illustrates the predicted lateral flow velocity $u(t,x,y)$ at 50 time steps.

???+ info "Note"

    This case is only shown as a demo and has not been fully tuned. Some of the results shown below may differ from OpenFOAM.

<figure markdown>
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/Cylinder2D_unsteady/cylinder_2d_unsteady_Re100.gif){ loading=lazy }
  <figcaption>Model prediction result of lateral flow velocity u</figcaption>
</figure>
