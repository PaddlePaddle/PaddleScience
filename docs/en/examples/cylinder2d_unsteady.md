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

The problem of flow around a cylinder can be applied to many fields. For example, in industrial design, it can be used to simulate and optimize fluid flow in various equipment, such as wind turbines, hydrodynamic performance of cars and aircraft, etc. In the field of environmental protection, the problem of flow around a cylinder also has applications, such as predicting and controlling river floods, studying the diffusion of pollutants, etc. In addition, in engineering practice, such as fluid dynamics, hydrostatics, heat exchange, aerodynamics and other fields, the problem of flow around a cylinder also has practical significance.

2D Flow Around a Cylinder refers to the flow pattern of low-speed steady flow around a two-dimensional cylinder, which is only related to the $Re$ number. When $Re \le 1$, the inertial force in the flow field occupies a secondary position compared with the viscous force, the streamlines upstream and downstream of the cylinder are symmetrical, and the drag coefficient is approximately inversely proportional to $Re$ (drag coefficient is 10~60). The flow around this $Re$ number range is called the Stokes region; as $Re$ increases, the streamlines upstream and downstream of the cylinder gradually lose symmetry.

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

Dimensionless coordinate $x：X = \dfrac{x}{x^*}$; Dimensionless coordinate $y：Y = \dfrac{y}{y^*}$

Dimensionless velocity $x：U = \dfrac{u}{u^*}$; Dimensionless velocity $y：V = \dfrac{v}{u^*}$

Dimensionless pressure $P = \dfrac{p}{p^*}$

Reynolds number $Re = \dfrac{L U_0}{\nu}$

The following dimensionless Navier-Stokes equations can be obtained and applied to the interior of the fluid domain:

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

In the 2D-Cylinder problem, each known coordinate point $(t, x, y)$ has three unknown quantities to be solved: lateral velocity $u$, longitudinal velocity $v$, and pressure $p$. Here we use a relatively simple MLP (Multilayer Perceptron) to represent the mapping function $f: \mathbb{R}^3 \to \mathbb{R}^3$ from $(t, x, y)$ to $(u, v, p)$, i.e.:

$$
u, v, p = f(t, x, y)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows

``` py linenums="33"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:33:34
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, we specify the input variable name of the network model as `["t", "x", "y"]` and the output variable name as `["u", "v", "p"]`, these names are consistent with the subsequent code.

Then by specifying the number of layers, number of neurons and activation function of MLP, we instantiated a neural network model `model` with 9 hidden layers, 50 neurons per layer, using "tanh" as the activation function.

### 3.2 Equation Construction

Since 2D-Cylinder uses the 2D transient form of the Navier-Stokes equation, `NavierStokes` built in PaddleScience can be used directly.

``` py linenums="36"
--8<--
examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py:36:39
--8<--
```

When instantiating the `NavierStokes` class, necessary parameters need to be specified: dynamic viscosity $\nu=0.02$, fluid density $\rho=1.0$.

### 3.3 Computational Domain Construction

The computational domain of 2D-Cylinder in this article is composed of point clouds stored in CSV files, so the point cloud geometry `PointCloud` and time domain `TimeDomain` built in PaddleScience can be used directly to combine into a time-space `TimeXGeometry` computational domain.

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

According to the dimensionless formulas and boundary conditions obtained in [2. Problem Definition](#2), corresponding to the three constraint conditions guiding model training in the computational domain, namely:

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

The first parameter of `InteriorConstraint` is the equation expression, used to describe how to calculate the constraint target. Here, fill in `equation["NavierStokes"].equations` instantiated in the [3.2 Equation Construction](#32) chapter;

The second parameter is the target value of the constraint variable. In this problem, we hope that the three intermediate results `continuity`, `momentum_x`, `momentum_y` generated by the Navier-Stokes equation are optimized to 0, so all their target values are set to 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["time_rect"]` instantiated in the [3.3 Computational Domain Construction](#33) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here we use full data points for training, so the `dataset` field is set to "IterableNamedArrayDataset" and `iters_per_epoch` is also set to 1, and the sampling point number `batch_size` is set to 9420 * 30 (indicating 9420 data points generated at one moment, a total of 30 moments);

The fifth parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"mean"`, that is, we will sum and average the loss terms generated by all data points involved in the calculation;

The sixth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "EQ".

#### 3.4.2 Boundary Constraint

Similarly, we also need to construct Dirichlet boundary constraints for the inflow boundary, outflow boundary, and circumference boundary of the fluid domain. Taking `bc_inlet_cylinder` boundary constraint as an example, since the action area is the boundary and the data on the boundary is recorded by CSV file, we use the `SupervisedConstraint` class and specify the first parameter `dataloader_cfg` configuration dictionary according to the following rules:

- The first parameter of the configuration dictionary is the configuration dictionary including the path of the CSV file `./datasets/domain_inlet_cylinder.csv`;

- The first parameter of the configuration dictionary specifies the data loading method. Here we use `IterableCSVDataset` as the full data loader;

- The second parameter of the configuration dictionary specifies the data loading path. Here fill in `./datasets/domain_inlet_cylinder.csv`;

- The third parameter of the configuration dictionary specifies the input columns to be read from the file, corresponding to the transformed keyword. Here fill in `("x", "y")`;

- The fourth parameter of the configuration dictionary specifies the label columns to be read from the file, corresponding to the transformed keyword. Here fill in `("u", "v")`;

- Considering that the same variable may have different field names in different CSV files, and some field names are too long and easy to write wrong when writing code, the fifth parameter of the configuration dictionary is used to specify the alias of the field column. Here fill in `{"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"}`;

- The sixth parameter of the configuration dictionary specifies the weight of each label when calculating the loss. Here we amplify the weights of "u" and "v" to 10, fill in `{"u": 10, "v": 10}`;

- The seventh parameter of the configuration dictionary specifies whether data reading involves time information. Here we set it to the training timestamp, that is, fill in `train_timestamps`;

The second parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"mean"`, that is, we will sum and average the loss terms generated by all data points involved in the calculation;

The third parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "BC_inlet_cylinder".

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

The equation setting is the same as the setting of [Constraint Construction](#32), indicating how to calculate the target variables to be evaluated;

Here we set the label value to 0 for the three target variables `momentum_x`, `continuity`, `momentum_y`;

The computational domain is the same as the setting of [Constraint Construction](#32), indicating evaluation on the specified computational domain;

The sampling point configuration needs to specify the total number of evaluation points `total_size`. Here we set it to 9662 \* 50 (9420 points in the fluid domain + 161 fluid domain inflow boundary points + 81 fluid domain outflow boundary points, a total of 50 evaluation moments);

For evaluation metric `metric`, select `ppsci.metric.MSE`;

Other configurations are similar to the settings of [Constraint Construction](#32).

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

The prediction results are shown below. The horizontal axis of the image is the horizontal direction, and the vertical axis represents the vertical direction. The fluid flow direction is from left to right. The picture shows the result of the lateral flow velocity $u(t,x,y)$ of the corresponding flow field predicted by the model at 50 moments.

???+ info "Note"

    This case is only shown as a demo and has not been fully tuned. Some of the results shown below may differ from OpenFOAM.

<figure markdown>
  ![u_pred.gif](https://paddle-org.bj.bcebos.com/paddlescience/docs/Cylinder2D_unsteady/cylinder_2d_unsteady_Re100.gif){ loading=lazy }
  <figcaption>Model prediction result of lateral flow velocity u</figcaption>
</figure>
