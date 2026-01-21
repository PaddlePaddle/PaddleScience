# LabelFree-DNN-Surrogate (Aneurysm flow & Pipe flow)

=== "Model Training Command"

    Case 1: Pipe Flow

    ``` sh
    python poiseuille_flow.py
    ```

    Case 2: Aneurysm Flow

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/LabelFree-DNN-Surrogate/LabelFree-DNN-Surrogate_data.zip
    unzip LabelFree-DNN-Surrogate_data.zip

    python aneurysm_flow.py
    ```

=== "Model Evaluation Command"

    Case 1: Pipe Flow

    ``` sh
    python poiseuille_flow.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LabelFree-DNN-Surrogate/poiseuille_flow_pretrained.pdparams
    ```

    Case 2: Aneurysm Flow

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/LabelFree-DNN-Surrogate/LabelFree-DNN-Surrogate_data.zip
    unzip LabelFree-DNN-Surrogate_data.zip

    python aneurysm_flow.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LabelFree-DNN-Surrogate/aneurysm_flow.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python poiseuille_flow.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python poiseuille_flow.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
|[aneurysm_flow.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/LabelFree-DNN-Surrogate/aneurysm_flow.pdparams)| L-2 error u : 2.548e-4 <br> L-2 error v : 7.169e-5 |

## 1. Background Introduction

Numerical simulation of fluid dynamics problems mainly relies on using polynomials to discretize governing equations in space and/or time into finite-dimensional algebraic systems. Due to the multi-scale nature of physics and sensitivity to meshing of complex geometries, such a process is prohibitively expensive for most real-time applications (e.g., clinical diagnosis and surgical planning) and multi-query analysis (e.g., optimization design and uncertainty quantification). In this paper, we provide a physics-constrained DL method for surrogate modeling of fluid flows without relying on any simulation data. Specifically, a structured deep neural network (DNN) architecture is designed to enforce initial and boundary conditions, and governing partial differential equations (i.e., Navier-Stokes equations) are incorporated into the DNN loss to drive training. Numerical experiments are conducted on a number of internal flows relevant to hemodynamic applications, and forward propagation of uncertainties in fluid properties and domain geometry is studied. Results show that flow fields and forward propagated uncertainties agree well between DL surrogate approximations and first-principle numerical simulations.

## 2. Case 1: PipeFlow

### 2.1 Problem Definition

Pipe flow is a very common and commonly used fluid system, such as blood in arteries or airflow in the trachea. Generally, pipe flow is driven by pressure differences at both ends of the pipe, or driven by gravitational body force.
In the cardiovascular system, the former is more dominant because blood flow is mainly controlled by pressure drops caused by heart pumping. In general, simulating fluid dynamics in a tube requires numerically solving the full Navier-Stokes equations, but if the tube is straight and has a constant circular cross-section, an analytical solution for fully developed steady flow can be obtained, i.e., an ideal benchmark to verify the performance of the proposed method. Therefore, we first study flow in a two-dimensional circular tube (also known as Poiseuille flow).

Mass conservation:

$$
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
$$

$x$ momentum conservation:

$$
u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} + \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2})
$$

$y$ momentum conservation:

$$
u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} + \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2})
$$

We only focus on this fully developed flow and impose no-slip boundary conditions at the boundaries. Different from traditional PINNs methods, we force no-slip boundary conditions on boundaries through velocity function assumptions:
For fluid domain boundaries and internal circular boundaries of the fluid domain, Dirichlet boundary conditions need to be imposed:

<figure markdown>
  ![pipe](https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/pipe.png){ loading=lazy }
  <figcaption>Flow field diagram</figcaption>
</figure>

Fluid domain inlet boundary:

$$
p=0.1
$$

Fluid domain outlet boundary:

$$
p=0
$$

Fluid domain upper and lower boundaries:

$$
u=0, v=0
$$

### 2.2 Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

#### 2.2.1 Model Construction

In this case, each known coordinate point and the dynamic viscosity coefficient triplet $(x, y, \nu)$ of that point has its own transverse velocity $u$, longitudinal velocity $v$, and pressure $p$
Three unknown quantities to be solved. Here we use three relatively simple MLPs (Multilayer Perceptrons) to represent the mapping functions $f_1, f_2, f_3: \mathbb{R}^3 \to \mathbb{R}^3$ from $(x, y, \nu)$ to $(u, v, p)$, i.e.:

$$
u= transform_{output}(f_1(transform_{input}(x, y, \nu)))
$$

$$
v= transform_{output}(f_2(transform_{input}(x, y, \nu)))
$$

$$
p= transform_{output}(f_3(transform_{input}(x, y, \nu)))
$$

In the above formula, $f_1, f_2, f_3$ are the MLP models themselves, $transform_{input}, transform_{output}$, represent imposing additional structured custom layers for imposing constraints and enriching inputs, expressed in PaddleScience code as follows:

``` py linenums="71"
--8<--
examples/pipe/poiseuille_flow.py:71:105
--8<--
```

In order to access the values of specific variables accurately and quickly during calculation, we specify here that the input variable names of the network model are `["x", "y", "nu"]`, and the output variable names are `["u", "v", "p"]`. These names are consistent with subsequent code.

Then by specifying the number of layers, number of neurons, and activation function of the MLP, we instantiated three neural network models `model_u` `model_v` `model_p` with 3 layers of hidden neurons and 1 layer of output layer neurons, each layer having 50 neurons, using "swish" as the activation function.

#### 2.2.2 Equation Construction

Since this case uses the 2D steady-state form of the Navier-Stokes equation, `NavierStokes` built into PaddleScience can be used directly.

``` py linenums="110"
--8<--
examples/pipe/poiseuille_flow.py:110:115
--8<--
```

When instantiating the `NavierStokes` class, necessary parameters need to be specified: dynamic viscosity $\nu$ is network output, fluid density $\rho=1.0$.

#### 2.2.3 Computational Domain Construction

In this paper, the computational domain and parameter independent variable $\nu$ of this case are composed of point clouds generated by `numpy` random numbers, so the built-in point cloud geometry `PointCloud` of PaddleScience can be used directly to combine into a spatial `Geometry` computational domain.

``` py linenums="45"
--8<--
examples/pipe/poiseuille_flow.py:45:69
--8<--
```

#### 2.2.4 Constraint Construction

According to the formulas and boundary conditions obtained in [2.1 Problem Definition](#21-problem-definition), corresponding to several constraints guiding model training in the computational domain, namely:

- Navier-Stokes equation constraints imposed on internal points of the fluid domain

    Mass conservation:

    $$
    \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
    $$

    $x$ momentum conservation:

    $$
    u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} - \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2}) = 0
    $$

    $y$ momentum conservation:

    $$
    u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} - \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2}) = 0
    $$

    In order to facilitate obtaining intermediate variables, the `NavierStokes` class internally names the results on the left side of the above equation as `continuity`, `momentum_x`, `momentum_y` respectively.

- Dirichlet boundary condition constraints imposed on the inlet and outlet of the fluid domain, and the upper and lower vessel wall boundaries of the fluid domain. As one of the innovations of this paper, this case innovatively uses structured boundary conditions, that is, adding a formula layer after the output layer of the network to impose boundary conditions (the value of the formula is zero at the boundary). Avoid the deficiency that data points as boundary conditions cannot be effectively constrained. Unified use of class function `Transform()` for initialization and management. The specific inference process is:

    The formula form of the correction function for the upper and lower boundaries (vessel walls) of the fluid domain is:

    $$
    \hat{u}(t,x,\theta;W,b) = u_{par}(t,x,\theta) + D(t,x,\theta)\tilde{u}(t,x,\theta;W,b)
    $$

    $$
    \hat{p}(t,x,\theta;W,b) = p_{par}(t,x,\theta) + D(t,x,\theta)\tilde{p}(t,x,\theta;W,b)
    $$

    Where $u_{par}$ and $p_{par}$ are particular solutions satisfying boundary conditions and initial conditions. After substituting specific correction functions, we get:

    $$
    \hat{u} = (\dfrac{d^2}{4} - y^2) \tilde{u}
    $$

    $$
    \hat{v} = (\dfrac{d^2}{4} - y^2) \tilde{v}
    $$

    $$
    \hat{p} = \dfrac{x - x_{in}}{x_{out} - x_{in}}p_{out} + \dfrac{x_{out} - x}{x_{out} - x_{in}}p_{in} + (x - x_{in})(x_{out} - x) \tilde{p}
    $$

Next, use PaddleScience built-in `InteriorConstraint` and model `Transform` custom layer to construct the above two constraints.

- Internal point constraint

    Taking `InteriorConstraint` acting on internal points of the fluid domain as an example, the code is as follows:

    ``` py linenums="122"
    --8<--
    examples/pipe/poiseuille_flow.py:122:142
    --8<--
    ```

    The first parameter of `InteriorConstraint` is the equation expression, used to describe how to calculate the constraint target. Here fill in `equation["NavierStokes"].equations` instantiated in section [2.2.2 Equation Construction](#222);

    The second parameter is the target value of the constraint variable. In this problem, we hope that the three intermediate results `continuity`, `momentum_x`, `momentum_y` generated by the Navier-Stokes equation are optimized to 0, so set all their target values to 0;

    The third parameter is the computational domain where the constraint equation acts. Here fill in `interior_geom` instantiated in section [2.2.3 Computational Domain Construction](#223);

    The fourth parameter is the sampling configuration on the computational domain. Here we use batch data point training, so the `dataset` field is set to `NamedArrayDataset` and `iters_per_epoch` is also set to 1, and the sampling point number `batch_size` is set to 128;

    The fifth parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"mean"`, which means we will sum and average the loss terms generated by all data points participating in calculation;

    The sixth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "EQ".

#### 2.2.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Use 3000 training epochs, and learning rate set to 0.005.

#### 2.2.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="107"
--8<--
examples/pipe/poiseuille_flow.py:107:108
--8<--
```

#### 2.2.7 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="144"
--8<--
examples/pipe/poiseuille_flow.py:144:156
--8<--
```

On the other hand, the visualization and quantitative evaluation of this case mainly rely on:

1. Comparison of velocity $u(y)$ with $y$ at $x=0$ section under four different dynamic viscosity coefficient ${\nu}$ samplings and analytical solutions

2. When we select truncated Gaussian distribution of dynamic viscosity coefficient ${\nu}$ sampling (mean $\hat{\nu} = 10^{−3}$, variance $\sigma_{\nu}​=2.67 \times 10^{−4}$), comparison of probability density function of velocity at center and analytical solution

``` py linenums="159"
--8<--
examples/pipe/poiseuille_flow.py:159:261
--8<--
```

### 2.3 Complete Code

``` py linenums="1" title="poiseuille_flow.py"
--8<--
examples/pipe/poiseuille_flow.py
--8<--
```

### 2.4 Result Display

<figure markdown>
  ![laplace 2d]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/pipe_result.png){ loading=lazy }
  <figcaption>(Left) Comparison of velocity u(y) with y at x=0 section under four different dynamic viscosity coefficient samplings and analytical solutions (Right) When we select truncated Gaussian distribution of dynamic viscosity coefficient nu sampling (mean nu=0.001, variance sigma​=2.67 x 10e−4), comparison of probability density function of velocity at center and analytical solution</figcaption>
</figure>

The result of the DNN surrogate model is shown in the left figure, compared with the exact solution of Poiseuille flow (Equation 13 in the paper):

$$
u_a = \dfrac{\delta p}{2 \nu \rho L} + (\dfrac{d^2}{4} - y^2)
$$

$y$ in the formula and picture represents the spanwise coordinate, $\delta p$. From the picture, we can observe that the velocity curves (red dashed lines) under 4 different viscosity samplings predicted by DNN almost perfectly match the velocity curves of the analytical solution (blue solid lines). Among them, the Reynolds numbers ($Re$) of the 4 cases are 283, 121, 33, 3 respectively. In fact, as long as the Reynolds number is moderate, DNN can accurately predict pipe flow for any given dynamic viscosity coefficient.

The right figure shows the uncertainty of the centerline (pipe center in x direction) velocity under a given dynamic viscosity coefficient (Gaussian distribution). The Gaussian distribution of the dynamic viscosity coefficient has a mean of $1e^{-3}$ and a variance of $2.67e^{-4}$, which ensures that the dynamic viscosity coefficient is a positive random variable. In addition, the interval of this Gaussian distribution is $(0,+\infty)$, and the probability density function is:

$$
f(\nu ; \bar{\nu}, \sigma_{\nu}) = \dfrac{\dfrac{1}{\sigma_{\nu}} N(\dfrac{(\nu - \bar{\nu})}{\sigma_{\nu}})}{1 - \phi(-\dfrac{\bar{\nu}}{\sigma_{\nu}})}
$$

For more details, please refer to page 9 of the paper.

## 3. Case 2: Aneurysm Flow

### 3.1 Problem Definition

This paper mainly studies two types of typical vascular flows (with standardized vascular geometries), stenosis flow and aneurysm flow.
Stenosis blood flow refers to blood flow through blood vessels where the vessel wall narrows and re-expands. This local restriction of blood vessels is associated with many cardiovascular diseases, such as arteriosclerosis, stroke and heart attack.
Vascular blood flow within an aneurysm, i.e., arterial dilation due to weak vessel walls, is called aneurysm blood flow. Aneurysm rupture can lead to life-threatening conditions, for example, subarachnoid hemorrhage (SAH) due to cerebral aneurysm rupture, and the study of hemodynamics can improve diagnosis and basic understanding of aneurysm progression and rupture.

Although realistic vascular geometries are usually irregular and complex, including curvature, bifurcations and junctions, idealized stenosis and aneurysm models are studied here for proof of concept. Namely, both stenotic vessels and aneurysm vessels are idealized as axisymmetric tubes with varying cross-sectional radii, parameterized by the following functions,

Mass conservation:

$$
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
$$

$x$ momentum conservation:

$$
u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} + \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2})
$$

$y$ momentum conservation:

$$
u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} + \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2})
$$

We only focus on this fully developed flow and impose no-slip boundary conditions at the boundaries. Different from traditional PINNs methods, we force no-slip boundary conditions on boundaries through velocity function assumptions:
For fluid domain boundaries and internal circular boundaries of the fluid domain, Dirichlet boundary conditions need to be imposed:

<figure markdown>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm.png){ loading=lazy }
  <figcaption>Flow field diagram</figcaption>
</figure>

Fluid domain inlet boundary:

$$
p=0.1
$$

Fluid domain outlet boundary:

$$
p=0
$$

Fluid domain upper and lower boundaries:

$$
u=0, v=0
$$

### 3.2 Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

#### 3.2.1 Model Construction

In this case, each known coordinate point and geometry magnification factor $(x, y, scale)$ has its own transverse velocity $u$, longitudinal velocity $v$, and pressure $p$
Three unknown quantities to be solved. Here we use three relatively simple MLPs (Multilayer Perceptrons) to represent the mapping functions $f_1, f_2, f_3: \mathbb{R}^3 \to \mathbb{R}^3$ from $(x, y, scale)$ to $(u, v, p)$, i.e.:

$$
u= transform_{output}(f_1(transform_{input}(x, y, scale)))
$$

$$
v= transform_{output}(f_2(transform_{input}(x, y, scale)))
$$

$$
p= transform_{output}(f_3(transform_{input}(x, y, scale)))
$$

In the above formula, $f_1, f_2, f_3$ are the MLP models themselves, $transform_{input}, transform_{output}$, represent imposing additional structured custom layers for imposing constraints and linking inputs, expressed in PaddleScience code as follows:

``` py linenums="117"
--8<--
examples/aneurysm/aneurysm_flow.py:117:151
--8<--
```

In order to access the values of specific variables accurately and quickly during calculation, we specify here that the input variable names of the network model are `["x", "y", "scale"]`, and the output variable names are `["u", "v", "p"]`. These names are consistent with subsequent code.

Then by specifying the number of layers, number of neurons, and activation function of the MLP, we instantiated three neural network models `model_1` `model_2` `model_3` with 3 layers of hidden neurons and 1 layer of output layer neurons, each layer having 20 neurons, using "silu" as the activation function.

In addition, initialize weights and biases using `kaiming normal` method.

``` py linenums="106"
--8<--
examples/aneurysm/aneurysm_flow.py:106:115
--8<--
```

#### 3.2.2 Equation Construction

Since this case uses the 2D steady-state form of the Navier-Stokes equation, `NavierStokes` built into PaddleScience can be used directly.

``` py linenums="172"
--8<--
examples/aneurysm/aneurysm_flow.py:172:172
--8<--
```

When instantiating the `NavierStokes` class, necessary parameters need to be specified: dynamic viscosity $\nu = 0.001$, fluid density $\rho = 1.0$.

``` py linenums="37"
--8<--
examples/aneurysm/aneurysm_flow.py:37:41
--8<--
```

#### 3.2.3 Computational Domain Construction

In this paper, the computational domain and parameter independent variable $scale$ of this case are composed of point clouds generated by `numpy` random numbers, so the built-in point cloud geometry `PointCloud` of PaddleScience can be used directly to combine into a spatial `Geometry` computational domain.

``` py linenums="43"
--8<--
examples/aneurysm/aneurysm_flow.py:43:104
--8<--
```

#### 3.2.4 Constraint Construction

According to the formulas and boundary conditions obtained in [3.1 Problem Definition](#31), corresponding to several constraints guiding model training in the computational domain, namely:

- Navier-Stokes equation constraints imposed on internal points of the fluid domain

    Mass conservation:

    $$
    \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0
    $$

    $x$ momentum conservation:

    $$
    u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial x} - \nu(\dfrac{\partial ^2 u}{\partial x ^2} + \dfrac{\partial ^2 u}{\partial y ^2}) = 0
    $$

    $y$ momentum conservation:

    $$
    u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} +\dfrac{1}{\rho}\dfrac{\partial p}{\partial y} - \nu(\dfrac{\partial ^2 v}{\partial x ^2} + \dfrac{\partial ^2 v}{\partial y ^2}) = 0
    $$

    In order to facilitate obtaining intermediate variables, the `NavierStokes` class internally names the results on the left side of the above equation as `continuity`, `momentum_x`, `momentum_y` respectively.

- Dirichlet boundary condition constraints imposed on the inlet and outlet of the fluid domain, and the upper and lower vessel wall boundaries of the fluid domain. As one of the innovations of this paper, this case innovatively uses structured boundary conditions, that is, adding a formula layer after the output layer of the network to impose boundary conditions (the value of the formula is zero at the boundary). Avoid the deficiency that data points as boundary conditions cannot be effectively constrained. Unified use of class function `Transform()` for initialization and management. The specific inference process is:

    Let stenosis scaling factor be $A$:

    $$
    R(x) = R_{0} - A\dfrac{1}{\sqrt{2\pi\sigma^2}}exp(-\dfrac{(x-\mu)^2}{2\sigma^2})
    $$

    $$
    d = R(x)
    $$

    Specific correction functions are substituted to get:

    $$
    \hat{u} = (\dfrac{d^2}{4} - y^2) \tilde{u}
    $$

    $$
    \hat{v} = (\dfrac{d^2}{4} - y^2) \tilde{v}
    $$

    $$
    \hat{p} = \dfrac{x - x_{in}}{x_{out} - x_{in}}p_{out} + \dfrac{x_{out} - x}{x_{out} - x_{in}}p_{in} + (x - x_{in})(x_{out} - x) \tilde{p}
    $$

Next, use PaddleScience built-in `InteriorConstraint` and model `Transform` custom layer to construct the above two constraints.

- Internal point constraint

    Taking `InteriorConstraint` acting on internal points of the fluid domain as an example, the code is as follows:

    ``` py linenums="174"
    --8<--
    examples/aneurysm/aneurysm_flow.py:174:193
    --8<--
    ```

    The first parameter of `InteriorConstraint` is the equation expression, used to describe how to calculate the constraint target. Here fill in `equation["NavierStokes"].equations` instantiated in section [3.2.2 Equation Construction](#322);

    The second parameter is the target value of the constraint variable. In this problem, we hope that the three intermediate results `continuity`, `momentum_x`, `momentum_y` generated by the Navier-Stokes equation are optimized to 0, so set all their target values to 0;

    The third parameter is the computational domain where the constraint equation acts. Here fill in `interior_geom` instantiated in section [3.2.3 Computational Domain Construction](#323);

    The fourth parameter is the sampling configuration on the computational domain. Here we use batch data point training, so the `dataset` field is set to `NamedArrayDataset` and `iters_per_epoch` is also set to 1, and the sampling point number `batch_size` is set to 128;

    The fifth parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"mean"`, which means we will sum and average the loss terms generated by all data points participating in calculation;

    The sixth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "EQ".

#### 3.2.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Use 400 training epochs, and learning rate set to 0.005.

``` yaml linenums="33"
--8<--
examples/aneurysm/conf/aneurysm_flow.yaml:33:42
--8<--
```

#### 3.2.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="152"
--8<--
examples/aneurysm/aneurysm_flow.py:152:170
--8<--
```

#### 3.2.7 Model Training, Evaluation and Visualization (Need to download data)

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then inference.

``` py linenums="212"
--8<--
examples/aneurysm/aneurysm_flow.py:212:212
--8<--
```

### 3.3 Complete Code

``` py linenums="1" title="aneurysm_flow.py"
--8<--
examples/aneurysm/aneurysm_flow.py
--8<--
```

### 3.4 Result Display

<figure markdown>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm_result_1.png)<br>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm_result_2.png)<br>
  ![pipe]( https://paddle-org.bj.bcebos.com/paddlescience/docs/labelfree_DNN_surrogate/aneurysm_result_3.png)
  <figcaption>First row is velocity in x direction, second row is velocity in y direction, third row is wall shear stress curve</figcaption>
</figure>

The picture shows the solving ability for geometrically varying aneurysm flow, where training is performed by sampling the geometric scaling factor $A$ from the interval $0$ to $-2e^{-2}$. Flow field predictions for three different geometries are shown in the figure. The size of the aneurysm increases from left to right, the flow velocity decreases in the vessel expansion area, and decays most at the center of the aneurysm. From the first two rows of pictures, it can be seen that the CFD results and the model prediction results agree well. For WSS wall shear stress, the curve is also accurately captured by the model as the geometry changes.

For more details, refer to page 13 of the paper.

## 4. References

Reference: [Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data](https://arxiv.org/abs/1906.02382)

Reference code: [LabelFree-DNN-Surrogate](https://github.com/Jianxun-Wang/LabelFree-DNN-Surrogate)
