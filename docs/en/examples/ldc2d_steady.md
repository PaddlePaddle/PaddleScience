# 2D-LDC(2D Lid Driven Cavity Flow)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6137973" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Re=1000"

    === "Model Training Command"

        ``` sh
        # linux
        wget -c -P ./data/ \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat --create-dirs -o ./data/ldc_Re100.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat --create-dirs -o ./data/ldc_Re400.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_sota.py
        ```

    === "Model Evaluation Command"

        ``` sh
        # linux
        wget -c -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        python ldc_2d_Re3200_sota.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re1000_sota_pretrained.pdparams
        ```

    === "Model Export Command"

        ``` sh
        python ldc_2d_Re3200_sota.py mode=export
        ```

    === "Model Inference Command"

        ``` sh
        # linux
        wget -c -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        python ldc_2d_Re3200_sota.py mode=infer
        ```

    | Pretrained Model | $Re$ | Metrics |
    | :-- | :-- | :-- |
    | - | 100 | U_validator/loss: 0.00017<br>U_validator/L2Rel.U: 0.04875 |
    | - | 400 | U_validator/loss: 0.00047<br>U_validator/L2Rel.U: 0.07554 |
    | [**ldc_re1000_sota_pretrained.pdparams**](https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re1000_sota_pretrained.pdparams) | 1000 | **U_validator/loss: 0.00053<br>U_validator/L2Rel.U: 0.07777** |
    | - | 3200 | U_validator/loss: 0.00227<br>U_validator/L2Rel.U: 0.15440 |

=== "Re=3200"

    === "Model Training Command"

        ``` sh
        # linux
        wget -c -P ./data/ \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1600.mat \
            https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat --create-dirs -o ./data/ldc_Re100.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat --create-dirs -o ./data/ldc_Re400.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat --create-dirs -o ./data/ldc_Re1000.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1600.mat --create-dirs -o ./data/ldc_Re1600.mat
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_piratenet.py
        ```

    === "Model Evaluation Command"

        ``` sh
        # linux
        wget -c -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_piratenet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re3200_piratenet_pretrained.pdparams
        ```

    === "Model Export Command"

        ``` sh
        python ldc_2d_Re3200_piratenet.py mode=export
        ```

    === "Model Inference Command"

        ``` sh
        # linux
        wget -c -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat --create-dirs -o ./data/ldc_Re3200.mat
        python ldc_2d_Re3200_piratenet.py mode=infer
        ```

    | Pretrained Model | $Re$ | Metrics |
    | :-- | :-- | :-- |
    | - | 100 | U_validator/loss: 0.00016<br>U_validator/L2Rel.U: 0.04741 |
    | - | 400 | U_validator/loss: 0.00071<br>U_validator/L2Rel.U: 0.09288 |
    | - | 1000 | U_validator/loss: 0.00191<br>U_validator/L2Rel.U: 0.14797 |
    | - | 1600 | U_validator/loss: 0.00276<br>U_validator/L2Rel.U: 0.17360 |
    | [**ldc_re3200_piratenet_pretrained.pdparams**](https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re3200_piratenet_pretrained.pdparams) | 3200 | **U_validator/loss: 0.00016<br>U_validator/L2Rel.U: 0.04166** |

!!! Note

    This case only provides pre-trained models under $Re=1000/3200$. If you need pre-trained models under other Reynolds numbers, please execute the training command manually to train and obtain model weights under each Reynolds number.

## 1. Background Introduction

The 2D Lid Driven Cavity Flow (LDC) problem is applied in many fields. For example, this problem can be used to verify the validity of calculation methods in the field of computational fluid dynamics (CFD). Although the boundary conditions of this problem are relatively simple, its flow characteristics are very complex. In the lid-driven flow LDC, the top wall moves in the x-direction at a speed of U=1, while the other three walls are defined as no-slip boundary conditions, that is, the speed is zero.

In addition, the LDC problem is also used to study and predict flow phenomena in aerodynamics. For example, in the automotive industry, simulating and analyzing the air flow inside the car body can help optimize the design and performance of the vehicle.

In general, the LDC problem has been widely used in computational fluid dynamics, aerodynamics and related fields, and has played an important role in studying and predicting flow phenomena and optimizing product design.

## 2. Problem Definition

This case assumes $Re=3200$, the calculation domain is a square cavity with length and width both being 1, and the following formula is applied to study the **steady** flow field problem of lid-driven cavity flow:

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

**Let:**

$t^* = \dfrac{L}{U_0}$

$x^*=y^* = L$

$u^*=v^* = U_0$

$p^* = \rho {U_0}^2$

**Define:**

Dimensionless coordinate $x: X = \dfrac{x}{x^*}$; Dimensionless coordinate $y: Y = \dfrac{y}{y^*}$

Dimensionless velocity $x: U = \dfrac{u}{u^*}$; Dimensionless velocity $y: V = \dfrac{v}{u^*}$

Dimensionless pressure $P = \dfrac{p}{p^*}$

Reynolds number $Re = \dfrac{L U_0}{\nu}$

Then the following dimensionless Navier-Stokes equation can be obtained, applied inside the cavity:

Mass conservation:

$$
\dfrac{\partial U}{\partial X} + \dfrac{\partial U}{\partial Y} = 0
$$

$x$ momentum conservation:

$$
U\dfrac{\partial U}{\partial X} + V\dfrac{\partial U}{\partial Y} = -\dfrac{\partial P}{\partial X} + \dfrac{1}{Re}(\dfrac{\partial ^2 U}{\partial X^2} + \dfrac{\partial ^2 U}{\partial Y^2})
$$

$y$ momentum conservation:

$$
U\dfrac{\partial V}{\partial X} + V\dfrac{\partial V}{\partial Y} = -\dfrac{\partial P}{\partial Y} + \dfrac{1}{Re}(\dfrac{\partial ^2 V}{\partial X^2} + \dfrac{\partial ^2 V}{\partial Y^2})
$$

For the cavity boundary, Dirichlet boundary conditions need to be imposed:

Upper boundary:

$$
u(x, y) = 1 − \dfrac{\cosh (C_0(x − 0.5))} {\cosh (0.5C_0)} ,
$$

Left boundary, lower boundary, right boundary:

$$
u=0, v=0
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the 2D-LDC problem, each known coordinate point $(x, y)$ has its own lateral velocity $u$, longitudinal velocity $v$, and pressure $p$
Three unknown quantities to be solved. Here we use PirateNet suitable for PINN tasks to represent the mapping function $f: \mathbb{R}^2 \to \mathbb{R}^3$ from $(x, y)$ to $(u, v, p)$, i.e.:

$$
u, v, p = f(x, y)
$$

In the above formula, $f$ is the `PirateNet` model itself, expressed in PaddleScience code as follows

``` py linenums="41"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:41:42
--8<--
```

The `cfg.MODEL` configuration is as follows:

``` yaml linenums="38"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:38:41
--8<--
```

In order to accurately and quickly access the value of a specific variable during calculation, we specify here that the input variable name of the network model is `["x", "y"]` and the output variable name is `["u", "v", "p"]`. These names are consistent with the subsequent code.

As shown above, by specifying the number of layers, number of neurons, and activation function of `PirateNet`, we instantiate a neural network model `model` with 12 layers of hidden neurons, 256 neurons per layer, using "tanh" as the activation function.

### 3.2 Curriculum Learning

To speed up convergence, we use the Curriculum learning method to train the model, that is, first train the model at a low Reynolds number, then gradually increase the Reynolds number, and finally reach convergence at a high Reynolds number.

``` py linenums="210"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:210:211
--8<--
```

### 3.3 Equation Construction

Since 2D-LDC uses the 2D steady-state form of the Navier-Stokes equation, `NavierStokes` built into PaddleScience can be used directly.

``` py linenums="88"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:88:91
--8<--
```

In the curriculum learning function, we need to specify the necessary parameters when instantiating the `NavierStokes` class: dynamic viscosity $\nu=\frac{1}{Re}$, fluid density $\rho=1.0$, where $Re$ is a variable that will gradually increase during the training process.

### 3.4 Computational Domain Construction

The data required for the training and evaluation of the 2D-LDC problem in this article is obtained by reading the file corresponding to the Reynolds number.

``` py linenums="93"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:93:103
--8<--
```

### 3.5 Constraint Construction

According to the dimensionless formula and boundary conditions obtained in [2. Problem Definition](#2-problem-definition), corresponding to two constraints guiding model training in the computational domain, namely:

1. Dimensionless Navier-Stokes equation constraint imposed on internal points of the rectangle (after simple term shifting)

    $$
    \dfrac{\partial U}{\partial X} + \dfrac{\partial U}{\partial Y} = 0
    $$

    $$
    U\dfrac{\partial U}{\partial X} + V\dfrac{\partial U}{\partial Y} + \dfrac{\partial P}{\partial X} - \dfrac{1}{Re}(\dfrac{\partial ^2 U}{\partial X^2} + \dfrac{\partial ^2 U}{\partial Y^2}) = 0
    $$

    $$
    U\dfrac{\partial V}{\partial X} + V\dfrac{\partial V}{\partial Y} + \dfrac{\partial P}{\partial Y} - \dfrac{1}{Re}(\dfrac{\partial ^2 V}{\partial X^2} + \dfrac{\partial ^2 V}{\partial Y^2}) = 0
    $$

    In order to facilitate obtaining intermediate variables, the `NavierStokes` class internally names the results on the left side of the above equation as `continuity`, `momentum_x`, `momentum_y`.

2. Dirichlet boundary condition constraints imposed on the upper, lower, left, and right boundaries of the rectangle

    Upper boundary:

    $$
    u(x, y) = 1 − \dfrac{\cosh (C_0(x − 0.5))} {\cosh (0.5C_0)} ,
    $$

    Left boundary, lower boundary, right boundary:

    $$
    u=0, v=0
    $$

Next, use `SupervisedConstraint` built into PaddleScience to construct the above two constraints.

#### 3.5.1 Interior Point Constraint

Taking `SupervisedConstraint` acting on rectangular internal points as an example, the code is as follows:

``` py linenums="105"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:105:132
--8<--
```

The first parameter of `SupervisedConstraint` is the dataset configuration, used to describe how to construct input data. Here fill in the constructor functions `gen_input_batch` and `gen_label_batch` for input data and label data, and the dataset name `ContinuousNamedArrayDataset`;

The second parameter is the target value of the constraint variable. Here fill in `equation["NavierStokes"].equations` instantiated in the [3.3 Equation Construction](#33-equation-construction) section;

The third parameter is the loss function. Here we choose the commonly used `MSE` function, and `reduction` is set to `"mean"`, which means we will average the loss terms generated by all data points participating in calculation;

The fourth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "PDE".

#### 3.5.2 Boundary Constraint

Process the label data of the upper boundary according to the above corresponding formula, and set the label data of other points to 0. Then continue to construct the Dirichlet constraint of the cavity boundary, we still use the `SupervisedConstraint` class.

``` py linenums="134"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:134:160
--8<--
```

After the differential equation constraint, boundary constraint, and initial value constraint are constructed, encapsulate them into a dictionary with the name we just named as the keyword for subsequent access.

``` py linenums="161"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:161:165
--8<--
```

### 3.6 Hyperparameter Setting

Next, you need to specify the number of training rounds in the configuration file, training 10, 20, 50, 50, 500 rounds on Re=100, 400, 1000, 1600, 3200 respectively, with 1000 iterations per round.

``` yaml linenums="33"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:33:35
--8<--
```

Secondly, set a suitable learning rate decay strategy,

``` yaml linenums="52"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:52:66
--8<--
```

Finally, set the automatic loss balancing strategy during training to `GradNorm`,

``` yaml linenums="72"
--8<--
examples/ldc/conf/ldc_2d_Re3200_piratenet.yaml:72:75
--8<--
```

### 3.7 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the commonly used `Adam` optimizer is selected.

``` py linenums="44"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:44:48
--8<--
```

### 3.8 Validator Construction

During the training process, the training status of the current model is usually evaluated using the validation set (test set) at a certain round interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator.

``` py linenums="167"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:167:185
--8<--
```

Here calculate the prediction error of $U=\sqrt{u^2+v^2}$;

Evaluation metric `metric` select `ppsci.metric.L2Rel`;

The rest of the configuration is similar to the setting of [Constraint Construction](#35-constraint-construction).

### 3.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="187"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py:187:208
--8<--
```

## 4. Complete Code

``` py linenums="1" title="ldc_2d_Re3200_piratenet.py"
--8<--
examples/ldc/ldc_2d_Re3200_piratenet.py
--8<--
```

## 5. Result Display

The following shows the prediction result $U=\sqrt{u^2+v^2}$ of the model for the internal points of the square computational domain with a side length of 1.

=== "Re=1000"

    <figure markdown>
    ![ldc_re1000_sota_ac](https://paddle-org.bj.bcebos.com/paddlescience/docs/ldc/ldc_re1000_sota_ac.png){ loading=lazy }
    <figcaption> </figcaption>
    </figure>

    It can be seen that at $Re=1000$, the prediction result is basically the same as the solver result (L2 relative error is 7.7%).

=== "Re=3200"

    <figure markdown>
    ![ldc_re3200_piratenet_ac](https://paddle-org.bj.bcebos.com/paddlescience/docs/ldc/ldc_re3200_piratenet_ac.png){ loading=lazy }
    <figcaption></figcaption>
    </figure>

    It can be seen that at $Re=3200$, the prediction result is basically the same as the solver result (L2 relative error is 4.1%).

## 6. References

- [PIRATENETS: PHYSICS-INFORMED DEEP LEARNING WITHRESIDUAL ADAPTIVE NETWORKS](https://arxiv.org/pdf/2402.00326.pdf)
- [jaxpi LDC example](https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/ldc#readme)
