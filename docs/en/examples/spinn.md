# SPINN (helmholtz3d)

<a href="https://aistudio.baidu.com/projectdetail/8219967" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    python helmholtz3d.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python helmholtz3d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/spinn/spinn_helmholtz3d_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python helmholtz3d.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python helmholtz3d.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [spinn_helmholtz3d.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/spinn/spinn_helmholtz3d_pretrained.pdparams) | l2_err: 0.0183 <br> rmse: 0.0064 |

## 1. Background Introduction

The Helmholtz equation is an important partial differential equation widely used in physics and engineering, especially in wave theory and vibration problems. It is named after the German physicist Hermann von Helmholtz. The standard form of the Helmholtz equation is as follows:

$$
\nabla^2 u + k^2 u = q
$$

Here:

- $\nabla^2$ is the Laplace operator (also known as the Laplacian), which in a three-dimensional Cartesian coordinate system takes the form: $\nabla^2 = \frac{\partial^2 }{\partial x^2} + \frac{\partial^2 }{\partial y^2} + \frac{\partial^2 }{\partial z^2}$
- $u$ is the function to be solved, usually representing the amplitude of a physical quantity, such as electromagnetic field, acoustic pressure, or quantum wave function.
- $k$ is the wave number, defined as $k = \frac{2\pi}{\lambda}$, where $\lambda$ is the wavelength.
- $q$ is the source term, usually representing the interaction between physical quantities and time and space derivatives.

This case solves the following three-dimensional Helmholtz equation:

$$
\begin{aligned}
  & \nabla^2 u + k^2 u = q, x \in \Omega \\
  & u(x) = 0, x \in \partial \Omega \\
\end{aligned}
$$

$$
\begin{aligned}
  & \text{source term } q = -(a_1 \pi)^2 u -(a_2 \pi)^2 u -(a_3 \pi)^2 u + k^2 u \\
  & \text{where }k=1, a_1=4, a_2=4, a_3=3
\end{aligned}
$$

## 2. Problem Definition

The computational domain of this problem is within a unit cube $[-1, 1] ^3$. For the interior points of the computational domain, the above Helmholtz equation is required to be satisfied, and for the boundary points of the computational domain, $u = 0$ is required.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

The model structure design of SPINN is as follows:

![SPINN_structure](https://paddle-org.bj.bcebos.com/paddlescience/docs/spinn/spinn_structure.png)

In the Helmholtz problem, each known coordinate point $(x, y, z)$ has a corresponding unknown quantity $u$ to be solved (here we use $u$ instead). Here, SPINN is used to represent the mapping function $f: \mathbb{R}^3 \to \mathbb{R}^1$ from $(x, y, z)$ to $(u)$, that is:

$$
u = m(x, y, z)
$$

In the above formula, $m$ is the SPINN model itself, expressed in PaddleScience code as follows

``` py linenums="99"
--8<--
examples/spinn/helmholtz3d.py:99:100
--8<--
```

In order to accurately and quickly access the value of specific variables during calculation, we specify here that the input variable names of the network model are `("x", "y", "z")`, and the output variable name is `("u")`. These names are consistent with subsequent code.

Then by specifying the number of layers and neurons of SPINN, we instantiate a neural network model `model` with 4 fully connected layers, each with 64 neurons, and the hidden layer feature dimension `r` of each output variable is 32, and `tanh` is used as the activation function.

``` yaml linenums="38"
--8<--
examples/spinn/conf/helmholtz3d.yaml:38:45
--8<--
```

### 3.2 Equation Construction

The Helmholtz differential equation can be represented by the following code:

``` py linenums="102"
--8<--
examples/spinn/helmholtz3d.py:102:104
--8<--
```

Note: Here we need to manually pass the model to `equation["Helmholtz"]` because the `Helmholtz` equation needs to use the forward differentiation function.

### 3.3 Constraint Construction

#### 3.3.1 Interior Point Constraint

Taking `SupervisedConstraint` acting on interior points as an example, the code for generating interior point training data is as follows:

``` py linenums="39"
--8<--
examples/spinn/helmholtz3d.py:39:83
--8<--
```

The code for constructing interior point constraints is as follows:

``` py linenums="106"
--8<--
examples/spinn/helmholtz3d.py:106:156
--8<--
```

The first parameter of `SupervisedConstraint` is the data configuration used for training. Since we use real-time randomly generated data instead of fixed data points, we fill in the custom input data/label generation function;

The second parameter is the equation expression, so pass in the Helmholtz equation object;

The third parameter is the loss function, here `MSELoss` is selected;

The fourth parameter is the name of the constraint condition. Each constraint condition needs to be named for subsequent indexing. Here it is named "PDE".

#### 3.3.2 Boundary Value Constraint

The third constraint condition is the boundary value constraint, and the code is as follows:

``` py linenums="158"
--8<--
examples/spinn/helmholtz3d.py:158:190
--8<--
```

### 3.4 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, we use 50 training epochs, 1000 steps per epoch, and an initial learning rate of 0.001.

``` yaml linenums="47"
--8<--
examples/spinn/conf/helmholtz3d.yaml:47:63
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the commonly used `Adam` optimizer is selected, and the ExponentialDecay learning rate adjustment strategy commonly used in machine learning is used together.

``` py linenums="192"
--8<--
examples/spinn/helmholtz3d.py:192:196
--8<--
```

### 3.6 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="198"
--8<--
examples/spinn/helmholtz3d.py:198:227
--8<--
```

## 4. Complete Code

``` py linenums="1" title="helmholtz3d.py"
--8<--
examples/spinn/helmholtz3d.py
--8<--
```

## 5. Result Display

Sample $100^3$ points uniformly on the computational domain, and their prediction results and analytical solutions are shown in the figure below.

<figure markdown>
  ![spinn_helmholtz3d.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/spinn/spinn_helmholtz3d.png){ loading=lazy }
  <figcaption> Left is PaddleScience prediction result, right is analytical solution result</figcaption>
</figure>

The error predicted by the model in this problem is l2_err = 0.0183, rmse = 0.0064, which is small and basically consistent with the analytical solution error.

## 6. References

- [Separable Physics-Informed Neural Networks](https://arxiv.org/pdf/2306.15969)
- [SPINN](https://github.com/stnamjef/SPINN?tab=readme-ov-file)
