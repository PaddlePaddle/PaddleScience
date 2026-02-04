# 2D-Biharmonic

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

=== "Model Training Command"

    ``` sh
    python biharmonic2d.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python biharmonic2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/biharmonic2d/biharmonic2d_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python biharmonic2d.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python biharmonic2d.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [biharmonic2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/biharmonic2d/biharmonic2d_pretrained.pdparams) | l2_error: 0.02774 |

## 1. Background Introduction

The Biharmonic Equation is an equation that characterizes the relationship between stress, strain, and load. It is a fourth-order partial differential equation, so it is difficult to solve in traditional numerical methods. This case attempts to use the PINNs (Physics Informed Neural Networks) method to solve the application problem of the Biharmonic Equation on a 2D rectangular plate, and uses deep learning methods to solve it based on linear elasticity and other equations.

## 2. Problem Definition

The structure of this case is a rectangular plate with length, width and thickness of 2 m, 3 m and 0.01 m respectively. The plate is fixed around the perimeter, and a sinusoidal distribution load $q=q_0sin(\dfrac{\pi x}{a})sin(\dfrac{\pi x}{b})$ is applied to the surface, where $q_0=980 Pa$. The PDE equation is the Biharmonic Equation in 2D, and the formula is:

$$\nabla^4w=(\dfrac{\partial^2}{\partial x^2}+\dfrac{\partial^2}{\partial y^2})(\dfrac{\partial^2}{\partial x^2}+\dfrac{\partial^2}{\partial y^2})w=\dfrac{q}{D}$$

Where $w$ is the plate deflection, $D$ is the bending stiffness, which can be calculated as follows:

$$D=\dfrac{Et^3}{12(1-\nu^2)}$$

Where $E=201880.0e+6 Pa$ is the Young's modulus of elasticity, and $\nu=0.25$ is the Poisson's ratio.

Based on the plate deflection $w$, torque and shear force can be calculated as follows:

$$
\begin{cases}
  M_x=-D(\dfrac{\partial^2w}{\partial x^2}+\nu\dfrac{\partial^2w}{\partial y^2}) \\
  M_y=-D(\dfrac{\partial^2w}{\partial y^2}+\nu\dfrac{\partial^2w}{\partial x^2}) \\
  M_{xy}=D(1-\nu\dfrac{\partial^2w}{\partial x y}) \\
  Q_x=-D\dfrac{\partial}{\partial x}(\dfrac{\partial^2w}{\partial x^2}+\dfrac{\partial^2w}{\partial y^2}) \\
  Q_y=-D\dfrac{\partial}{\partial y}(\dfrac{\partial^2w}{\partial x^2}+\dfrac{\partial^2w}{\partial y^2}) \\
\end{cases}
$$

Since the plate is fixed around the perimeter, on $x=0$ and $x=x_{max}$, the deflection $w$ and the moment $M_y$ in the $y$ direction are 0; on $y=0$ and $y=y_{max}$, the deflection $w$ and the moment $M_x$ in the $x$ direction are 0, that is:

$$
\begin{cases}
  w|_{x=0\ |\ x=\ a}=0 \\
  M_y|_{x=0\ |\ x=\ a}=0 \\
  w|_{y=0\ |\ y=\ b}=0 \\
  M_x|_{y=0\ |\ y=\ b}=0 \\
\end{cases}
$$

The goal is to solve the deflection $w$ of each point on the plate surface, and calculate the moment and shear force $M_x$, $M_y$, $M_{xy}$, $Q_x$, $Q_y$, a total of 6 physical quantities. The constant definition code is as follows:

``` yaml linenums="28"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:28:34
--8<--
```

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the biharmonic2d problem, each known coordinate point $(x, y)$ has corresponding unknown quantities to be solved: deflection $w$ in the force direction (i.e., z direction), moments $(M_x, M_y, M_{xy})$ and shear forces $(Q_x, Q_y)$. However, since moments and shear forces are calculated from deflection, the only unknown quantity that actually needs to be solved is deflection $w$, so only one model needs to be constructed:

$$w = f(x,y)$$

In the above formula, $f$ is the deflection model `disp_net`, expressed in PaddleScience code as follows:

``` py linenums="77"
--8<--
examples/biharmonic2d/biharmonic2d.py:77:78
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, the input variable name of the strain model is specified as `("x", "y")`. In order to match the PaddleScience built-in equation API ppsci.equation.Biharmonic, the output variable name is `("u")` instead of `("w")`. These names are consistent with the subsequent code.

Then by specifying the number of layers and neurons of MLP, a neural network model `disp_net` with 5 hidden layers and 20 neurons per layer is instantiated, using `tanh` as the activation function, and using `WeightNorm` weight normalization.

### 3.2 Equation Construction

This case involves the biharmonic equation, so PaddleScience's built-in `ppsci.equation.Biharmonic` can be used. Since the load $q$ is a non-uniform load, a custom load distribution function needs to be defined and passed to the API.

``` py linenums="84"
--8<--
examples/biharmonic2d/biharmonic2d.py:84:91
--8<--
```

### 3.3 Computational Domain Construction

Since the height of the plate is very small, the geometric area of this problem is considered to be a 2D rectangle with length 2 and width 3, constructed by PaddleScience's built-in `ppsci.geometry.Rectangle` API:

``` py linenums="93"
--8<--
examples/biharmonic2d/biharmonic2d.py:93:95
--8<--
```

### 3.4 Constraint Construction

This case involves 9 constraints. Before constructing specific constraints, data reading configuration can be constructed first, so that this configuration can be reused when constructing multiple constraints later.

``` py linenums="97"
--8<--
examples/biharmonic2d/biharmonic2d.py:97:106
--8<--
```

#### 3.4.1 Interior Constraint

Taking `InteriorConstraint` acting on the interior points of the backplane as an example, the code is as follows:

``` py linenums="205"
--8<--
examples/biharmonic2d/biharmonic2d.py:205:214
--8<--
```

The first parameter of `InteriorConstraint` is the equation (system) expression, which is used to describe how to calculate the constraint target. Here, fill in `equation["Biharmonic"].equations` instantiated in the [3.2 Equation Construction](#32-equation-construction) chapter;

The second parameter is the target value of the constraint variable. In this problem, it is hoped that 1 value `biharmonic` related to the Biharmonic equation is optimized to 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["geo"]` instantiated in the [3.3 Computational Domain Construction](#33-computational-domain-construction) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here, `batch_size` is set to:

``` yaml linenums="57"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:57:59
--8<--
```

The fifth parameter is the loss function. Here, the commonly used MSE function is selected, and `reduction` is set to `"mean"`, that is, the loss terms generated by all data points involved in the calculation will be summed;

The sixth parameter is geometric point filtering. Since this constraint is only applied to the backplane area, the points sampled on geo need to be filtered. Just pass in a lambda filter function here, which accepts the tensor `x, y` formed by the point set, and returns a boolean tensor indicating whether each point meets the filtering conditions. Not meeting is `False`, meeting is `True`;

The seventh parameter is the weight of each point participating in the loss calculation. Here it is set to:

``` yaml linenums="60"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:60:62
--8<--
```

The eighth parameter is the name of the constraint condition. Each constraint condition needs to be named to facilitate subsequent indexing. Here it is named "INTERIOR".

#### 3.4.2 Boundary Constraint

As mentioned in [2. Problem Definition](#2-problem-definition), the deflection $w$ at $x=0$ is 0. There are the following boundary conditions, and the other 7 boundary conditions are similar:

``` py linenums="108"
--8<--
examples/biharmonic2d/biharmonic2d.py:108:118
--8<--
```

After the equation constraint and boundary constraint are constructed, encapsulate them into a dictionary with the names just given as keys for subsequent access.

``` py linenums="215"
--8<--
examples/biharmonic2d/biharmonic2d.py:215:226
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, `Adam` is selected for a small amount of training first, and then the `LBFGS` optimizer is used for fine-tuning.

``` py linenums="81"
--8<--
examples/biharmonic2d/biharmonic2d.py:81:83
--8<--
```

### 3.6 Hyperparameter Setting

Next, you need to specify optimizer parameters such as training rounds and learning rate in the configuration file.

``` yaml linenums="46"
--8<--
examples/biharmonic2d/conf/biharmonic2d.yaml:46:56
--8<--
```

### 3.7 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training. Note that the two optimization processes need to build `Solver` separately.

``` py linenums="228"
--8<--
examples/biharmonic2d/biharmonic2d.py:228:267
--8<--
```

### 3.8 Model Evaluation and Visualization

After training, the trained model can be evaluated and visualized in `eval` mode. Due to the specificity of the case, there is no need to build a validator and visualizer, but use custom code.

``` py linenums="270"
--8<--
examples/biharmonic2d/biharmonic2d.py:270:350
--8<--
```

## 4. Complete Code

``` py linenums="1" title="biharmonic2d.py"
--8<--
examples/biharmonic2d/biharmonic2d.py
--8<--
```

## 5. Result Display

The following shows the model prediction results and theoretical solution results of deflection $w$, moments $M_x, M_y, M_{xy}$ and shear forces $Q_x, Q_y$.

<figure markdown>
  ![biharmonic2d_pred.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/biharmonic2d/eval_Mx_Mxy_My_Qx_Qy_w.png){ loading=lazy }
  <figcaption>Model prediction results of moments Mx, My, Mxy, shear forces Qx, Qy and deflection w</figcaption>
</figure>

<figure markdown>
  ![biharmonic2d_label_M.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/biharmonic2d/label_M.png){ loading=lazy }
  <figcaption>Theoretical solution results of moments Mx, My, Mxy</figcaption>
</figure>

<figure markdown>
  ![biharmonic2d_label_Q_w.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/biharmonic2d/label_Q_w.png){ loading=lazy }
  <figcaption>Theoretical solution results of shear forces Qx, Qy and deflection w</figcaption>
</figure>

It can be seen that the model prediction results are basically consistent with the theoretical solution results.

## 6. References

Reference: [A Physics Informed Neural Network Approach to Solution and Identification of Biharmonic Equations of Elasticity](https://arxiv.org/abs/2108.07243)
