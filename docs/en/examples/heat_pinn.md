# Heat_PINN

=== "Model Training Command"

    ``` sh
    python heat_pinn.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python heat_pinn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heat_pinn/heat_pinn_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python heat_pinn.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python heat_pinn.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [heat_pinn_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heat_pinn/heat_pinn_pretrained.pdparams) | norm MSE loss between the FDM and PINN is 1.30174e-03 |

## 1. Background Introduction

Heat conduction is a common phenomenon in nature and is widely used in engineering, science and technology fields. Heat conduction problems have wide applications and importance in many fields, playing a crucial role in improving energy efficiency, improving material properties, promoting scientific research and promoting technological innovation. Therefore, understanding and simulating heat transfer processes is crucial for designing and optimizing heat transfer equipment, materials and systems. The 2D steady heat conduction equation describes the steady-state heat conduction process. Traditional solution methods involve using numerical methods such as finite element method or finite difference method, which usually require discretizing the domain and solving large-scale matrix systems. In recent years, Physics-informed neural networks (PINN) have gradually become a new method for solving partial differential equations. PINN combines the flexibility of neural networks with the modeling ability of physical constraints, and can directly solve partial differential equation problems in continuous domains.

## 2. Problem Definition

Assume that in the two-dimensional heat conduction equation, the temperature $T$ at each position $(x,y)$ satisfies the following relationship:

$$
\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}=0,
$$

And in the following area:

$$
D = \{(x, y)|-1\leq{x}\leq{+1},-1\leq{y}\leq{+1}\},
$$

With the following boundary conditions:

$$
\begin{cases}
T(-1, y) = 75.0 ^\circ{C}, \\
T(+1, y) = 0.0 ^\circ{C}, \\
T(x, -1) = 50.0 ^\circ{C}, \\
T(x, +1) = 0.0 ^\circ{C}.
\end{cases}
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the two-dimensional heat conduction problem, each known coordinate point $(x, y)$ has a corresponding unknown quantity $T$ to be solved. Here we use a relatively simple MLP (Multilayer Perceptron) to represent the mapping function $f: \mathbb{R}^2 \to \mathbb{R}^1$ from $(x, y)$ to $u$, i.e.:

$$
u = f(x, y),
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows:

``` py linenums="102"
--8<--
examples/heat_pinn/heat_pinn.py:102:103
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, we specify the input variable name of the network model as `("x", "y")` and the output variable name as `"u"`, these names are consistent with the subsequent code.

Then by specifying the number of layers, number of neurons and activation function of MLP, we instantiated a neural network model `model` with 9 hidden layers, 20 neurons per layer, and activation function `tanh`.

### 3.2 Equation Construction

Since the two-dimensional heat conduction equation uses the 2D form of the Laplace equation, the `Laplace` built in PaddleScience can be used directly, specifying the parameter `dim` of this class as 2.

``` py linenums="105"
--8<--
examples/heat_pinn/heat_pinn.py:105:106
--8<--
```

### 3.3 Computational Domain Construction

In this article, the two-dimensional heat conduction problem acts on a two-dimensional rectangular area with (-1.0, -1.0), (1.0, 1.0) as diagonals, so the spatial geometry `Rectangle` built in PaddleScience can be used directly as the computational domain.

``` py linenums="108"
--8<--
examples/heat_pinn/heat_pinn.py:108:109
--8<--
```

### 3.4 Constraint Construction

In this case, we use two constraints to guide the training of the model in the computational domain, namely the heat conduction equation constraint acting on the sampling points and the constraint acting on the boundary points.

Before defining constraints, you need to specify the number of sampling points for each constraint, indicating the number of sampled data for each constraint in its corresponding computational domain, as well as general sampling configuration.

``` py linenums="117"
--8<--
examples/heat_pinn/heat_pinn.py:117:122
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on internal points as an example, the code is as follows:

``` py linenums="123"
--8<--
examples/heat_pinn/heat_pinn.py:123:131
--8<--
```

The first parameter of `InteriorConstraint` is the equation expression, used to describe how to calculate the constraint target. Here, fill in `equation["Laplace"].equations` instantiated in the [3.2 Equation Construction](#32) chapter;

The second parameter is the target value of the constraint variable. According to the definition of the heat conduction equation, we hope that the result generated by the Laplace equation is all 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["rect"]` instantiated in the [3.3 Computational Domain Construction](#33) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here we use full data points for training, so the `dataset` field is set to "IterableNamedArrayDataset" and `iters_per_epoch` is also set to 1, and the sampling point number `batch_size` is set to `NPOINT_PDE` (indicating a 99x99 sampling grid);

The fifth parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"mean"`, that is, we will average the loss terms generated by all data points involved in the calculation;

The sixth parameter is the weight of the constraint when calculating loss. Referring to the PINN paper, we set it to 1 here;

The seventh parameter is to choose whether to perform equidistant sampling on the computational domain. Here we choose to enable equidistant sampling, so that the training points can be evenly distributed on the computational domain, which is conducive to training convergence;

The eighth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "EQ".

#### 3.4.2 Boundary Constraint

Similarly, we also need to construct constraints for the four boundaries of the rectangle. However, unlike constructing `InteriorConstraint`, since the action area is the boundary, we use the `BoundaryConstraint` class, code as follows:

``` py linenums="132"
--8<--
examples/heat_pinn/heat_pinn.py:132:171
--8<--
```

The first parameter of the `BoundaryConstraint` class indicates that we directly use the output result `out["u"]` of the network model as the constraint object during program operation;

The second parameter refers to how much the true value of our constraint object is. In this problem, the boundary condition is Dirichlet boundary condition, that is, the boundary condition directly describes the physical quantity on the boundary of the physical system, given a fixed boundary value. Specific boundary condition values have been given in [2. Problem Definition](#2);

The meanings of other parameters of the `BoundaryConstraint` class are basically consistent with `InteriorConstraint` and will not be introduced here.

After the differential equation constraint and boundary constraint are constructed, encapsulate them into a dictionary with the names we just named as keys for subsequent access.

``` py linenums="172"
--8<--
examples/heat_pinn/heat_pinn.py:172:179
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and the learning rate is set to 0.0005.

``` py linenums="181"
--8<--
examples/heat_pinn/heat_pinn.py:181:182
--8<--
```

### 3.6 Model Training

After completing the above settings, you only need to pass all the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="184"
--8<--
examples/heat_pinn/heat_pinn.py:184:201
--8<--
```

### 3.7 Model Evaluation and Visualization

After the model training is completed, it is necessary to compare it with the result calculated by the formal FDM method. Here we use `geom["rect"].sample_interior` to sample the coordinate data required for testing.
Then, input the sampled coordinate data into the model to obtain the prediction result of the model, and finally compare the prediction result with the FDM result to obtain the error of the model.

``` py linenums="203"
--8<--
examples/heat_pinn/heat_pinn.py:203:212
--8<--
```

## 4. Complete Code

``` py linenums="1" title="heat_pinn.py"
--8<--
examples/heat_pinn/heat_pinn.py
--8<--
```

## 5. Result Display

<figure markdown>
  ![T_comparison](https://paddle-org.bj.bcebos.com/paddlescience/docs/Heat_PINN/pinn_fdm_comparison.png.PNG){ loading=lazy }
  <figcaption>Top: PINN calculation result, Bottom: FDM calculation result
</figure>

The figure above shows the temperature distribution maps calculated using PINN and FDM methods respectively. It can be seen that the results between them are very close. In addition, the mean square error (MSE Loss) between PINN and FDM is only 0.0013. Considering both graphical and numerical results, it can be concluded that PINN can effectively solve the heat transfer problem in this case.

<figure markdown>
  ![profile](https://paddle-org.bj.bcebos.com/paddlescience/docs/Heat_PINN/profiles.PNG){ loading=lazy }
  <figcaption>Top: Comparison of T results between PINN and FDM in x direction, Bottom: Comparison of T results between PINN and FDM in y direction
</figure>

The figure above shows the cross-sectional plot ($y=\{-0.75,-0.50,-0.25,0.00,0.25,0.50,0.75\}$) and longitudinal cross-sectional plot ($x=\{-0.75,-0.50,-0.25,0.00,0.25,0.50,0.75\}$) of temperature $T$. It can be seen that the calculation results of PINN and FDM methods are basically consistent.

## 6. References

- [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)
- [Heat-PINN](https://github.com/314arhaam/heat-pinn)
