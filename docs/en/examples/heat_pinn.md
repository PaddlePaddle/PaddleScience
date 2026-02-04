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

Heat conduction is a fundamental physical process with extensive applications in engineering and science. Accurate simulation of heat transfer is essential for optimizing energy efficiency, enhancing material properties, and designing thermal systems. The 2D steady heat conduction equation governs steady-state thermal distribution. While traditional numerical methods like Finite Element Method (FEM) or Finite Difference Method (FDM) require domain discretization and matrix solving, Physics-Informed Neural Networks (PINNs) offer a mesh-free alternative. PINNs leverage the flexibility of neural networks constrained by physical laws to solve partial differential equations directly in continuous domains.

## 2. Problem Definition

The 2D steady heat conduction problem is governed by the Laplace equation for temperature $T(x, y)$:

$$
\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}=0
$$

defined on the domain:

$$
D = \{(x, y) \mid -1 \leq x \leq 1, -1 \leq y \leq 1\}
$$

subject to the Dirichlet boundary conditions:

$$
\begin{cases}
T(-1, y) = 75.0 ^\circ\text{C}, \\
T(+1, y) = 0.0 ^\circ\text{C}, \\
T(x, -1) = 50.0 ^\circ\text{C}, \\
T(x, +1) = 0.0 ^\circ\text{C}.
\end{cases}
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

We aim to solve for the unknown temperature $T$ at each coordinate $(x, y)$. We approximate the solution using a Multilayer Perceptron (MLP) to learn the mapping $f: \mathbb{R}^2 \to \mathbb{R}^1$:

$$
u = f(x, y)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows:

``` py linenums="102"
--8<--
examples/heat_pinn/heat_pinn.py:102:103
--8<--
```

We define the model input keys as `("x", "y")` and the output key as `"u"`, ensuring consistency with the code.

The MLP is instantiated with 9 hidden layers, 20 neurons per layer, and the `tanh` activation function.

### 3.2 Equation Construction

Since the governing equation is the 2D Laplace equation, we utilize the built-in `Laplace` class in PaddleScience, setting `dim=2`.

``` py linenums="105"
--8<--
examples/heat_pinn/heat_pinn.py:105:106
--8<--
```

### 3.3 Computational Domain Construction

The problem domain is a rectangle defined by corners (-1.0, -1.0) and (1.0, 1.0). We use the built-in `Rectangle` geometry to define this domain.

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

- **Equation**: `equation["Laplace"].equations` (the residual of the Laplace equation).
- **Target**: 0 (we aim to minimize the residual to zero).
- **Domain**: `geom["rect"]` (the rectangular domain).
- **Sampling**: Full batch training with `batch_size=NPOINT_PDE` (99x99 grid).
- **Loss**: MSE with `reduction="mean"`.
- **Weight**: 1.0.
- **Equidistant**: Enabled to ensure uniform sampling for better convergence.
- **Name**: "EQ".

#### 3.4.2 Boundary Constraint

Similarly, we also need to construct constraints for the four boundaries of the rectangle. However, unlike constructing `InteriorConstraint`, since the action area is the boundary, we use the `BoundaryConstraint` class, code as follows:

``` py linenums="132"
--8<--
examples/heat_pinn/heat_pinn.py:132:171
--8<--
```

- **Constraint Object**: `out["u"]` (the model output).
- **Target Value**: The Dirichlet boundary values specified in Section 2.

Other parameters follow the same logic as `InteriorConstraint`.

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

The figure compares the temperature distributions calculated by PINN and FDM. The results are highly consistent, with an MSE loss of only 0.0013, demonstrating PINN's effectiveness in solving this heat transfer problem.

<figure markdown>
  ![profile](https://paddle-org.bj.bcebos.com/paddlescience/docs/Heat_PINN/profiles.PNG){ loading=lazy }
  <figcaption>Top: Comparison of T results between PINN and FDM in x direction, Bottom: Comparison of T results between PINN and FDM in y direction
</figure>

The plots above show cross-sectional temperature profiles at various $x$ and $y$ locations ($ \pm 0.75, \pm 0.50, \pm 0.25, 0.00 $). The PINN predictions align closely with the FDM results.

## 6. References

- [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)
- [Heat-PINN](https://github.com/314arhaam/heat-pinn)
