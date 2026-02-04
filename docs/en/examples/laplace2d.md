# 2D-Laplace

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6169897?sUid=455441&shared=1&ts=1684122038217" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    python laplace2d.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python laplace2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/laplace2d/laplace2d_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python laplace2d.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python laplace2d.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [laplace2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/laplace2d/laplace2d_pretrained.pdparams) | loss(MSE_Metric): 0.00002<br>MSE.u(MSE_Metric): 0.00002 |

## 1. Background Introduction

The Laplace equation, named after the French mathematician Pierre-Simon Laplace, is a pivotal partial differential equation in fields such as electromagnetism, astronomy, and fluid mechanics. While analytical solutions exist for simple geometries, complex practical problems typically require numerical methods like the Finite Element Method (FEM) or Finite Difference Method (FDM).

In this example, we demonstrate how to solve the 2D Laplace equation using deep learning techniques, specifically Physics-Informed Neural Networks (PINNs).

## 2. Problem Definition

The 2D Laplace equation is defined as:

$$
\dfrac{\partial^{2} u}{\partial x^{2}} + \dfrac{\partial^{2} u}{\partial y^{2}} = 0, \quad (x, y) \in (0, 1) \times (0, 1)
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

We aim to determine the scalar field $u(x, y)$ for any coordinate $(x, y)$ in the domain. We approximate this function using a Multilayer Perceptron (MLP) to learn the mapping $f: \mathbb{R}^2 \to \mathbb{R}^1$:

$$
u = f(x, y)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows

``` py linenums="23"
--8<--
examples/laplace/laplace2d.py:23:24
--8<--
```

We define the model input keys as `("x", "y")` and the output key as `("u",)`, ensuring consistency with the code.

The MLP is instantiated with 5 hidden layers, each containing 20 neurons.

### 3.2 Equation Construction

Since we are solving the 2D Laplace equation, we can directly use the built-in `Laplace` class in PaddleScience, setting `dim=2`.

``` py linenums="26"
--8<--
examples/laplace/laplace2d.py:26:27
--8<--
```

### 3.3 Computational Domain Construction

The problem domain is a unit square defined by diagonal corners (0.0, 0.0) and (1.0, 1.0). We use the built-in `Rectangle` geometry to define this domain.

``` py linenums="29"
--8<--
examples/laplace/laplace2d.py:29:34
--8<--
```

### 3.4 Constraint Construction

In this case, we use two constraints to guide the training of the model in the computational domain, namely the Laplace equation constraint acting on the sampling points and the constraint acting on the boundary points.

Before defining the constraints, it is necessary to specify the number of sampling points for each constraint, indicating the number of sampling data for each constraint in its corresponding computational domain, as well as the general sampling configuration.

``` yaml linenums="30"
--8<--
examples/laplace/conf/laplace2d.yaml:30:31
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on interior points as an example, the code is as follows:

``` py linenums="50"
--8<--
examples/laplace/laplace2d.py:50:59
--8<--
```

- **Equation**: `equation["laplace"].equations` (the residual of the Laplace equation).
- **Target**: 0 (minimizing the residual to zero).
- **Domain**: `geom["rect"]` (the rectangular domain).
- **Sampling**: Full batch training with `batch_size=10201` (representing a 101x101 grid).
- **Loss**: MSE with `reduction="sum"` to sum the loss across all points.
- **Equidistant**: Enabled to ensure uniform sampling for better convergence.
- **Name**: "EQ".

#### 3.4.2 Boundary Constraint

Similarly, we also need to construct constraints for the four boundaries of the rectangle. But unlike constructing `InteriorConstraint`, since the active area is the boundary, we use the `BoundaryConstraint` class, the code is as follows:

``` py linenums="60"
--8<--
examples/laplace/laplace2d.py:60:72
--8<--
```

- **Constraint Object**: `out["u"]` (the model output).
- **Target Value**: Calculated directly from the analytical solution:

``` py linenums="36"
--8<--
examples/laplace/laplace2d.py:36:40
--8<--
```

Other parameters for `BoundaryConstraint` follow the same logic as `InteriorConstraint`.

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs in the configuration file. Here, based on experimental experience, we use 20,000 training epochs, and the evaluation interval is 200 epochs.

``` yaml linenums="45"
--8<--
examples/laplace/conf/laplace2d.yaml:45:50
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the commonly used `Adam` optimizer is selected.

``` py linenums="74"
--8<--
examples/laplace/laplace2d.py:74:75
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="77"
--8<--
examples/laplace/laplace2d.py:77:92
--8<--
```

### 3.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, we can select a suitable visualizer to visualize the output result.

The output data in this article is a two-dimensional point set in an area, so we only need to save the evaluation output data as a **vtu format** file, and finally open it with visualization software to view it. The code is as follows:

``` py linenums="94"
--8<--
examples/laplace/laplace2d.py:94:103
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training, evaluation, and visualization.

``` py linenums="105"
--8<--
examples/laplace/laplace2d.py:105:125
--8<--
```

## 4. Complete Code

``` py linenums="1" title="laplace2d.py"
--8<--
examples/laplace/laplace2d.py
--8<--
```

## 5. Result Display

We use the trained model to predict values at `NPOINT_TOTAL` uniformly sampled points $(x_i, y_i)$. The figure below displays the predicted solution $u(x, y)$ across the domain.

<figure markdown>
  ![laplace 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/laplace2d/laplace2d.png){ loading=lazy }
  <figcaption>Model prediction result</figcaption>
</figure>
