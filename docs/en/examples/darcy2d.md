# 2D-Darcy

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6184070?contributionType=1&sUid=438690&shared=1&ts=1684239806160" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    python darcy2d.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python darcy2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/darcy2d/darcy2d_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python darcy2d.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python darcy2d.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [darcy2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/darcy2d/darcy2d_pretrained.pdparams) | loss(Residual): 0.36500<br>MSE.poisson(Residual): 0.00006 |

## 1. Background Introduction

Darcy Flow is a tool based on Darcy's Law for calculating fluid flow. Darcy Flow is widely used in fields such as groundwater modeling, hydrology, hydrogeology, and petroleum engineering.

For example, in petroleum engineering, Darcy Flow is used to predict and simulate the flow of oil in porous media. Porous media is a substance composed of small particles with voids between them. Oil fills these voids and flows through them. With Darcy Flow, engineers can predict and control the flow of oil, thereby optimizing oil extraction and production processes.

In addition, Darcy Flow is also used to study and predict groundwater flow. For example, in agriculture, simulating groundwater flow can predict the impact of irrigation on soil moisture, thereby optimizing crop irrigation plans. in urban planning and environmental protection, Darcy Flow is also used to predict and prevent groundwater pollution.

2D-Darcy is a type of Darcy flow. When fluid flows in porous media, the seepage velocity is small, the flow obeys Darcy's law, and there is a linear relationship between seepage velocity and pressure gradient. This flow is called linear seepage.

## 2. Problem Definition

Assume that in the Darcy flow model, the flow velocity $\mathbf{u}$ and pressure $p$ at each position $(x,y)$ satisfy the following relationship:

$$
\begin{cases}
\begin{aligned}
  \mathbf{u}+\nabla p =& 0,(x,y) \in \Omega \\
  \nabla \cdot \mathbf{u} =& f,(x,y) \in \Omega \\
  p(x,y) =& \sin(2 \pi x )\cos(2 \pi y), (x,y) \in \partial \Omega
\end{aligned}
\end{cases}
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the darcy-2d problem, each known coordinate point $(x, y)$ has a corresponding unknown quantity $p$ to be solved. We use a relatively simple MLP (Multilayer Perceptron) here to represent the mapping function $f: \mathbb{R}^2 \to \mathbb{R}^1$ from $(x, y)$ to $p$, i.e.:

$$
p = f(x, y)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows:

``` py linenums="33"
--8<--
examples/darcy/darcy2d.py:33:34
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, we specify the input variable name of the network model as `("x", "y")` and the output variable name as `"p"`, these names are consistent with the subsequent code.

Then by specifying the number of layers and neurons of MLP, we instantiated a neural network model `model` with 5 hidden layers and 20 neurons per layer.

### 3.2 Equation Construction

Since 2D-Poisson uses the 2D form of the Poisson equation, the `Poisson` built in PaddleScience can be used directly, specifying the parameter `dim` of this class as 2.

``` py linenums="36"
--8<--
examples/darcy/darcy2d.py:36:37
--8<--
```

### 3.3 Computational Domain Construction

In this article, the 2D darcy problem acts on a two-dimensional rectangular area with (0.0, 0.0), (1.0, 1.0) as diagonals, so the spatial geometry `Rectangle` built in PaddleScience can be used directly as the computational domain.

``` py linenums="39"
--8<--
examples/darcy/darcy2d.py:39:40
--8<--
```

### 3.4 Constraint Construction

In this case, we use two constraints to guide the training of the model in the computational domain, namely the darcy equation constraint acting on the sampling points and the constraint acting on the boundary points.

Before defining constraints, you need to specify the number of sampling points for each constraint, indicating the number of sampled data for each constraint in its corresponding computational domain, as well as general sampling configuration.

``` py linenums="42"
--8<--
examples/darcy/darcy2d.py:42:46
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on internal points as an example, the code is as follows:

``` py linenums="48"
--8<--
examples/darcy/darcy2d.py:48:65
--8<--
```

The first parameter of `InteriorConstraint` is the equation expression, used to describe how to calculate the constraint target. Here, fill in `equation["Poisson"].equations` instantiated in the [3.2 Equation Construction](#32) chapter;

The second parameter is the target value of the constraint variable. In this problem, we hope that the result generated by the Poisson equation is optimized to be consistent with its standard solution, so set all its target values to the result generated by `poisson_ref_compute_func`;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["rect"]` instantiated in the [3.3 Computational Domain Construction](#33) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here we use full data points for training, so the `dataset` field is set to "IterableNamedArrayDataset" and `iters_per_epoch` is also set to 1, and the sampling point number `batch_size` is set to 9801 (indicating a 99x99 sampling grid);

The fifth parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"sum"`, that is, we will sum the loss terms generated by all data points involved in the calculation;

The sixth parameter is to choose whether to perform equidistant sampling on the computational domain. Here we choose to enable equidistant sampling, so that the training points can be evenly distributed on the computational domain, which is conducive to training convergence;

The seventh parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "EQ".

#### 3.4.2 Boundary Constraint

Similarly, we also need to construct constraints for the four boundaries of the rectangle. However, unlike constructing `InteriorConstraint`, since the action area is the boundary, we use the `BoundaryConstraint` class, code as follows:

``` py linenums="67"
--8<--
examples/darcy/darcy2d.py:67:77
--8<--
```

The first parameter of the `BoundaryConstraint` class indicates that we directly use the output result `out["p"]` of the network model as the constraint object during program operation;

The second parameter refers to how to obtain the true value of our constraint object. Here we calculate it directly through its analytical solution. The code for defining the analytical solution is as follows:

``` py
lambda _in: np.sin(2.0 * np.pi * _in["x"]) * np.cos(2.0 * np.pi * _in["y"])
```

The meanings of other parameters of the `BoundaryConstraint` class are basically consistent with `InteriorConstraint` and will not be introduced here.

After the differential equation constraint, boundary constraint, and initial value constraint are constructed, encapsulate them into a dictionary with the names we just named as keys for subsequent access.

``` py linenums="78"
--8<--
examples/darcy/darcy2d.py:78:82
--8<--
```

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, we use 10,000 training epochs.

``` yaml linenums="39"
--8<--
examples/darcy/conf/darcy2d.yaml:39:47
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and the OneCycle learning rate adjustment strategy commonly used in machine learning is used together.

``` py linenums="84"
--8<--
examples/darcy/darcy2d.py:84:86
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="88"
--8<--
examples/darcy/darcy2d.py:88:105
--8<--
```

### 3.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, we can choose a suitable visualizer to visualize the output result.

The output data in this article is a two-dimensional point set in an area, so we only need to save the evaluated output data as a **vtu format** file, and finally open it with visualization software to view. The code is as follows:

``` py linenums="106"
--8<--
examples/darcy/darcy2d.py:106:147
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

#### 3.9.1 Training with Adam

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="148"
--8<--
examples/darcy/darcy2d.py:148:169
--8<--
```

#### 3.9.2 Finetuning with L-BFGS [Optional]

After training with the `Adam` optimizer, we can replace the optimizer with the second-order optimizer `L-BFGS` to continue training for a small number of rounds (here we use 10% of the `Adam` optimization rounds) to further improve model accuracy.

``` py linenums="172"
--8<--
examples/darcy/darcy2d.py:172:198
--8<--

```

???+ tip "Tip"

    Using `L-BFGS` to fine-tune for a small number of rounds after training with conventional optimizers can further effectively improve model accuracy in most scenarios.

## 4. Complete Code

``` py linenums="1" title="darcy2d.py"
--8<--
examples/darcy/darcy2d.py
--8<--
```

## 5. Result Display

The following shows the prediction results, reference results, and the difference between the two for pressure $p(x,y)$, x (horizontal) direction velocity $u(x,y)$, and y (vertical) direction velocity $v(x,y)$ at each point in the square computational domain.

<figure markdown>
  ![darcy 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/Darcy2D/darcy2d_p.png){ loading=lazy }
  <figcaption>Left: Predicted pressure p, Middle: Reference pressure p, Right: Pressure difference</figcaption>
  ![darcy 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/Darcy2D/darcy2d_u_x.png){ loading=lazy }
  <figcaption>Left: Predicted x-direction velocity p, Middle: Reference x-direction velocity p, Right: x-direction velocity difference</figcaption>
  ![darcy 2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/Darcy2D/darcy2d_u_y.png){ loading=lazy }
  <figcaption>Left: Predicted y-direction velocity p, Middle: Reference y-direction velocity p, Right: y-direction velocity difference</figcaption>
</figure>
