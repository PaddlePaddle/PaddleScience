# NTopo: Mesh-free Topology Optimization using Implicit Neural Representations

=== "Model Training Command"

    ``` sh
    python ntopo.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python ntopo.py mode=eval PROBLEM=Beam2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/beam2d_pretrained.pdparams
    python ntopo.py mode=eval PROBLEM=Bridge2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/bridge2d_pretrained.pdparams
    python ntopo.py mode=eval PROBLEM=Distributed2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/distributed2d_pretrained.pdparams
    python ntopo.py mode=eval PROBLEM=LongBeam2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/longbeam2d_pretrained.pdparams
    python ntopo.py mode=eval PROBLEM=LShape2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/lshape2d_pretrained.pdparams
    python ntopo.py mode=eval PROBLEM=Triangle2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/triangle2d_pretrained.pdparams
    python ntopo.py mode=eval PROBLEM=TriangleVariants2D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/trianglevariants2d_pretrained.pdparams
    python ntopo.py --config-name ntopo.yaml mode=eval PROBLEM=Beam3D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/beam3d_pretrained.pdparams
    python ntopo.py --config-name ntopo.yaml mode=eval PROBLEM=Bridge3D EVAL.pretrained_model_path_density=https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/bridge3d_pretrained.pdparams
    ```

*Note: Since the training method of this case is special and there is no reference metric, the visual results are generated directly after training to judge the training effect.*

## 1. Background Introduction

In topology optimization problems, there is a common method called SIMP (Solid Isotropic Material with Penalization), which is a topology optimization method based on the density method. It describes the material distribution at each point in the design domain through continuous design variables (material density), and finally approximates the ideal "0-1" binary distribution (material presence or absence). Its core goal is to optimize the material layout to achieve specific performance goals (such as minimum compliance) under constraints (such as volume, stiffness).

In traditional numerical calculations, SIMP discretizes the design domain into finite element meshes, assigns a continuous density variable $\rho \in [0,1]$ to each element, and then introduces a power law interpolation function and a penalty factor to penalize intermediate density values towards 0 or 1 through mathematical formulas to suppress gray areas. For example, the elastic modulus interpolation formula is: $E(\rho) = \rho^p E_1$

This case proposes a new machine learning method based on implicit neural representation to solve the difficult inverse problem of topology optimization. Traditional methods rely on meshing, while this case parameterizes the density field and displacement field mesh-free through MLP, using the continuous differentiability of neural networks to generate high-detail solutions. Experiments show that this method performs well in structural compliance objective optimization, and can explore the continuous solution space of topology optimization problems through self-supervised learning, overcoming the limitations of traditional methods in high-dimensional parameter spaces and nonlinear objective functions. The core innovation lies in combining neural representation with mesh-free optimization, providing an efficient and flexible solution for complex inverse problems.

## 2. Problem Definition

The goal of Topology Optimization (TO) is to find the material distribution that makes the structure most rigid under given boundary conditions, forces, and target material volume fraction. This problem can be formalized as a constrained bilevel minimization problem:

$$
\begin{cases}
  \min_{\rho}L_{comp}(\rho)=\int_{\Omega}e(\rho,u(\rho),\omega)d\omega \\
  s.t. \quad u(\rho)=\arg\min_{u} L_{sim}(u,\rho) \\
  \rho(\omega) \in {0, 1}, \quad \frac{1}{|\Omega|} \int_{\Omega} \rho d\omega = \hat{V} \\
\end{cases}
$$

Where $L_{comp}(\rho)$ is the compliance loss function; $e(\rho, u(\rho), \omega)$ is the point compliance, proportional to the internal energy; $u(\rho)$ is the displacement field, satisfying the force balance condition, obtained by minimizing the simulation loss $L_{sim}(u, \rho)$; $\rho(\omega)$ is the material density field, theoretically taking values of $0$ or $1$ (indicating absence or presence of material), but continuous values are allowed in actual optimization, and convergence to binary solutions is encouraged; $\Omega$ is the design domain, $\omega$ is the spatial coordinate; $\hat{V}$ is the target material volume fraction.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

<figure markdown>
  ![pipeline](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/train.png){ loading=lazy style="margin:0 auto"}
  <figcaption> Overall Training Process </figcaption>
</figure>

The above figure shows the overall training process. By alternately training two neural networks, the displacement network and the density network, mapping the spatial coordinate $ω$ to the equilibrium displacement $u$ and the optimal density $\rho$ respectively, the optimal material distribution is calculated. In each iteration, the displacement network is first updated by minimizing the total potential energy of the system, followed by sensitivity analysis to calculate the density space gradient, and generating the target density field $\hat{\rho}$ through sensitivity filtering. Finally, the density network is updated by minimizing the convex optimization objective function based on the mean square error between the current density and the target density.

Both the displacement network and the density network are MLP networks using the SIREN activation function. For specific code, please refer to the model.py file in [Complete Code](#4).

### 3.2 Parameter and Hyperparameter Setting

We need to specify problem-related parameters, such as the name of the problem to be optimized (geometry type), material parameters, optimization target (volume percentage), etc.:

``` yaml linenums="34"
--8<--
examples/ntopo/conf/ntopo_2d.yaml:34:50
--8<--
```

In addition, parameters required for other training such as training rounds and `batch_size` need to be specified in the configuration file. Note that the separate `epochs` parameter for the two networks needs to be set to $1$:

``` yaml linenums="69"
--8<--
examples/ntopo/conf/ntopo_2d.yaml:69:101
--8<--
```

It is particularly important to note that some techniques are used in this case, such as Moving Mean Square Error (MMSE), Optimality Criteria method based on multiple batches (OC), and filtering. Their related parameters and the `iters_per_epoch` of training need to be set carefully. It is not that the larger a certain parameter is, the better. Different parameter settings may lead to different optimization results.

### 3.3 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `Adam` optimizer is selected.

``` py linenums="44"
--8<--
examples/ntopo/ntopo.py:44:50
--8<--
```

### 3.4 Equation Construction

As described in [Problem Definition](#2), formulas such as the elastic modulus interpolation formula are needed during model training, so equations need to be defined. For specific code, please refer to the equation.py file in [Complete Code](#4).

### 3.5 Problem Construction (including loss)

The computational domain of this problem is the initial geometric structure. This case provides classes for some 2D and 3D problems, which contain definitions of various parameters and conditions such as computational domain, boundary conditions, and force conditions. For specific code, please refer to the problems.py file in [Complete Code](#4).

It is worth noting that some techniques are used in this case, such as the Optimality Criteria method based on multiple batches (OC). This method requires a batch of input and output data, and then calculates the loss of this batch in some way.

### 3.6 Constraint Construction

There are 1 type of interior point constraint and 1 type of supervised constraint in the code of this case (but actually the label is not used when calculating loss. Since the API is called, it is introduced here according to the constraint method).

#### 3.6.1 Interior Point Constraint

There is constraint `InteriorConstraint` for points inside the geometry:

``` py linenums="74"
--8<--
examples/ntopo/ntopo.py:74:86
--8<--
```

The first parameter of `InteriorConstraint` is the equation (system) expression, used to describe how to calculate the constraint target. Here, fill in `problem.equation["EEquation"].equations` instantiated in the [3.4 Equation Construction](#34) chapter;

The second parameter is the target value of the constraint variable. In this problem, it is hoped that the $E$ value `E_xyz` or `E_xy` related to the equation is optimized to 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in the computational domain `problem.geom["geo"]` of the corresponding problem instantiated in the [3.5 Problem Construction](#35-loss) chapter;

The fourth parameter is the sampling configuration on the computational domain.

The fifth parameter is the loss function. Here, the custom loss function `problem.disp_loss_func` is passed in through `ppsci.loss.FunctionalLoss`;

The sixth is the name of the constraint condition. Each constraint condition needs to be named for subsequent indexing. Here it is named "INTERIOR_DISP".

#### 3.6.2 Supervised Constraint

Since a custom sampling method is defined in this case, the supervised constraint `SupervisedConstraint` is called here, and the sampling points are passed to it in the form of input:

``` py linenums="95"
--8<--
examples/ntopo/ntopo.py:95:115
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of the supervised constraint, where the `dataset` field represents the training dataset information used, and each field represents:

1. `name`: Dataset type, here `NamedArrayDataset` means dataset read from Array;
2. `input`: Input data of Array type;

Note that there is no label value `label`.

The `sampler` field represents the sampling method, where each field represents:

1. `name`: Sampler type, here `BatchSampler` means batch sampler;
2. `drop_last`: Whether to discard samples that cannot make up a full mini-batch at the end, set to False;
3. `shuffle`: Whether to shuffle the order when generating sample indices, set to True;

The `num_workers` field represents the number of threads when loading input;

The `batch_size` field represents the size of the batch;

The second parameter is the loss function. Here a loss function class is customized to receive the special batch loss function `problem.density_loss_func` of this case;

The third parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here it is named "INTERIOR_DENSITY".

### 3.7 Visualizer Construction

This case saves the optimization results as vtu files through the visualizer `ppsci.visualize.VisualizerVtu` at certain training intervals:

``` py linenums="130"
--8<--
examples/ntopo/ntopo.py:130:146
--8<--
```

### 3.8 Other Functions

As mentioned above, two models need to be trained alternately in this case, and some techniques are added, such as Moving Mean Square Error (MMSE), Optimality Criteria method based on multiple batches (OC), and filtering. Therefore, the training process of this case is quite different from single model training.

Therefore, in this case, based on the PaddleScience code, the following are customized:

1. `Trainer` class, which defines a new training process based on the information in the received `solver`;
2. `FunctionalLossBatch` class, which is based on `ppsci.loss.base.Loss`, redefines the loss processing method, and is called in `Trainer`;
3. `Sampler` class, which defines the sampling method required by the case;
3. `Plot` class. For geometries with symmetrical shapes, this case chooses to define only half of the symmetrical part, and then restore the complete result according to the `mirror` parameter in the problem. This class provides related processing functions;

For specific code, please refer to the functions.py file in [Complete Code](#4).

### 3.9 Model Training and Evaluation

After completing the above settings, pass the instantiated objects to `ppsci.solver.Solver` in order, and then train according to the customized training process. For specific code, please refer to the ntopo.py file in [Complete Code](#4).

Since topology optimization problems have no labels, multiple optimization results may be effective, so the training results need to be manually evaluated based on visual results.

## 4. Complete Code

``` py linenums="1" title="ntopo.py"
--8<--
examples/ntopo/ntopo.py
--8<--
```

``` py linenums="1" title="model.py"
--8<--
examples/ntopo/model.py
--8<--
```

``` py linenums="1" title="equation.py"
--8<--
examples/ntopo/equation.py
--8<--
```

``` py linenums="1" title="problems.py"
--8<--
examples/ntopo/problems.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/ntopo/functions.py
--8<--
```

## 5. Result Display

The optimization results on different problems are shown below.

| No. | Problem Name | Pretrained Model | Result |
| :-- | :-- | :-- | :-- |
| 1 | Beam2D | [beam2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/beam2d_pretrained.pdparams) | ![beam2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/beam2d.png) |
| 2 | Bridge2D |[bridge2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/bridge2d_pretrained.pdparams) | ![bridge2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/bridge2d.png)  |
| 3 | Distributed2D | [distributed2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/distributed2d_pretrained.pdparams) | ![distributed2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/distributed2d.png) |
| 4 | LongBeam2D |[longbeam2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/longbeam2d_pretrained.pdparams) | ![longbeam2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/longbeam2d.png)  |
| 5 | LShape2D | [lshape2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/lshape2d_pretrained.pdparams) | ![lshape2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/Lshape2d.png) |
| 6 | Triangle2D |[triangle2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/triangle2d_pretrained.pdparams) | ![triangle2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/triangle2d.png)  |
| 7 | TriangleVariants2D | [trianglevariants2d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/trianglevariants2d_pretrained.pdparams) | ![trianglevariants2d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/trianglevariants2d.png) |
| 8 | Beam3D |[beam3d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/beam3d_pretrained.pdparams) | ![beam3d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/beam3d.png)  |
| 9 | Bridge3D |[bridge3d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ntopo/bridge3d_pretrained.pdparams) | ![bridge3d](https://paddle-org.bj.bcebos.com/paddlescience/docs/ntopo/bridge3d.png)  |

## 6. References

- [NTopo: Mesh-free Topology Optimization using Implicit Neural Representations](https://arxiv.org/abs/2102.10782)

- [Reference Code](https://github.com/JonasZehn/ntopo)
