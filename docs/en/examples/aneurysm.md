# Aneurysm

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar
    # unzip it
    tar -xvf aneurysm_dataset.tar
    python aneurysm.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar
    # unzip it
    tar -xvf aneurysm_dataset.tar
    python aneurysm.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python aneurysm.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar
    # unzip it
    tar -xvf aneurysm_dataset.tar
    python aneurysm.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [aneurysm_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams) | loss(ref_u_v_w_p): 0.01488<br>MSE.p(ref_u_v_w_p): 0.01412<br>MSE.u(ref_u_v_w_p): 0.00021<br>MSE.v(ref_u_v_w_p): 0.00024<br>MSE.w(ref_u_v_w_p): 0.00032 |

## 1. Background Introduction

Deep learning methods can be used to deal with intracranial aneurysm problems, including physics-informed deep learning methods. This method can be used for pressure modeling of intracranial aneurysms to predict and evaluate the risk of intracranial aneurysm rupture.

Targeting the following intracranial aneurysm geometric model, this case applies appropriate physical equation constraints inside and on the boundary through deep learning, and models the wall pressure in an unsupervised learning manner.

<figure markdown>
  ![equation](https://paddle-org.bj.bcebos.com/paddlescience/docs/Aneurysm/aneurysm.png){ loading=lazy style="height:80%;width:80%"}
</figure>

## 2. Problem Definition

Assume that in the intracranial aneurysm model, at the inlet part, the velocity at the center point is 1.5 and gradually decreases towards the surroundings; in the outlet area, the pressure is constantly 0; there is no slip on the boundary, and the velocity is 0; inside the blood vessel, it conforms to the motion law of N-S equation, the average flow rate in the middle section is negative (inflow), and the average flow rate in the outlet section is positive (outflow).

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the aneurysm problem, each known coordinate point $(x, y, z)$ has corresponding unknown quantities $(u, v, w, p)$ (velocity and pressure) to be solved.
Here, a relatively simple MLP (Multilayer Perceptron) is used to represent the mapping function $f: \mathbb{R}^3 \to \mathbb{R}^4$ from $(x, y, z)$ to $(u, v, w, p)$, namely:

$$
(u, v, w, p) = f(x, y, z)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows

``` py linenums="14"
--8<--
examples/aneurysm/aneurysm.py:14:15
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, the input variable name of the network model is specified as `("x", "y", "z")` and the output variable name is `("u", "v", "w", "p")`, these names are consistent with the subsequent code.

Then by specifying the number of layers and neurons of MLP, a neural network model `model` with 6 hidden layers and 512 neurons per layer is instantiated, using `silu` as the activation function, and using `WeightNorm` weight normalization.

### 3.2 Equation Construction

The intracranial aneurysm model involves 2 equations, one is the fluid N-S equation, and the other is the flow calculation equation, so PaddleScience's built-in `NavierStokes` and `NormalDotVec` can be used.

``` py linenums="17"
--8<--
examples/aneurysm/aneurysm.py:17:23
--8<--
```

### 3.3 Computational Domain Construction

The geometric area of this problem is specified by the stl file. Follow the command below to download and unzip it to the `aneurysm/` folder.

**Note: The stl file and test set data (generated using OpenFOAM) in the dataset are from [Aneurysm - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)**.

``` sh
# linux
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar

# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar -o aneurysm_dataset.tar

# unzip it
tar -xvf aneurysm_dataset.tar
```

After unzipping, the `aneurysm/stl` folder stores the stl geometric files required for computational domain construction.

???+ warning "Note"

    **Before using the `Mesh` class, you must install the three geometric dependency packages open3d, pysdf, and PyMesh according to the [1.4.2 Install Mesh Geometry [Optional]](https://paddlescience-docs.readthedocs.io/en/latest/en/install_setup/#142-mesh) document.**

Then use PaddleScience's built-in STL geometry class `Mesh` to read and parse these geometric files, and combine each computational domain through Boolean operations. The code is as follows:

``` py linenums="25"
--8<--
examples/aneurysm/aneurysm.py:25:30
--8<--
```

After that, the geometric domain can be scaled and translated to scale the coordinate range of input data and promote model training convergence.

``` py linenums="32"
--8<--
examples/aneurysm/aneurysm.py:32:44
--8<--
```

### 3.4 Constraint Construction

This case involves 6 constraints. Before constructing specific constraints, data reading configuration can be constructed first, so that this configuration can be reused when constructing multiple constraints later.

``` py linenums="46"
--8<--
examples/aneurysm/aneurysm.py:46:56
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on interior points as an example, the code is as follows:

``` py linenums="103"
--8<--
examples/aneurysm/aneurysm.py:103:110
--8<--
```

The first parameter of `InteriorConstraint` is the equation (system) expression, which is used to describe how to calculate the constraint target. Here, fill in `equation["NavierStokes"].equations` instantiated in the [3.2 Equation Construction](#32-equation-construction) chapter;

The second parameter is the target value of the constraint variable. In this problem, it is hoped that the four values related to the N-S equation `continuity`, `momentum_x`, `momentum_y`, `momentum_z` are all optimized to 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["interior_geo"]` instantiated in the [3.3 Computational Domain Construction](#33-computational-domain-construction) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here, `batch_size` is set to `6000`.

The fifth parameter is the loss function. Here, the commonly used MSE function is selected, and `reduction` is set to `"sum"`, that is, the loss terms generated by all data points involved in the calculation will be summed;

The sixth parameter is the name of the constraint condition. Each constraint condition needs to be named to facilitate subsequent indexing. Here it is named "interior".

#### 3.4.2 Boundary Constraint

Next, constraints need to be imposed on the three surfaces of **vascular inlet, outlet, and vessel wall**, including inlet velocity constraint, outlet pressure constraint, and vessel wall no-slip constraint.
In the `bc_inlet` constraint, the flow velocity at the inlet satisfies a quadratic parabolic decay from the center point to the surroundings. Here, a parabolic function is used to represent the velocity decay as it moves away from the center of the circle, and then it is used as the value of the second parameter (dictionary) of `BoundaryConstraint`.

``` py linenums="62"
--8<--
examples/aneurysm/aneurysm.py:62:86
--8<--
```

The construction methods for vessel outlet and vessel wall no-slip constraints are similar, as shown below:

``` py linenums="87"
--8<--
examples/aneurysm/aneurysm.py:87:102
--8<--
```

#### 3.4.3 Integral Boundary Constraint

For a section below the vascular inlet and the outlet area (surface), additional inflow and outflow flow constraints need to be imposed. Since flow calculation involves specific areas, discrete integration needs to be used for calculation. These processes have been built into the `IntegralConstraint` constraint condition. As shown below:

``` py linenums="111"
--8<--
examples/aneurysm/aneurysm.py:111:138
--8<--
```

Corresponding flow calculation formula:

$$
flow_i = \sum_{i=1}^{M}{s_{i} (\mathbf{u_i} \cdot \mathbf{n_i})}
$$

Where $M$ represents the number of discrete integration points, $s_i$ represents the (approximate) area of a certain point, $\mathbf{u_i}$ represents the velocity vector of a certain point, and $\mathbf{n_i}$ represents the outward normal vector of a certain point.

In addition to the common parameters described in the previous chapters, the `integral_batch_size` parameter is added here, which indicates the number of sampling points used for discrete integration. Here, 310 discrete points are used to approximate the integral calculation; at the same time, the loss function is specified as `IntegralLoss`, indicating that the final predicted value used to calculate the loss is approximated by multiple discrete points, and then the loss is calculated with the label value.

After the differential equation constraint, boundary constraint, and initial value constraint are constructed, encapsulate them into a dictionary with the names just given as keys for subsequent access.

``` py linenums="139"
--8<--
examples/aneurysm/aneurysm.py:139:147
--8<--
```

### 3.5 Hyperparameter Setting

Next, you need to specify the number of training epochs and learning rate. Here, based on experimental experience, 1500 training epochs and an initial learning rate of 0.001 are used.

``` yaml linenums="63"
--8<--
examples/aneurysm/conf/aneurysm.yaml:63:79
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and the ExponentialDecay learning rate adjustment strategy commonly used in machine learning is used together.

``` py linenums="149"
--8<--
examples/aneurysm/aneurysm.py:149:153
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="155"
--8<--
examples/aneurysm/aneurysm.py:155:219
--8<--
```

### 3.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, you can choose a suitable visualizer to visualize the output result.

The output data in this article is a set of three-dimensional points in a region, so you only need to save the evaluated output data as a **vtu format** file, and finally open it with visualization software to view it. The code is as follows:

``` py linenums="206"
--8<--
examples/aneurysm/aneurysm.py:206:219
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="221"
--8<--
examples/aneurysm/aneurysm.py:221:248
--8<--
```

## 4. Complete Code

``` py linenums="1" title="aneurysm.py"
--8<--
examples/aneurysm/aneurysm.py
--8<--
```

## 5. Result Display

For the intracranial aneurysm test set (a total of 2,962,708 three-dimensional coordinate points), the model prediction results are as follows.

<figure markdown>
  ![aneurysm_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Aneurysm/aneurysm_compare.png){ loading=lazy }
  <figcaption> Left: PaddleScience prediction result, Middle: OpenFOAM solver prediction result, Right: Difference between the two</figcaption>
</figure>

It can be seen that for the wall pressure $p(x,y,z)$, the prediction result of the model is basically consistent with the OpenFOAM result.

## 6. References

- [Aneurysm - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)
