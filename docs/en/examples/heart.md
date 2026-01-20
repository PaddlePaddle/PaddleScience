# Heart

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

=== "Model Training Command"

    === "Forward Problem: Stress Analysis"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python forward.py
        ```

    === "Inverse Problem: Parameter Inversion"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python inverse.py TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams
        ```

=== "Model Evaluation Command"

    === "Forward Problem: Stress Analysis"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python forward.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/forward_pretrained.pdparams
        ```

    === "Inverse Problem: Parameter Inversion"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
        tar -xvf heart_dataset.tar
        python inverse.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams EVAL.param_E_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/param_E.pdparams
        ```

| Pretrained Model | Metrics |
|:--| :--|
| [forward_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heart/forward_pretrained.pdparams) | loss(ref_u_v_w): 0.00076<br>L2Rel.u(ref_u_v_w): 0.01162<br>L2Rel.v(ref_u_v_w): 0.00511<br>L2Rel.w(ref_u_v_w): 0.00737 |
| [inverse_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams)<br>[param_E.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/heart/param_E.pdparams) | loss(ref_u_v_w): 0.00576<br>L2Rel.u(ref_u_v_w): 0.03082<br>L2Rel.v(ref_u_v_w): 0.01412<br>L2Rel.w(ref_u_v_w): 0.02075<br>L2_Error(E): 0.04975 |

## 1. Background Introduction

Cardiovascular disease has become the number one killer threatening human health. Biomechanical modeling based on individual images has played an important role in understanding heart disease and developing new diagnosis and treatment plans. However, in the field of cardiac biomechanical modeling, traditional finite element methods have problems such as tedious meshing and slow solution speed, which limits their application in personalized heart modeling and diagnosis and treatment. Physics Informed Neural Network (PINN) is an algorithm for solving partial differential equations based on neural networks that has emerged in recent years. It has achieved certain results in the field of fluid mechanics and has attracted the attention of a large number of researchers. It has broad application prospects. Solving cardiac biomechanical equations and simulating calculations through PINN can greatly improve the efficiency of individual heart modeling.

This case uses the PINN network. On the left ventricle model of an individualized heart, according to the theory of elasticity, the linear elastic constitutive relationship, geometric equation, equilibrium equation and boundary conditions satisfied by the displacement field of the heart model are given, and the true displacement of the finite element simulation results under the same conditions is used as a constraint to train a PINN network that solves two material parameters in the linear elastic Hooke's law of the left ventricle.

## 2. Problem Definition

The model input is the coordinates $(x,y,z)$ of the mesh points before the left ventricle deformation, and the output is the displacement $(u,v,w)$ corresponding to the mesh points after the left ventricle diastolic deformation.

In this case, the heart is considered to be a linear elastic material, satisfying Hooke's law for linear elastic materials, i.e.:

$$
\begin{pmatrix}
    t_{xx} \\ t_{yy} \\ t_{zz} \\ t_{xy} \\ t_{xz} \\ t_{yz} \\
\end{pmatrix}
=
\begin{bmatrix}
    \frac{1}{E} & -\frac{\nu}{E} & -\frac{\nu}{E} & 0 & 0 & 0 \\
    -\frac{\nu}{E} & \frac{1}{E} & -\frac{\nu}{E} & 0 & 0 & 0 \\
    -\frac{\nu}{E} & -\frac{\nu}{E} & \frac{1}{E} & 0 & 0 & 0 \\
    0 & 0 & 0 & \frac{1}{G} & 0 & 0 \\
    0 & 0 & 0 & 0 & \frac{1}{G} & 0 \\
    0 & 0 & 0 & 0 & 0 & \frac{1}{G} \\
\end{bmatrix}
\begin{pmatrix}
    \varepsilon _{xx} \\ \varepsilon _{yy} \\ \varepsilon _{zz} \\ \varepsilon _{xy} \\ \varepsilon _{xz} \\ \varepsilon _{yz} \\
\end{pmatrix}
$$

Where $G=\frac{E}{2(1+\nu)}$, $E=9kpa$ and $\nu=0.45$ are two independent constants, $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$, $\sigma_{xy}$, $\sigma_{xz}$, $\sigma_{yz}$ are the stresses in the three dimensions of the corresponding coordinate point, and its relationship with displacement is:

$$
\begin{pmatrix}
    \sigma_{xx} = \frac{\partial u}{\partial x} \\
    \sigma_{yy} = \frac{\partial v}{\partial y} \\
    \sigma_{zz} = \frac{\partial w}{\partial z} \\
    \sigma_{xy} = \frac{1}{2}(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}) \\
    \sigma_{xz} = \frac{1}{2}(\frac{\partial u}{\partial z}+\frac{\partial w}{\partial x}) \\
    \sigma_{yz} = \frac{1}{2}(\frac{\partial v}{\partial z}+\frac{\partial w}{\partial y}) \\
\end{pmatrix}
$$

In this case, it is considered that the passive mechanics of the left ventricle in diastole are quasi-static, so it is considered that this case has the following boundary conditions:

1. In the entire geometric computational domain, it is necessary to satisfy $\nabla t=0$, i.e.:

$$
\begin{pmatrix}
    \frac{\partial t_{xx}}{\partial x}+\frac{\partial t_{xy}}{\partial y}+\frac{\partial t_{xz}}{\partial z}=0 \\
    \frac{\partial t_{xy}}{\partial x}+\frac{\partial t_{yy}}{\partial y}+\frac{\partial t_{yz}}{\partial z}=0 \\
    \frac{\partial t_{xz}}{\partial x}+\frac{\partial t_{yz}}{\partial y}+\frac{\partial t_{zz}}{\partial z}=0 \\
\end{pmatrix}
$$

2. On the endocardial surface, it is necessary to satisfy $tn=-P_{endo}n$, where $n$ is the unit normal direction of the endocardial surface, and $P_{endo}=1.064kpa(8mmHg)$ represents the left ventricular cavity pressure;

3. On the epicardium, it is necessary to satisfy $P_{epi}=0$;

4. On the basal plane, it is necessary to satisfy $u_{x,y,z}=0$, i.e., $u,v,w=0$

<figure markdown>
  ![boundary conditions](https://paddle-org.bj.bcebos.com/paddlescience/docs/heart/doc1.png){ loading=lazy }
  <figcaption>Schematic diagram of boundary conditions</figcaption>
</figure>

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Stress Analysis Solution

#### 3.1.1 Model Construction

As mentioned above, each known coordinate point $(x, y, z)$ has a corresponding strain $(u, v, w)$ to be solved. Use a model to predict:

$u, v, w = f(x,y,z)$

Expressed in PaddleScience code as follows:

``` py linenums="24"
--8<--
examples/heart/forward.py:24:26
--8<--
```

#### 3.1.2 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and combined with the ExponentialDecay learning rate adjustment strategy commonly used in machine learning.

``` py linenums="27"
--8<--
examples/heart/forward.py:27:32
--8<--
```

#### 3.1.3 Equation Construction

Construct equations in the equation.py file. The corresponding equation instantiation code is as follows:

``` py linenums="33"
--8<--
examples/heart/forward.py:33:35
--8<--
```

#### 3.1.4 Computational Domain Construction

The geometric area of this problem is specified by the stl file. Download and extract it to the `./stl/` folder according to the "Model Training Command" at the beginning of this document.

???+ warning "Note"

    **Before using the `Mesh` class, you must first install the open3d, pysdf, and PyMesh 3 geometric dependency packages according to the [1.4.2 Install Mesh Geometry [Optional]](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/install_setup/#142-mesh) document.**

Then, through the STL geometry class `ppsci.geometry.Mesh` built into PaddleScience, you can read and parse the geometry file, obtain the computational domain, and obtain the geometric structure boundary:

``` py linenums="36"
--8<--
examples/heart/forward.py:36:44
--8<--
```

#### 3.1.5 Hyperparameter Setting

Next, you need to specify the number of training epochs in the configuration file. Here, based on experimental experience, 200 training epochs are used, with 1000 optimization steps per epoch.

``` yaml linenums="61"
--8<--
examples/heart/conf/forward.yaml:61:62
--8<--
```

#### 3.1.6 Constraint Construction

This problem involves 4 constraints in [2. Problem Definition](#2). Before constructing specific constraints, you can construct data reading configurations so that this configuration can be reused when constructing multiple constraints later.

``` py linenums="45"
--8<--
examples/heart/forward.py:45:56
--8<--
```

##### 3.1.6.1 Interior Point Constraint

Take `InteriorConstraint` acting on structural interior points as an example, the code is as follows:

``` py linenums="85"
--8<--
examples/heart/forward.py:85:102
--8<--
```

The first parameter of `InteriorConstraint` is the equation (system) expression, used to describe how to calculate the constraint target. Here, fill in `equation["Hooke"].equations` instantiated in the [3.1.3 Equation Construction](#313) section;

The second parameter is the target value of the constraint variable. In this problem, `hooke_x`, `hooke_y`, `hooke_z` are optimized to 0;

The third parameter is the computational domain where the constraint equation acts. Here, fill in `geom["geo"]` instantiated in the [3.1.4 Computational Domain Construction](#314) section;

The fourth parameter is the sampling configuration on the computational domain. Here, set `batch_size` as:

``` yaml linenums="70"
--8<--
examples/heart/conf/forward.yaml:70:74
--8<--
```

The fifth parameter is the loss function. Here, the commonly used MSE function is selected, and `reduction` is set to `"mean"`, which means that the loss terms generated by all data points involved in the calculation will be averaged;

The sixth parameter is geometric point filtering. It is necessary to filter the points sampled on geo. Here, pass in a lambda filtering function, which accepts the tensor `x, y, z` composed of point sets and returns a boolean tensor indicating whether each point meets the filtering conditions. If not, it is `False`, and if it meets, it is `True`. Because the structure of this case comes from the network and the parameters are not completely accurate, `1e-1` is added as a tolerable sampling error;

The seventh parameter is the weight of each point when calculating the loss;

The eighth parameter is the name of the constraint condition. Each constraint condition needs to be named for subsequent indexing. Here it is named "INTERIOR".

##### 3.1.6.2 Boundary Constraint

Refer to [2. Problem Definition](#2) for constraints on the endocardium, epicardium, and basal plane respectively:

``` py linenums="57"
--8<--
examples/heart/forward.py:57:84
--8<--
```

#### 3.1.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval. The data of the validation set comes from an external csv file, so first use the `ppsci.utils.reader` module to read the validation point set from the csv file:

``` py linenums="125"
--8<--
examples/heart/forward.py:125:137
--8<--
```

Then convert it to a dictionary and perform dimensionless and normalization, and then wrap it into a dictionary and pass it to `ppsci.validate.SupervisedValidator` together with `eval_dataloader_cfg` (validation set dataloader configuration, constructed similarly to `train_dataloader_cfg`) to construct the validator.

``` py linenums="138"
--8<--
examples/heart/forward.py:138:168
--8<--
```

#### 3.1.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, you can choose a suitable visualizer to visualize the output result.

The input data in this article is the input dictionary `input_dict` prepared in the validator construction, and the output data is the corresponding 3 predicted physical quantities. Therefore, you only need to save the evaluation output data as a **vtu format** file, and finally open it with visualization software to view it. The code is as follows:

``` py linenums="170"
--8<--
examples/heart/forward.py:170:182
--8<--
```

#### 3.1.8 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training:

``` py linenums="184"
--8<--
examples/heart/forward.py:184:200
--8<--
```

Calling `ppsci.solver.Solver.plot_loss_history` after training can plot the `loss` during training:

``` py linenums="201"
--8<--
examples/heart/forward.py:201:203
--8<--
```

#### 3.1.9 Model Evaluation and Visualization

After training is completed or the pre-trained model is downloaded, model evaluation and visualization are performed through the "Model Evaluation Command" at the beginning of this document.

The evaluation and visualization process does not require optimizer construction, etc., only need to construct the model, computational domain, validator (not included in this case), visualizer, and then pass them to `ppsci.solver.Solver` in order to start evaluation and visualization:

``` py linenums="269"
--8<--
examples/heart/forward.py:269:275
--8<--
```

### 3.2 Parameter Inversion Solution

#### 3.2.1 Equation Construction

This case attempts to model and implement the complex hyperelastic constitutive relationship of heart soft tissue under the PINN framework. In the case where the preset equation parameter $E$ is unknown, it attempts to train to obtain the value of the unknown parameter through partial data. The case still uses the equation constructed in the equation.py file, but the unknown parameter needs to be set as a learnable variable and passed to the equation:

``` py linenums="31"
--8<--
examples/heart/inverse.py:31:38
--8<--
```

#### 3.2.2 Model Construction

The model settings are the same as the forward problem:

``` py linenums="39"
--8<--
examples/heart/inverse.py:39:41
--8<--
```

#### 3.2.3 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and combined with the `ExponentialDecay` learning rate adjustment strategy commonly used in machine learning. When setting the optimizer, the learnable parameters in the equation need to be passed to the optimizer so that the parameters participate in optimization:

``` py linenums="42"
--8<--
examples/heart/inverse.py:42:47
--8<--
```

#### 3.2.4 Other Settings

Other settings for this problem are similar to the forward problem and will not be repeated here.

## 4. Complete Code

``` py linenums="1" title="forward.py"
--8<--
examples/heart/forward.py
--8<--
```

``` py linenums="1" title="inverse.py"
--8<--
examples/heart/inverse.py
--8<--
```

``` py linenums="1" title="equation.py"
--8<--
examples/heart/equation.py
--8<--
```

## 5. Result Display

### 5.1 Stress Analysis Solution

The figure below shows the model prediction results of the strain $u, v, w$ in 3 directions when the force direction is the positive x direction. The results are basically consistent with cognition.

<figure markdown>
  ![forward_result.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/heart/doc2.jpg){ loading=lazy }
  <figcaption>Left is predicted structural strain u; Middle is predicted structural strain v; Right is predicted structural strain w</figcaption>
</figure>

### 5.2 Parameter Inversion Solution

The table below shows the model prediction results of the learnable equation parameter $E$, with an error of about 5%.

| data | E |
| :---: | :---: |
| outs | 9 |
| label | 9.44778 |
