# Bracket

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar
    # unzip it
    tar -xvf bracket_dataset.tar
    python bracket.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar
    # unzip it
    tar -xvf bracket_dataset.tar
    python bracket.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/bracket/bracket_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python bracket.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar
    # unzip it
    tar -xvf bracket_dataset.tar
    python bracket.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [bracket_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/bracket/bracket_pretrained.pdparams) | loss(commercial_ref_u_v_w_sigmas): 32.28704<br>MSE.u(commercial_ref_u_v_w_sigmas): 0.00005<br>MSE.v(commercial_ref_u_v_w_sigmas): 0.00000<br>MSE.w(commercial_ref_u_v_w_sigmas): 0.00734<br>MSE.sigma_xx(commercial_ref_u_v_w_sigmas): 27.64751<br>MSE.sigma_yy(commercial_ref_u_v_w_sigmas): 1.23101<br>MSE.sigma_zz(commercial_ref_u_v_w_sigmas): 0.89106<br>MSE.sigma_xy(commercial_ref_u_v_w_sigmas): 0.84370<br>MSE.sigma_xz(commercial_ref_u_v_w_sigmas): 1.42126<br>MSE.sigma_yz(commercial_ref_u_v_w_sigmas): 0.24510 |

## 1. Background Introduction

The linear elasticity equation plays a central role in deformation analysis. In physics and engineering, deformation analysis is a method of studying the change in shape and size of an object under the action of external forces. The linear elasticity equation is a mathematical model that describes the ability of an object to return to its original state after being stressed. Specifically, the linear elasticity equation usually refers to the relationship between stress and strain. Stress is a physical quantity used to describe the force per unit area generated inside an object due to external forces. Strain describes the change in shape and size of an object. The linear elasticity equation can usually be expressed as a linear relationship between stress and strain, that is, stress and strain are proportional. This relationship can be expressed by a linear equation, where the coefficient is called the modulus of elasticity (or Young's modulus). This model assumes that the object can completely return to its original state after being stressed, that is, there is no permanent deformation. This assumption is reasonable in many cases, such as when studying the mechanical behavior of metals. However, for some materials (such as plastics or rubber), this assumption may be inaccurate because they may produce permanent deformation after being stressed. The linear elasticity equation is only part of deformation analysis. To fully understand deformation, other factors need to be considered, such as the initial shape and size of the object, the history of external forces, other physical properties of the material (such as thermal expansion coefficient and density), etc. However, the linear elasticity equation provides a basic framework for describing and understanding the behavior of objects after being stressed.

This case mainly studies the deformation of the following metal bracket under a given load, and uses deep learning methods to solve it based on linear elasticity and other equations. The bracket is shown below (reference [Matlab deflection-analysis-of-a-bracket](https://www.mathworks.com/help/pde/ug/deflection-analysis-of-a-bracket.html)).

<figure markdown>
  ![bracket](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/stl.png){ loading=lazy }
  <figcaption>Schematic diagram of the load on the Bracket metal part, the red area represents the load surface</figcaption>
</figure>

## 2. Problem Definition

The above connection includes a back plate perpendicular to the x-axis and a perforated flat plate connected to it perpendicular to the z-axis. The back plate is fixed, and the rightmost surface (red area) of the perforated flat plate is subjected to a stress of $4 \times 10^4 Pa$ per unit area in the negative z-axis direction; in addition, other parameters include elastic modulus $E=10^{11} Pa$, Poisson's ratio $\nu=0.3$. By setting the characteristic length $L=1m$, characteristic displacement $U=0.0001m$, and dimensionless shear modulus $0.01\mu$, the goal is to solve 9 physical quantities $u$, $v$, $w$, $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$, $\sigma_{xy}$, $\sigma_{xz}$, $\sigma_{yz}$ at each point on the surface of the metal part. The constant definition code is as follows:

``` py linenums="21"
--8<--
examples/bracket/bracket.py:21:30
--8<--
```

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.0 Dataset Description

The data used in this project includes: geometric model files (STL) and physical field evaluation data (TXT).

#### 3.0.1 Geometric Model (STL File)

The geometric area is defined by the following STL files, which are used to construct the geometric structure of the metal bracket in this case, including boundaries and internal holes:

- `./stl/support.stl`
- `./stl/bracket.stl`
- `./stl/aux_lower.stl`
- `./stl/aux_upper.stl`
- `./stl/cylinder_hole.stl`
- `./stl/cylinder_lower.stl`
- `./stl/cylinder_upper.stl`

#### 3.0.2 Physical Field Evaluation Data (TXT File)

Evaluation data, including displacement field and stress field accuracy:

- `./data/deformation_x.txt`: x-direction displacement
- `./data/deformation_y.txt`: y-direction displacement
- `./data/deformation_z.txt`: z-direction displacement
- `./data/normal_x.txt`: x-direction normal stress
- `./data/normal_y.txt`: y-direction normal stress
- `./data/normal_z.txt`: z-direction normal stress
- `./data/shear_xy.txt`: xy plane shear stress
- `./data/shear_xz.txt`: xz plane shear stress
- `./data/shear_yz.txt`: yz plane shear stress

Each line format is:

```
id    x    y    z    value
```

Where `(x, y, z)` are spatial coordinates, `value` is the true value (or high-precision numerical solution) of the corresponding physical quantity, and `id` is the sampling point index.

### 3.1 Model Construction

In the bracket problem, each known coordinate point $(x, y, z)$ has corresponding unknown quantities to be solved: strain $(u, v, w)$ and stress $(\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz})$ in three directions.

Considering that the two sets of physical quantities correspond to different equations, two models are used to predict these two sets of physical quantities respectively:

$$
\begin{cases}
u, v, w = f(x,y,z) \\
\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz} = g(x,y,z)
\end{cases}
$$

In the above formula, $f$ is the strain model `disp_net`, and $g$ is the stress model `stress_net`, expressed in PaddleScience code as follows:

``` py linenums="15"
--8<--
examples/bracket/bracket.py:15:19
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, the input variable name of the strain model is specified as `("x", "y", "z")`, and the output variable name is `("u", "v", "w")`, these names are consistent with subsequent codes (the same applies to the stress model).

Then by specifying the number of layers and neurons of MLP, a neural network model `disp_net` with 6 hidden layers and 512 neurons per layer is instantiated, using `silu` as the activation function, and using `WeightNorm` weight normalization (the same applies to the stress model `stress_net`).

### 3.2 Equation Construction

The Bracket case involves the following linear elasticity equations, just use `LinearElasticity` built in PaddleScience.

--8<--
ppsci/equation/pde/linear_elasticity.py:31:43
--8<--

The corresponding equation instantiation code is as follows:

``` py linenums="32"
--8<--
examples/bracket/bracket.py:32:37
--8<--
```

### 3.3 Computational Domain Construction

The geometric area of this problem is specified by the stl file. Follow the command below to download and unzip it to the `bracket/` folder.

**Note: The stl file and test set data in the dataset are from [Bracket - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html#linear-elasticity-in-the-differential-form)**.

``` sh
# linux
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar

# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar -o bracket_dataset.tar

# unzip it
tar -xvf bracket_dataset.tar
```

After unzipping, the `bracket/stl` folder stores the stl geometric files required for computational domain construction.

???+ warning "Note"

    **Before using the `Mesh` class, you must install the three geometric dependency packages open3d, pysdf, and PyMesh according to the [1.4.2 Install Mesh Geometry [Optional]](https://paddlescience-docs.readthedocs.io/zh-cn/latest/en/install_setup/#142-mesh) document.**

Then use PaddleScience's built-in STL geometry class `Mesh` to read and parse these geometric files, and combine each computational domain through Boolean operations. The code is as follows:

``` py linenums="39"
--8<--
examples/bracket/bracket.py:39:51
--8<--
```

### 3.4 Constraint Construction

This case involves 5 constraints. Before constructing specific constraints, data reading configuration can be constructed first, so that this configuration can be reused when constructing multiple constraints later.

``` py linenums="53"
--8<--
examples/bracket/bracket.py:53:63
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on interior points of the backplane as an example, the code is as follows:

``` py linenums="106"
--8<--
examples/bracket/bracket.py:106:142
--8<--
```

The first parameter of `InteriorConstraint` is the equation (system) expression, which is used to describe how to calculate the constraint target. Here, fill in `equation["LinearElasticity"].equations` instantiated in the [3.2 Equation Construction](#32-equation-construction) chapter;

The second parameter is the target value of the constraint variable. In this problem, it is hoped that the 9 values `equilibrium_x`, `equilibrium_y`, `equilibrium_z`, `stress_disp_xx`, `stress_disp_yy`, `stress_disp_zz`, `stress_disp_xy`, `stress_disp_xz`, `stress_disp_yz` related to the LinearElasticity equation are all optimized to 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["geo"]` instantiated in the [3.3 Computational Domain Construction](#33-computational-domain-construction) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here, `batch_size` is set to `2048`.

The fifth parameter is the loss function. Here, the commonly used MSE function is selected, and `reduction` is set to `"sum"`, that is, the loss terms generated by all data points involved in the calculation will be summed;

The sixth parameter is geometric point filtering. Since this constraint is only applied to the backplane area, the points sampled on geo need to be filtered. Just pass in a lambda filter function here, which accepts the tensor `x, y, z` formed by the point set, and returns a boolean tensor indicating whether each point meets the filtering conditions. Not meeting is `False`, meeting is `True`;

The seventh parameter is the weight of each point participating in the loss calculation. Here we use `"sdf"` to indicate using the shortest distance (signed distance function value) of each point to the boundary as the weight. This sdf weighting method can increase the weight of points far from the boundary (hard samples) and reduce the weight of points close to the boundary (simple samples), which is beneficial to improve the accuracy and convergence speed of the model.

The eighth parameter is the name of the constraint condition. Each constraint condition needs to be named to facilitate subsequent indexing. Here it is named "support_interior".

Another constraint condition acting on the perforated plate is similar, the code is as follows:

``` py linenums="143"
--8<--
examples/bracket/bracket.py:143:179
--8<--
```

#### 3.4.2 Boundary Constraint

For the rear surface of the backplane, since it is fixed, the deformation of points on it in three directions is 0, so there are the following boundary constraint conditions:

``` py linenums="76"
--8<--
examples/bracket/bracket.py:76:85
--8<--
```

For the rectangular load surface on the right side of the perforated plate, each point on it is only subjected to a load in the positive z direction with magnitude $T$, and the stress in other directions is 0. There are the following boundary condition constraints:

``` py linenums="86"
--8<--
examples/bracket/bracket.py:86:94
--8<--
```

For surfaces other than the back of the backplane and the rectangular load surface on the right side of the perforated plate, there is no load, that is, the internal forces in three directions are balanced and the resultant force is 0. There are the following boundary condition constraints:

``` py linenums="95"
--8<--
examples/bracket/bracket.py:95:105
--8<--
```

After the equation constraint and boundary constraint are constructed, encapsulate them into a dictionary with the names just given as keys for subsequent access.

``` py linenums="180"
--8<--
examples/bracket/bracket.py:180:187
--8<--
```

### 3.5 Hyperparameter Setting

Next, you need to specify the number of training epochs in the configuration file. Here, based on experimental experience, 2000 training epochs are used, with 1000 optimization steps per epoch.

``` yaml linenums="74"
--8<--
examples/bracket/conf/bracket.yaml:74:77
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and the ExponentialDecay learning rate adjustment strategy commonly used in machine learning is used together.

``` py linenums="189"
--8<--
examples/bracket/bracket.py:189:193
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval. The data of the validation set comes from an external txt file, so first use the `ppsci.utils.reader` module to read the validation point set from the txt file:

``` py linenums="195"
--8<--
examples/bracket/bracket.py:195:256
--8<--
```

Then convert it to a dictionary and perform non-dimensionalization and normalization, and then wrap it into a dictionary and `eval_dataloader_cfg` (validation set dataloader configuration, construction method is similar to `train_dataloader_cfg`) together pass to `ppsci.validate.SupervisedValidator` to construct the validator.

``` py linenums="258"
--8<--
examples/bracket/bracket.py:258:303
--8<--
```

### 3.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, you can choose a suitable visualizer to visualize the output result.

The input data in this article is the input dictionary `input_dict` prepared in the validator construction, and the output data is the corresponding 9 predicted physical quantities, so you only need to save the evaluated output data as a **vtu format** file, and finally open it with visualization software to view it. The code is as follows:

``` py linenums="305"
--8<--
examples/bracket/bracket.py:305:322
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="324"
--8<--
examples/bracket/bracket.py:324:351
--8<--
```

## 4. Complete Code

``` py linenums="1" title="bracket.py"
--8<--
examples/bracket/bracket.py
--8<--
```

## 5. Result Display

The following shows the model prediction results, traditional algorithm solution results, and the difference between the two for the deflection $u, v, w$ in 3 directions and 6 stresses $\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz}$ on the test point set.

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/u.png){ loading=lazy }
  <figcaption>Left: Predicted deflection u on metal surface; Middle: Deflection u solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/v.png){ loading=lazy }
  <figcaption>Left: Predicted deflection v on metal surface; Middle: Deflection v solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/w.png){ loading=lazy }
  <figcaption>Left: Predicted deflection w on metal surface; Middle: Deflection w solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_xx.png){ loading=lazy }
  <figcaption>Left: Predicted stress sigma_xx on metal surface; Middle: Stress sigma_xx solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_xy.png){ loading=lazy }
  <figcaption>Left: Predicted stress sigma_xy on metal surface; Middle: Stress sigma_xy solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_xz.png){ loading=lazy }
  <figcaption>Left: Predicted stress sigma_xz on metal surface; Middle: Stress sigma_xz solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_yy.png){ loading=lazy }
  <figcaption>Left: Predicted stress sigma_yy on metal surface; Middle: Stress sigma_yy solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_yz.png){ loading=lazy }
  <figcaption>Left: Predicted stress sigma_yz on metal surface; Middle: Stress sigma_yz solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

<figure markdown>
  ![bracket_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Bracket/sigma_zz.png){ loading=lazy }
  <figcaption>Left: Predicted stress sigma_zz on metal surface; Middle: Stress sigma_zz solved by traditional algorithm; Right: Difference between the two</figcaption>
</figure>

It can be seen that the model prediction results are basically consistent with the traditional algorithm solution results.

## 6. References

- [Bracket - NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html)
- [Scaling of Differential Equations](https://hplgit.github.io/scaling-book/doc/pub/book/html/sphinx-cbc/index.html)
- [Matlab PDE toolbox](https://www.mathworks.com/help/pde/ug/deflection-analysis-of-a-bracket.html)
