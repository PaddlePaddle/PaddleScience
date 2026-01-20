# Chip Heat Simulation

<a href="https://aistudio.baidu.com/projectdetail/7682679" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    python chip_heat.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python chip_heat.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/ChipHeat/chip_heat_pretrained.pdparams
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [chip_heat_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/ChipHeat/ChipHeat_pretrained.pdparams) | MSE.chip(down_mse): 0.04177<br>MSE.chip(left_mse): 0.01783<br>MSE.chip(right_mse): 0.03767<br>MSE.chip(top_mse): 0.05034 |

## 1. Background Introduction

Chip thermal simulation research mainly focuses on predicting and analyzing the temperature distribution of integrated circuits (ICs) during operation, as well as the impact of thermal effects on chip performance, power consumption, reliability, and lifespan. As electronic devices evolve towards higher performance, higher density, and smaller sizes, thermal management has become a critical challenge in chip design and manufacturing.

Chip thermal simulation research provides important tools and methods for understanding and solving chip thermal management problems, playing a crucial role in improving chip performance, reducing power consumption, ensuring reliability, and extending lifespan. As electronic devices develop towards higher performance and compactness, the importance of thermal simulation research will further increase.

Chip thermal simulation has multiple importances in engineering and scientific fields, mainly reflected in the following aspects:

- Design optimization and validation: Chip thermal simulation can help engineers and scientists evaluate the thermal characteristics of different structures and materials in the early stages of design to optimize design and verify its reliability. By simulating temperature distribution and heat conduction effects under different workloads, potential thermal problems can be discovered in advance and targeted improvements can be made, thereby reducing later development costs and risks.
- Thermal management and heat dissipation design: Chip thermal simulation can help design effective thermal management systems and heat dissipation schemes to ensure that the chip remains within a safe operating temperature range during long-term high-load operation. By analyzing the heat dissipation structure, fan configuration, heat sink design, etc. around the chip, heat conduction and heat dissipation efficiency can be optimized, improving system stability and reliability.
- Performance prediction and optimization: Temperature has a significant impact on chip performance and stability. Chip thermal simulation can help predict chip performance under different workloads and environmental conditions, including processor speed, power consumption, and lifespan of electronic devices. By modeling and analyzing thermal effects, chip design and operating conditions can be optimized to achieve better performance and reliability.
- Energy saving and environmental protection: Effective thermal management and heat dissipation design can reduce system energy consumption and improve energy utilization efficiency, thereby achieving energy saving and environmental protection goals. By reducing heat loss and waste in the system, energy consumption and carbon emissions can be reduced, minimizing negative impacts on the environment.

In summary, chip thermal simulation plays an important role and value in engineering and scientific fields, helping to achieve positive results in design optimization, performance improvement, cost reduction, environmental protection, etc.

## 2. Problem Definition

### 2.1 Problem Description

To build a general thermal simulation model, we first briefly describe the thermal simulation problem in general. Thermal simulation aims to predict the temperature field of a given object by globally solving the heat conduction equation, which can usually be represented by the following governing equation:

$$
k \Delta T(x,t) + S(x,t) = \rho c_p \dfrac{\partial T(x,t)}{\partial t},\quad \text { in } \Omega\times (0,t_{*}),
$$

where $\Omega\subset \mathbb{R}^{n},~n=1,2,3$ is the simulation area of the given object material, as shown in the figure for a 2D chip simulation area with random heat source distribution. $T(x,t),~S(x,t)$ represent the temperature and heat source distribution at any spatio-temporal position $(x,t)$, respectively, and $t_*$ is the temperature threshold. Here $k$, $\rho$, $c_p$ are all material properties of the given object, representing thermal conductivity, mass density, and specific heat capacity, respectively. For convenience, we focus on the static temperature field of the given object material and simplify the equation by setting $\frac{dT}{dt}=0$:

$$
\tag{1} k \Delta T(x) + S(x) = 0,\quad \text { in } \Omega.
$$

<figure markdown>
  ![domain_chip.pdf](https://paddle-org.bj.bcebos.com/paddlescience/docs/ChipHeat/chip_domain.PNG){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> 2D chip simulation area with random heat source distribution inside, and arbitrary boundary conditions on the boundary.</figcaption>
</figure>

For a general thermal simulation model of a given object material, in addition to satisfying the governing equation (1), its temperature field also depends on some key PDE configurations, including but not limited to material properties and geometric parameters.

The first type of PDE configuration is the boundary conditions of the given object material:

- Dirichlet boundary condition: The temperature field on the surface is fixed at $q_d$:

$$
T = q_d.
$$

- Neumann boundary condition: The temperature flux on the surface is fixed at $q_n$. When $q_n =0$, it indicates that the surface is completely insulated, called adiabatic boundary condition.

$$
\tag{2} -k \dfrac{\partial T}{\partial n} = q_n.
$$

- Convection boundary condition: Also known as Newton boundary condition, this boundary condition corresponds to the balance between heat conduction and convection in the same direction on the surface, where $h$ and $T_{amb}$ represent the surface convection coefficient and ambient temperature.

$$
-k \dfrac{\partial T}{\partial n} = h(T-T_{amb}).
$$

- Radiation boundary condition: This boundary condition corresponds to electromagnetic radiation generated by temperature difference on the surface, where $\epsilon$ and $\sigma$ represent thermal radiation coefficient and Stefan-Boltzmann coefficient, respectively.

$$
-k \dfrac{\partial T}{\partial n} = \epsilon \sigma (T^4-T_{amb}^4).
$$

The second type of PDE configuration is the position and intensity of boundary or internal heat sources of the given object material. This work considers the following two types of heat sources:

- Boundary random heat source: Defined by Neumann boundary condition (2), where $q_n$ is a function of $x$, i.e., any given temperature flux distribution;
- Internal random heat source: Defined by governing equation (1), where $S(x)$ is a function of $x$, i.e., any given heat source distribution.

Our goal is to obtain the corresponding temperature field distribution on the general thermal simulation model of a given object material by inputting any first or second type of design configuration, where we arbitrarily specify the boundary type and parameters on the boundary. It is worth noting that the PI-DeepONet method for general thermal simulation developed in this work is not limited to the conditions of the first or second type of design configuration and regular geometric shapes. With further code modifications beyond the scope of current work, they can be applied to various loads, material properties, and even various irregular geometric shapes.

### 2.2 PI-DeepONet Model

The PI-DeepONet model combines DeepONet and PINN methods, which is a deep neural network model combining physical information and operator learning. This model can enhance the DeepONet model through the physical information of governing equations, and can use different PDE configurations as input data for different branch networks, so it can be effectively used for ultra-fast model prediction under various (parametric and non-parametric) PDE configurations.

For the chip thermal simulation problem, the PI-DeepONet model can be represented as the model structure shown in the figure:

<figure markdown>
  ![pi_deeponet.pdf](https://paddle-org.bj.bcebos.com/paddlescience/docs/ChipHeat/pi_deeponet.PNG){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

As shown in the figure, we used a total of 3 branch networks and one trunk network. The branch networks input boundary type index, random heat source distribution $S(x, y)$ and boundary function $Q(x, y)$ respectively, and the trunk network inputs 2D coordinate point information. Each branch network and trunk network outputs a $q$-dimensional feature vector. All these output features are combined through Hadamard (element-wise) product, and then the resulting vectors are summed as the scalar output of the predicted temperature field.

## 3. Problem Solving

Next, we will explain how to convert this problem into PaddleScience code step by step and solve the heat exchanger thermal simulation problem using deep learning methods. In order to quickly understand PaddleScience, only key steps such as model construction and constraint construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the chip thermal simulation problem, each known coordinate point $(x, y)$ and each set of boundary type $bt$, random heat source distribution $S(x, y)$ and boundary function $Q(x, y)$ correspond to a set of chip temperature distribution $T$, an unknown quantity to be solved. Here we use 3 branch networks and one trunk network, all 4 networks are MLP (Multilayer Perceptron). The 3 branch networks represent the mapping functions $f_1, f_2, f_3: \mathbb{R}^3 \to \mathbb{R}^{q}$ from $(bt, S, Q)$ to output functions $(b_1, b_2, b_3)$ respectively, i.e.:

$$
\begin{aligned}
b_1 &= f_1(bt),\\
b_2 &= f_2(S),\\
b_3 &= f_3(Q).
\end{aligned}
$$

In the above formula, $f_1, f_2, f_3$ are all MLP models, $(b_1, b_2, b_3)$ are the output functions of the three branch networks respectively, and $q$ is the dimension of the output function. The trunk network represents the mapping function $f_4: \mathbb{R} \to \mathbb{R}^{q}$ from $(x, y)$ to output function $t_0$, i.e.:

$$
\begin{aligned}
t_0 &= f_4(x, y).
\end{aligned}
$$

In the above formula, $f_4$ is an MLP model, $(t_0)$ is the output function of the trunk network, and $q$ is the dimension of the output function. We can perform Hadamard (element-wise) product on the output functions of the three branch networks and the trunk network $(b_1, b_2, b_3, t_0)$ and then sum them up to obtain the scalar temperature field, i.e.:

$$
T = \sum_{i=1}^q b_1^ib_2^ib_3^it_0^i.
$$

We define the ChipHeats model class built in PaddleScience and call it. The PaddleScience code is as follows:

``` py linenums="77"
--8<--
examples/chip_heat/chip_heat.py:77:78
--8<--
```

In this way, we instantiated a ChipHeats model with 4 MLP models. Each branch network contains 9 hidden layers with 256 neurons per layer. The trunk network contains 6 hidden layers with 128 neurons per layer. "Swish" is used as the activation function. The neural network model `model` contains an output function $T$. For more relevant content, please refer to [A fast general thermal simulation model based on MultiBranch Physics-Informed deep operator neural network](https://doi.org/10.1063/5.0194245).

### 3.2 Computational Domain Construction

Construct the training area for the chip thermal simulation problem in this article, which is a 2D area of $[0, 1]\times[0, 1]$. This area can directly use the spatial geometry `Rectangle` built in PaddleScience to construct the computational domain. The code is as follows:

``` py linenums="79"
--8<--
examples/chip_heat/chip_heat.py:79:81
--8<--
```

???+ tip "Tip"

    `Rectangle` and `TimeDomain` are two `Geometry` derived classes that can be used independently.

    If the input data only comes from a two-dimensional rectangular geometric domain, you can directly use `ppsci.geometry.Rectangle(...)` to create a spatial geometric domain object;

    If the input data only comes from a one-dimensional time domain, you can directly use `ppsci.geometry.TimeDomain(...)` to construct a time domain object.

### 3.3 Input Data Construction

Use 2D correlated and scale-invariant Gaussian random fields to generate random heat source distribution $S(x)$ and boundary function $Q(x)$. We refer to the Python implementation described in [gaussian-random-fields](https://github.com/bsciolla/gaussian-random-fields), where correlation is explained by scale-free spectrum, i.e.:

$$
P(k) \sim \dfrac{1}{|k|^{\alpha/2}}.
$$

The smoothness of the sampling function is determined by the length scale coefficient $\alpha$. The larger the $\alpha$ value, the smoother the random heat source distribution $S(x)$ and boundary function $Q(x)$ obtained. In this article we use $\alpha = 4$. This parameter can also be adjusted to generate heat source distribution $S(x)$ and boundary function $Q(x)$ similar to specific optimization tasks.

Generate training and test input data for random heat source distribution $S(x)$ and boundary function $Q(x)$ through Gaussian random fields. The code is as follows:

``` py linenums="84"
--8<--
examples/chip_heat/chip_heat.py:84:95
--8<--
```

Then classify the training data and test data according to spatial coordinates, classifying them into left, right, top, bottom and internal data. The code is as follows:

``` py linenums="97"
--8<--
examples/chip_heat/chip_heat.py:97:192
--8<--
```

### 3.4 Constraint Construction

Before constructing constraints, we need to introduce `ChipHeatDataset`, which inherits from `Dataset` class and can iteratively read array datasets composed of different `numpy.ndarray`. Due to the large number of model branch networks used, the amount of data used is large. If the data is combined first, the memory occupied by the input data will be large, so `ChipHeatDataset` is used to iteratively read data.

The chip thermal simulation problem consists of equations described in [2.1 Problem Description](#21). At this time, we set five constraint conditions for left, right, top, bottom and internal data respectively. Next, use `SupervisedConstraint` built in PaddleScience to construct the above four constraint conditions. The code is as follows:

``` py linenums="194"
--8<--
examples/chip_heat/chip_heat.py:194:381
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraint, where the `"dataset"` field represents the training dataset information used, and each field represents:

1. `name`: Dataset type, here `ChipHeatDataset` means iteratively reading data in batches;
2. `input`: Input variable name;
3. `label`: Label variable name;
4. `index`: Index of input dataset;
5. `data_type`: Type of input data;
6. `weight`: Weight size.

The "sampler" field defines the `Sampler` class name used as `BatchSampler`, and also specifies that the parameters `drop_last` is `False` and `shuffle` is `True` during initialization of this class.

The second parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is `"mean"`, that is, we will sum and average the loss terms generated by all data points involved in the calculation;

The third parameter is the label expression list. Here we use equation expressions corresponding to left, right, top, bottom and internal regions. At the same time, we use $0, 1, 2, 3$ to represent Dirichlet boundary, Neumann boundary, convection boundary and radiation boundary respectively. Different boundary conditions are set for different boundary types;

The fourth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing.

After the differential equation constraint and supervised constraint are constructed, encapsulate them into a dictionary with the names we just named as keys for subsequent access.

``` py linenums="382"
--8<--
examples/chip_heat/chip_heat.py:382:389
--8<--
```

### 3.5 Optimizer Construction

Next we need to specify the learning rate, which is set to 0.001. The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="391"
--8<--
examples/chip_heat/chip_heat.py:391:392
--8<--
```

### 3.6 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval. We use `ppsci.validate.SupervisedValidator` to construct the validator.

``` py linenums="394"
--8<--
examples/chip_heat/chip_heat.py:394:495
--8<--
```

The configuration is similar to the setting of [3.4 Constraint Construction](#34). It should be noted that since the amount of data used for evaluation is not very large, we do not need to use `ChipHeatDataset` to iteratively read data, but use `NamedArrayDataset` to read data here.

### 3.7 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="497"
--8<--
examples/chip_heat/chip_heat.py:497:513
--8<--
```

### 3.8 Result Visualization

Finally, prediction and visualization are performed on the given visualization area. The visualization data is a two-dimensional point set in the area. At each coordinate $(x, y)$, the corresponding temperature value $T$ is plotted. Here we plot the image of $T$ change on the area. At the same time, different boundary types, random heat source distribution $S(x)$ and boundary function $Q(x)$ can be set as needed. The code is as follows:

``` py linenums="514"
--8<--
examples/chip_heat/chip_heat.py:514:535
--8<--
```

## 4. Complete Code

``` py linenums="1" title="chip_heat.py"
--8<--
examples/chip_heat/chip_heat.py
--8<--
```

## 5. Result Display

Three sets of random heat source distributions $S(x)$ are generated by Gaussian random fields, as shown in the first row of the figure. Next, we can set any boundary condition in the first type of PDE. Here we give five types of boundary conditions, as shown in the boundary equation in the first column of governing equations in the figure. During the test, we set $k = 100,~h = 100,~T_{amb} = 1,~\epsilon\sigma= 5.6 \times 10^{-7}$. Under different random heat source $S(x)$ distributions and different boundary conditions, the temperature field distribution tested by the PI-DeepONet model is shown in the figure. From the figure, it can be seen that although there are significant differences in random heat source distribution $S(x)$ and boundary conditions between test samples, the PI-DeepONet model can correctly predict the two-dimensional diffusion property solutions inside and on the boundary controlled by the heat conduction equation.

<figure markdown>
  ![chip.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/ChipHeat/chip.png){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

## 6. References

Reference: [A fast general thermal simulation model based on MultiBranch Physics-Informed deep operator neural network](https://doi.org/10.1063/5.0194245)

Reference Code: [gaussian-random-fields](https://github.com/bsciolla/gaussian-random-fields)
