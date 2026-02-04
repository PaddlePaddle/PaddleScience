# Heat_Exchanger

=== "Model Training Command"

    ``` sh
    python heat_exchanger.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python heat_exchanger.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/HEDeepONet/HEDeepONet_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python heat_exchanger.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python heat_exchanger.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [heat_exchanger_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/HEDeepONet/HEDeepONet_pretrained.pdparams) | The L2 norm error between the actual heat exchanger efficiency and the predicted heat exchanger efficiency: 0.02087<br>MSE.heat_boundary(interior_mse): 0.52005<br>MSE.cold_boundary(interior_mse): 0.16590<br>MSE.wall(interior_mse): 0.01203 |

## 1. Background Introduction

### 1.1 Heat Exchanger

Heat exchanger (also known as heat exchange equipment) is a device used to transfer heat from a hot fluid to a cold fluid to meet specified process requirements. It is an industrial application of convective heat transfer and heat conduction.

Heat exchangers are found in general air conditioning equipment, that is, the cooling and heating coils of indoor and outdoor air conditioning units; when the heat exchanger is used for heat release, it is called a "condenser", and when it is used for heat absorption, it is called an "evaporator". The physical reactions of the refrigerant in these two are opposite. Therefore, when a household air conditioner is used as a cooling machine, the heat exchanger of the indoor unit is called an evaporator, and the outdoor unit is called a condenser; when it acts as a heater, the opposite is true. The figure shows an evaporative cycle refrigeration system. Research on heat exchanger thermal simulation can provide important references and guidance for optimizing design, improving performance and reliability, energy conservation and emission reduction, and new technology research and development.

<figure markdown>
  ![heat_exchanger.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/heat_exchanger.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Evaporative Cycle Refrigeration System</figcaption>
</figure>

Heat exchangers have multiple importance in engineering and scientific fields, and their role and value are mainly reflected in the following aspects:

- Energy conversion efficiency: Heat exchangers play an important role in energy conversion. By optimizing the transfer and utilization of heat energy, the efficiency of power plants, industrial production and other energy conversion processes can be improved. They help convert heat energy in fuel into electrical or mechanical energy, maximizing the use of energy resources.
- Industrial production optimization: In chemical, petroleum, pharmaceutical and other industries, heat exchangers are used for processes such as heating, cooling, distillation and evaporation. Through effective heat exchanger design and application, production efficiency can be improved, temperature and pressure can be controlled, product quality can be improved, and energy consumption can be reduced.
- Temperature control and regulation: Heat exchangers can be used to control temperature. In industrial production, maintaining appropriate temperature is crucial for reaction rate, product quality and equipment life. Heat exchangers can help regulate and maintain system temperature within ideal operating ranges.
- Environmental protection and sustainable development: By improving energy conversion efficiency and energy utilization in industrial production processes, heat exchangers help reduce dependence on natural resources and reduce negative impacts on the environment. The improvement of energy efficiency can also reduce greenhouse gas emissions, which is conducive to environmental protection and sustainable development.
- Engineering design and innovation: In the field of engineering design, the optimal design and innovation of heat exchangers promote the development of engineering technology. Continuously improved heat exchanger designs can improve performance, reduce space occupation, and adapt to a variety of complex process requirements.

In summary, the importance of heat exchangers in engineering and scientific fields is reflected in their important contributions to energy utilization efficiency, industrial production process optimization, temperature control, environmental protection and engineering technology innovation. Continuous improvement and innovation in these aspects promote the development of engineering technology and help solve important challenges in energy and environment.

## 2. Problem Definition

### 2.1 Problem Description

Assume that the fluid flow inside the heat exchanger is one-dimensional, as shown in the figure.

<figure markdown>
  ![1DHE.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/1DHE.png){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

Ignore the thermal resistance of the wall and axial heat conduction; there is no heat exchange with the outside world, as shown in the figure. The energy conservation equations for the three nodes of hot and cold fluids and heat transfer wall are:

$$
\begin{aligned}
& L\left(\frac{q_m c_p}{v}\right)_{\mathrm{c}} \frac{\partial T_{\mathrm{c}}}{\partial \tau}-L\left(q_m c_p\right)_{\mathrm{c}} \frac{\partial T_{\mathrm{c}}}{\partial x}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{c}}\left(T_{\mathrm{w}}-T_{\mathrm{c}}\right), \\
& L\left(\frac{q_m c_p}{v}\right)_{\mathrm{h}} \frac{\partial T_{\mathrm{h}}}{\partial \tau}+L\left(q_m c_p\right)_{\mathrm{h}} \frac{\partial T_{\mathrm{h}}}{\partial x}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{h}}\left(T_{\mathrm{w}}-T_{\mathrm{h}}\right), \\
& \left(M c_p\right)_{\mathrm{w}} \frac{\partial T_{\mathrm{w}}}{\partial \tau}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{h}}\left(T_{\mathrm{h}}-T_{\mathrm{w}}\right)+\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{c}}\left(T_{\mathrm{c}}-T_{\mathrm{w}}\right).
\end{aligned}
$$

Where:

- $T$ represents temperature,
- $q_m$ represents mass flow rate,
- $c_p$ represents specific heat capacity,
- $v$ represents flow velocity,
- $L$ represents flow length,
- $\eta_{\mathrm{o}}$ represents fin surface efficiency,
- $\alpha$ represents heat transfer coefficient,
- $A$ represents heat transfer area,
- $M$ represents mass of heat transfer structure,
- $\tau$ represents corresponding time,
- $x$ represents flow direction,
- Subscripts $\mathrm{h}$, $\mathrm{c}$ and $\mathrm{w}$ represent hot fluid, cold fluid and heat exchange wall respectively.

The inlet and outlet parameters of cold and hot fluids in the heat exchanger satisfy energy conservation, i.e.:

$$
\left(q_m c_p\right)_{\mathrm{h}}\left(T_{\mathrm{h}, \text { in }}-T_{\mathrm{h}, \text { out }}\right)=\left(q_m c_p\right)_c\left(T_{\mathrm{c}, \text {out }}-T_{\mathrm{c}, \text {in }}\right).
$$

Heat exchanger efficiency $\eta$ is the ratio of actual heat transfer to theoretical maximum heat transfer, i.e.:

$$
\eta=\frac{\left(q_m c_p\right)_{\mathrm{h}}\left(T_{\mathrm{h}, \text { in }}-T_{\mathrm{h}, \text { out }}\right)}{\left(q_m c_p\right)_{\text {min }}\left(T_{\mathrm{h}, \text { in }}-T_{\mathrm{c}, \text { in }}\right)},
$$

In the formula, subscript $min$ represents the smaller value of heat capacity of cold and hot fluids.

### 2.2 PI-DeepONet Model

The PI-DeepONet model combines DeepONet and PINN methods, and is a deep neural network model combining physical information and operator learning. This model can enhance the DeepONet model through physical information of governing equations, and can use different PDE configurations as input data for different branch networks, so it can be effectively used for ultra-fast model prediction under various (parametric and non-parametric) PDE configurations.

For the heat exchanger problem, the PI-DeepONet model can be represented as the model structure shown in the figure:

<figure markdown>
  ![PI-DeepONet.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/PI-DeepONet.png){ loading=lazy style="height:80%;width:80%" align="center" }
</figure>

As shown in the figure, we use a total of 2 branch networks and one trunk network. The branch networks input the mass flow rate of the hot side and the mass flow rate of the cold side respectively, and the trunk network inputs the one-dimensional coordinate point coordinates and time information. Each branch network and trunk network outputs a $q$-dimensional feature vector. All these output features are combined through Hadamard (element-wise) product, and then the resulting vectors are summed as the scalar output of the predicted temperature field.

## 3. Problem Solving

Next, we will explain how to convert this problem into PaddleScience code step by step and solve this heat exchanger thermal simulation problem using deep learning methods. In order to quickly understand PaddleScience, only key steps such as model construction and constraint construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the heat exchanger thermal simulation problem, each known coordinate point $(t, x)$ and each set of hot side mass flow rate and cold side mass flow rate $(q_{mh}, q_{mc})$ correspond to a set of hot side fluid temperature $T_h$, cold side fluid temperature $T_c$ and heat exchange wall temperature $T_h$, three unknown quantities to be solved. Here we use 2 branch networks and one trunk network, all 3 networks are MLP (Multilayer Perceptron). The 2 branch networks represent the mapping functions $f_1, f_2: \mathbb{R}^2 \to \mathbb{R}^{3q}$ from $(q_{mh}, q_{mc})$ to output functions $(b_1, b_2)$, i.e.:

$$
\begin{aligned}
b_1 &= f_1(q_{mh}),\\
b_2 &= f_2(q_{mc}).
\end{aligned}
$$

In the above formula, $f_1, f_2$ are MLP models, $(b_1, b_2)$ are the output functions of the two branch networks respectively, and $3q$ is the dimension of the output function. The trunk network represents the mapping function $f_3: \mathbb{R}^2 \to \mathbb{R}^{3q}$ from $(t, x)$ to output function $t_0$, i.e.:

$$
\begin{aligned}
t_0 &= f_3(t,x).
\end{aligned}
$$

In the above formula, $f_3$ is an MLP model, $(t_0)$ is the output function of the trunk network, and $3q$ is the dimension of the output function. We can divide the output functions $(b_1, b_2, t_0)$ of the two branch networks and the trunk network into 3 groups, and then perform Hadamard (element-wise) product on the output functions of each group and sum them up to obtain the scalar temperature field, i.e.:

$$
\begin{aligned}
T_h &= \sum_{i=1}^q b_1^ib_2^i t_0^i,\\
T_c &= \sum_{i=q+1}^{2q} b_1^ib_2^i t_0^i,\\
T_w &= \sum_{i=2q+1}^{3q} b_1^ib_2^i t_0^i.
\end{aligned}
$$

We define the HEDeepONets model class built in PaddleScience and call it. The PaddleScience code is as follows:

``` py linenums="33"
--8<--
examples/heat_exchanger/heat_exchanger.py:33:34
--8<--
```

In this way, we instantiated a HEDeepONets model with 3 MLP models. Each branch network contains 9 hidden layers with 256 neurons per layer. The trunk network contains 6 hidden layers with 128 neurons per layer. "Swish" is used as the activation function. The neural network model `model` contains three output functions $T_h, T_c, T_w$.

### 3.2 Computational Domain Construction

Construct the training area for the heat exchanger problem in this article, which is a one-dimensional area of [0, 1], and the time domain is 21 moments [0,1,2,...,21]. This area can directly use the spatial geometry `Interval` and time domain `TimeDomain` built in PaddleScience to combine into a time-space `TimeXGeometry` computational domain. The code is as follows:

``` py linenums="36"
--8<--
examples/heat_exchanger/heat_exchanger.py:36:43
--8<--
```

???+ tip "Tip"

    `Rectangle` and `TimeDomain` are two `Geometry` derived classes that can be used independently.

    If the input data only comes from a two-dimensional rectangular geometric domain, you can directly use `ppsci.geometry.Rectangle(...)` to create a spatial geometric domain object;

    If the input data only comes from a one-dimensional time domain, you can directly use `ppsci.geometry.TimeDomain(...)` to construct a time domain object.

### 3.3 Input Data Construction

- Construct input time and space uniform data through `TimeXGeometry` computational domain,
- Generate random numbers between (0, 2) through `np.random.rand`. These random numbers are used to construct training and test data for mass flow rates on the hot and cold sides.

Combine time and space uniform data with hot and cold side mass flow rate data to obtain the final training and test input data. The code is as follows:

``` py linenums="45"
--8<--
examples/heat_exchanger/heat_exchanger.py:45:63
--8<--
```

Then classify the training data according to spatial coordinates and time, classifying training data and test data into left boundary data, internal data, right boundary data and initial value data. The code is as follows:

``` py linenums="65"
--8<--
examples/heat_exchanger/heat_exchanger.py:65:124
--8<--
```

### 3.4 Equation Construction

The heat exchanger thermal simulation problem consists of equations described in [2.1 Problem Description](#21-problem-description). Here we define the `HeatEquation` equation class built in PaddleScience to construct this equation. Specify that the parameters of this class are all 1. The code is as follows:

``` py linenums="126"
--8<--
examples/heat_exchanger/heat_exchanger.py:126:136
--8<--
```

### 3.5 Constraint Construction

The heat exchanger thermal simulation problem consists of equations described in [2.1 Problem Description](#21-problem-description). We set the following boundary conditions:

$$
\begin{aligned}
T_h(t,0) &= 10,\\
T_c(t,1) &= 1.
\end{aligned}
$$

At the same time, we set initial value conditions:

$$
\begin{aligned}
T_h(0,x) &= 10,\\
T_c(0,x) &= 1,\\
T_w(0,x) &= 5.5.
\end{aligned}
$$

At this time, we set four constraint conditions for left boundary data, internal data, right boundary data and initial value data. Next, use `SupervisedConstraint` built in PaddleScience to construct the above four constraint conditions. The code is as follows:

``` py linenums="138"
--8<--
examples/heat_exchanger/heat_exchanger.py:138:263
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraint, where the `"dataset"` field represents the training dataset information used, and each field represents:

1. `name`: Dataset type, here `"NamedArrayDataset"` means reading data sequentially in batches;
2. `input`: Input variable name;
3. `label`: Label variable name;
4. `weight`: Weight size.

The "sampler" field defines the `Sampler` class name used as `BatchSampler`, and also specifies that the parameters `drop_last` is `False` and `shuffle` is `True` during initialization of this class.

The second parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is `"mean"`, that is, we will sum and average the loss terms generated by all data points involved in the calculation;

The third parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing.

After the differential equation constraint and supervised constraint are constructed, encapsulate them into a dictionary with the names we just named as keys for subsequent access.

``` py linenums="264"
--8<--
examples/heat_exchanger/heat_exchanger.py:264:270
--8<--
```

### 3.6 Optimizer Construction

Next we need to specify the learning rate, which is set to 0.001. The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="272"
--8<--
examples/heat_exchanger/heat_exchanger.py:272:273
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval. We use `ppsci.validate.SupervisedValidator` to construct the validator.

``` py linenums="275"
--8<--
examples/heat_exchanger/heat_exchanger.py:275:349
--8<--
```

The configuration is similar to the setting of [3.5 Constraint Construction](#35-constraint-construction).

### 3.8 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="351"
--8<--
examples/heat_exchanger/heat_exchanger.py:351:371
--8<--
```

### 3.9 Result Visualization

Finally, prediction and visualization are performed on the given visualization area. Assuming that the mass flow rates of the cold and hot sides are both 1, the visualization data is a one-dimensional point set in the area. The coordinate corresponding to each moment $t$ is $x^i$, and the corresponding value is $(T_h^{i}, T_c^i, T_w^i)$. Here we plot the change images of $T_h, T_c, T_w$ with time. At the same time, calculate the heat exchanger efficiency $\eta$ according to the heat exchanger efficiency formula, and plot the change image of heat exchanger efficiency $\eta$ with time. The code is as follows:

``` py linenums="373"
--8<--
examples/heat_exchanger/heat_exchanger.py:373:430
--8<--
```

## 4. Complete Code

``` py linenums="1" title="heat_exchanger.py"
--8<--
examples/heat_exchanger/heat_exchanger.py
--8<--
```

## 5. Result Display

As shown in the figure, the variation images of hot side temperature, cold side temperature, wall temperature $T_h, T_c, T_w$ with heat transfer area $A$ at different moments and the variation image of heat exchanger efficiency $\eta$ with time.

???+ info "Note"

    This case is only shown as a demo and has not been fully tuned. Some of the results shown below may differ from OpenFOAM.

<figure markdown>
  ![T_h.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/T_h.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Variation image of hot side temperature T_h with heat transfer area A at different moments</figcaption>
</figure>

<figure markdown>
  ![T_c.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/T_c.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Variation image of cold side temperature T_c with heat transfer area A at different moments</figcaption>
</figure>

<figure markdown>
  ![T_w.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/T_w.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Variation image of wall temperature T_w with heat transfer area A at different moments</figcaption>
</figure>

<figure markdown>
  ![eta.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/eta.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Variation image of heat exchanger efficiency with time</figcaption>
</figure>

It can be seen from the figure:

- The hot side temperature gradually decreases with time at $A=1$, and the cold side temperature gradually increases with time at $A=0$;
- The wall temperature gradually decreases with time at $A=1$, and gradually increases with time at $A=0$;
- The heat exchanger efficiency gradually increases with time and reaches the maximum value at $t=21$.

At the same time, we can assume that the mass flow rate on the hot side and the mass flow rate on the cold side are equal, i.e., $q_h=q_c$, define the number of heat transfer units:

$$
NTU = \dfrac{Ak}{(q_mc)_{min}}.
$$

For different numbers of heat transfer units, we can calculate the corresponding heat exchanger efficiency respectively, and draw the variation image of heat exchanger efficiency with the number of heat transfer units, as shown in the figure.

<figure markdown>
  ![eta-1.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/HEDeepONet/eta-1.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Variation image of heat exchanger efficiency with number of heat transfer units</figcaption>
</figure>

It can be seen from the figure: the heat exchanger efficiency gradually increases with the change of the number of heat transfer units, which is also consistent with the actual change rule of heat exchanger efficiency with the number of heat transfer units.
