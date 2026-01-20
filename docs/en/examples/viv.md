# VIV (Vortex Induced Vibration)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6160556?contributionType=1&sUid=438690&shared=1&ts=1683961088129" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    python viv.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python viv.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/viv/viv_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python viv.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python viv.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [viv_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/viv_pretrained.pdparams)<br>[viv_pretrained.pdeqn](https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/viv_pretrained.pdeqn) | eta_l2/MSE.eta: 0.00875<br/>eta_l2/MSE.f: 0.00921 |

## 1. Background Introduction

Vortex-Induced Vibration (VIV) is a fluid-structure interaction vibration phenomenon, mainly occurring when fluid flows around a cylinder or tube. This vibration phenomenon has important applications in ocean engineering and wind engineering.

In ocean engineering, VIV problems mainly involve the VIV response analysis of offshore platforms (such as pile foundations, risers, etc.). These platforms operate in ocean currents and are subject to VIV. This vibration may cause fatigue damage to the platform structure, so this issue needs to be considered when designing offshore platforms.

In wind engineering, VIV problems mainly involve the VIV response analysis of wind turbines. Wind turbine blades are subject to airflow VIV during operation, which may cause fatigue damage to the blades. In order to ensure the safe operation of wind turbines, in-depth research on this issue is required.

In short, the application of VIV problems mainly involves the fields of ocean engineering and wind engineering, which is of great significance for the development of these fields.

When the vortex shedding frequency is close to the natural frequency of the structure, the cylinder will undergo VIV. The VIV system is equivalent to a spring-damper system:

![VIV_1D_SpringDamper](https://paddle-org.bj.bcebos.com/paddlescience/docs/ViV/VIV_1D_SpringDamper.png)

## 2. Problem Definition

The governing equation involved in this problem involves three physical quantities: $\lambda_1$, $\lambda_2$ and $\rho$, representing natural damping, structural characteristic stiffness and mass respectively. The governing equation is defined as follows:

$$
\rho \dfrac{\partial^2 \eta}{\partial t^2} + \lambda_1 \dfrac{\partial \eta}{\partial t} + \lambda_2 \eta = f
$$

The model is based on the assumption of dimensionless velocity $U_r=\dfrac{u}{f_n*d}=8.5$ corresponding to $Re=500$. We use the transverse amplitude $\eta$ of the cylinder vibration caused by the fluid passing through the cylinder and the corresponding lift $f$ as supervision data.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the VIV problem, given time $t$, the above system has transverse amplitude $\eta$ and lift $f$ as unknown quantities to be solved, and the system itself also contains two parameters $\lambda_1, \lambda_2$. Therefore, we use a relatively simple MLP (Multilayer Perceptron) here to represent the mapping function $g: \mathbb{R}^1 \to \mathbb{R}^2$ from $t$ to $(\eta, f)$, that is:

$$
\eta, f = g(t)
$$

In the above formula, $g$ is the MLP model itself, expressed in PaddleScience code as follows

``` py linenums="22"
--8<--
examples/fsi/viv.py:22:23
--8<--
```

In order to accurately and quickly access the value of specific variables during calculation, we specify here that the input variable name of the network model is `("t_f",)`, and the output variable name is `("eta",)`,
`t_f` represents input time $t$, `eta` represents output amplitude $\eta$. These names are consistent with subsequent code.

Then by specifying the number of layers, number of neurons, and activation function of the MLP, we instantiate a neural network model `model` with 5 hidden layers, 50 neurons per layer, using "tanh" as the activation function.

### 3.2 Equation Construction

Since VIV uses the VIV equation, `VIV` built into PaddleScience can be used directly.

``` py linenums="25"
--8<--
examples/fsi/viv.py:25:26
--8<--
```

We added two learnable parameters `k1` and `k2` to the equation to estimate $\lambda_1$ and $\lambda_2$, and their relationship is $\lambda_1 = e^{k1}, \lambda_2 = e^{k2}$

Therefore, when instantiating the `VIV` class, we need to specify necessary parameters: mass `rho=2`, initialization values `k1=-4`, `k2=0`.

### 3.3 Computational Domain Construction

In this paper, the VIV problem acts on 100 discrete time points in $t \in [0.0625, 9.9375]$. These 100 time points have been saved in the file `examples/fsi/VIV_Training_Neta100.mat` as input data, so there is no need to explicitly construct the computational domain.

### 3.4 Constraint Construction

This paper uses supervised learning to constrain the model output $\eta$ and the lift $f$ calculated based on $\eta$.

#### 3.4.1 Supervised Constraint

Since we train in a supervised learning manner, we use supervised constraint `SupervisedConstraint` here:

``` py linenums="28"
--8<--
examples/fsi/viv.py:28:48
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraint, here fill in `train_dataloader_cfg` instantiated in [3.2 Equation Construction](#32) section;

The second parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is set to `"mean"`, that is, we will sum and average the loss terms generated by all data points participating in the calculation;

The third parameter is the equation expression, used to describe how to calculate the constraint target. Here fill in the calculation function of `eta` and `equation["VIV"].equations` instantiated in [3.2 Equation Construction](#32) section;

The fourth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing. Here we name it "Sup".

After the supervised constraint construction is completed, encapsulate it into a dictionary with the name we just named as the keyword for subsequent access.

``` py linenums="49"
--8<--
examples/fsi/viv.py:49:50
--8<--
```

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, we use 10000 training epochs, and evaluate the model accuracy every 10000 epochs.

``` yaml linenums="42"
--8<--
examples/fsi/conf/viv.yaml:42:49
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the commonly used `Adam` optimizer and `Step` interval decay learning rate are selected.

``` py linenums="52"
--8<--
examples/fsi/viv.py:52:54
--8<--
```

???+ note "Note"

    The VIV equation contains two **learnable parameters** k1 and k2, so the equation needs to be passed to the optimizer together with `model`.

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator.

``` py linenums="56"
--8<--
examples/fsi/viv.py:56:72
--8<--
```

Evaluation metric `metric` selects `ppsci.metric.MSE`;

Other configurations are similar to the settings in [3.4.1 Supervised Constraint Construction](#341).

### 3.8 Visualizer Construction

In model evaluation, if the evaluation result is data that can be visualized, we can choose a suitable visualizer to visualize the output result.

The data that needs to be visualized in this paper are two sets of relationship diagrams $t-\eta$ and $t-f$. Assuming that the coordinate of each moment $t$ is $t_i$, the corresponding network output is $\eta_i$, and the lift is $f_i$, so we only need to save all $(t_i, \eta_i, f_i)$ generated during the evaluation process as pictures. The code is as follows:

``` py linenums="74"
--8<--
examples/fsi/viv.py:74:93
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training, evaluation, and visualization.

``` py linenums="95"
--8<--
examples/fsi/viv.py:95:111
--8<--
```

## 4. Complete Code

``` py linenums="1" title="viv.py"
--8<--
examples/fsi/viv.py
--8<--
```

## 5. Result Display

The model prediction results are shown below. The horizontal axis is the time independent variable $t$, $\eta_{gt}$ is the reference amplitude, $\eta$ is the model predicted amplitude, $f_{gt}$ is the reference lift, and $f$ is the model predicted lift.

<figure markdown>
  ![Viv_result](https://paddle-org.bj.bcebos.com/paddlescience/docs/ViV/eta_f_pred.png){ loading=lazy }
  <figcaption> Prediction results and reference results of amplitude eta and lift f varying with time t</figcaption>
</figure>

It can be seen that the prediction results of amplitude and lift by the model in the time range $[0,10]$ are basically consistent with the reference results.
