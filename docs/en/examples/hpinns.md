# hPINNs(PINN with hard constraints)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6390502" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat --create-dirs -o ./datasets/hpinns_holo_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat --create-dirs -o ./datasets/hpinns_holo_valid.mat
    python holography.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat --create-dirs -o ./datasets/hpinns_holo_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat --create-dirs -o ./datasets/hpinns_holo_valid.mat
    python holography.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/hPINNs/hpinns_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python holography.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat --create-dirs -o ./datasets/hpinns_holo_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat --create-dirs -o ./datasets/hpinns_holo_valid.mat
    python holography.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [hpinns_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/hPINNs/hpinns_pretrained.pdparams) | loss(opt_sup): 0.05352<br>MSE.eval_metric(opt_sup): 0.00002<br>loss(val_sup): 0.02205<br>MSE.eval_metric(val_sup): 0.00001 |

## 1. Background Introduction

Solving partial differential equations (PDEs) is a fundamental physical problem. In the past few decades, various numerical solutions for systems of partial differential equations represented by finite difference method (FDM), finite volume method (FVM), and finite element method (FEM) have matured. With the rapid development of artificial intelligence technology, using deep learning to solve partial differential equations has become a new research trend. PINNs (Physics-informed neural networks) are deep learning networks that incorporate physical constraints. Therefore, compared with pure data-driven neural network learning, PINNs can learn models with stronger generalization ability using fewer data samples. Its application scope includes but is not limited to fluid mechanics, heat conduction, electromagnetic fields, quantum mechanics and other fields.

The constraints in traditional PINNs networks are soft constraints, that is, PDE (partial differential equation) participates in network training as a loss term. In this case, hPINNs strictly incorporate constraints into the network structure by modifying the network output, forming a more effective hard constraint.

At the same time, hPINNs designed different combinations of constraints and conducted experiments under 3 conditions: soft constraints, hard constraints with regularization, and hard constraints applying augmented Lagrangian method. This document mainly explains the hard constraints applying augmented Lagrangian method, but the three training modes can be switched through the `train_mode` parameter in the complete code.

For this problem, please refer to [AI Studio Project](https://aistudio.baidu.com/aistudio/projectdetail/4117361?channelType=0&channel=0).

## 2. Problem Definition

This problem uses hPINNs to solve the holography problem based on Fourier optics, aiming to design the permittivity map of the scattering plate. This method makes the propagation intensity of the scattered light of the permittivity map have the shape of the target function.

objective function:

$$
\begin{aligned}
\mathcal{J}(E) &= \dfrac{1}{Area(\Omega_3)} \left\| |E(x,y)|^2-f(x,y)\right\|^2_{2,\Omega_3} \\
&= \dfrac{1}{Area(\Omega_3)} \int_{\Omega_3} (|E(x,y)|^2-f(x,y))^2 {\rm d}x {\rm d}y
\end{aligned}
$$

Where E is the electric field intensity: $\vert E\vert^2 = (\mathfrak{R} [E])^2+(\mathfrak{I} [E])^2$

target function:

$$ f(x,y) =
\begin{cases}
\begin{aligned}
& 1, \ (x,y) \in [-0.5,0.5] \cap [1,2]\\
& 0, \ otherwise
\end{aligned}
\end{cases}
$$

PDE formula:

$$
\nabla^2 E + \varepsilon \omega^2 E = -i \omega \mathcal{J}
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods. In order to quickly understand PaddleScience, only key steps such as model construction and constraint construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset is a processed holography dataset, containing $x, y$ of training and test data, and the value $bound$ representing the boundary between optimizer area data and full area data, stored in the form of a dictionary in a `.mat` file.

Before running the code for this problem, please download the [training dataset](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat) and [validation dataset](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat) according to the following commands:

``` sh
wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat
wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat
```

### 3.2 Model Construction

The model structure diagram of the holography problem is:

<figure markdown>
  ![holography-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/holograpy_arch.png){ loading=lazy style="margin:0 auto"}
  <figcaption>hPINNs network model for holography problem</figcaption>
</figure>

In the holography problem, after applying the PMLs (perfectly matched layers) method, the PDE formula becomes:

$$
\dfrac{1}{1+i \dfrac{\sigma_x\left(x\right)}{\omega}} \dfrac{\partial}{\partial x} \left(\dfrac{1}{1+i \dfrac{\sigma_x\left(x\right)}{\omega}} \dfrac{\partial E}{\partial x}\right)+\dfrac{1}{1+i \dfrac{\sigma_y\left(y\right)}{\omega}} \dfrac{\partial}{\partial y} \left(\dfrac{1}{1+i \dfrac{\sigma_y\left(y\right)}{\omega}} \dfrac{\partial E}{\partial y}\right) + \varepsilon \omega^2 E = -i \omega \mathcal{J}
$$

For PMLs method, please refer to [Related Paper](https://arxiv.org/abs/2108.05348).

In this problem, the frequency $\omega$ is a constant $\dfrac{2\pi}{\mathcal{P}}$ ($\mathcal{P}$ is Period), the unknown quantity $E$ to be solved is related to the position parameter $(x, y)$. In this example, the permittivity $\varepsilon$ is also an unknown quantity, $\sigma_x(x)$ and $\sigma_y(y)$ are variables related to $x, y$ respectively obtained by PMLs. Here we use a relatively simple MLP (Multilayer Perceptron) to represent the mapping function $f: \mathbb{R}^2 \to \mathbb{R}^2$ from $(x, y)$ to $(E, \varepsilon)$. However, as shown in the network structure above, in this problem $E$ is divided into two parts $(\mathfrak{R} [E],\mathfrak{I} [E])$ according to the real part and imaginary part, and 3 parallel MLP networks are used to map $(\mathfrak{R} [E], \mathfrak{I} [E], \varepsilon)$ respectively. The mapping function is $f_i: \mathbb{R}^2 \to \mathbb{R}^1$, that is:

$$
\mathfrak{R} [E] = f_1(x,y), \ \mathfrak{R} [E] = f_2(x,y), \ \varepsilon = f_3(x,y)
$$

In the above formula, $f_1, f_2, f_3$ are each an MLP model, and the three together form a Model List, expressed in PaddleScience code as follows

``` py linenums="42"
--8<--
examples/hpinns/holography.py:42:44
--8<--
```

In order to access the values of specific variables accurately and quickly during calculation, we specify here that the input variable names of the network model are `("x_cos_1","x_sin_1",...,"x_cos_6","x_sin_6","y","y_cos_1","y_sin_1")`, and the output variable names are `("e_re",)`, `("e_im",)`, `("eps",)` respectively.
Note that the input variables here are much more than the two variables $(x, y)$, because as shown in the figure above, the input of the model is actually the terms of the Fourier expansion of $(x, y)$ rather than themselves. The training data provided in the dataset are $(x, y)$ values, which means we need to transform the input. At the same time, as shown in the figure above, due to the existence of hard constraints, the output variable name of the model is not the final output, so the output also needs to be transformed.

### 3.3 transform Construction

The transform of the input is the transformation of variables $(x, y)$ to $(\cos(\omega x),\sin(\omega x),...,\cos(6 \omega x),\sin(6 \omega x),y,\cos(\omega y),\sin(\omega y))$, and the output transform is the hard constraint on $(\mathfrak{R} [E], \mathfrak{I} [E], \varepsilon)$ respectively. The code is as follows

``` py linenums="49"
--8<--
examples/hpinns/functions.py:49:92
--8<--
```

The corresponding transform needs to be registered for each MLP model separately, and then the 3 MLP models are formed into a Model List.

``` py linenums="50"
--8<--
examples/hpinns/holography.py:50:59
--8<--
```

In this way, we instantiated a neural network model `model list` containing 3 MLP models, each MLP containing 4 layers of hidden neurons, each layer having 48 neurons, using "tanh" as the activation function, and containing input and output transforms.

### 3.4 Parameter and Hyperparameter Setting

We need to specify problem-related parameters, such as specifying training with hard constraints applying augmented Lagrangian method through the `train_mode` parameter.

``` py linenums="35"
--8<--
examples/hpinns/holography.py:35:40
--8<--
```

``` py linenums="46"
--8<--
examples/hpinns/holography.py:46:48
--8<--
```

``` py linenums="28"
--8<--
examples/hpinns/functions.py:28:46
--8<--
```

Since the augmented Lagrangian method is applied, parameters $\mu$ and $\lambda$ are not constants, but change with training round $k$. At this time, $\beta$ is the coefficient of change, that is, every training round

$\mu_k = \beta \mu_{k-1}$, $\lambda_k = \beta \lambda_{k-1}$

At the same time, hyperparameters such as training epochs and learning rate need to be specified.

``` yaml linenums="53"
--8<--
examples/hpinns/conf/hpinns.yaml:53:61
--8<--
```

### 3.5 Optimizer Construction

The training is divided into two stages. First, use the Adam optimizer for rough training, and then use the LBFGS optimizer to approximate the optimal point. Therefore, two optimizers are required, which also corresponds to the two `EPOCHS` values in the hyperparameters in the previous section.

``` py linenums="62"
--8<--
examples/hpinns/holography.py:62:64
--8<--
```

``` py linenums="203"
--8<--
examples/hpinns/holography.py:203:205
--8<--
```

### 3.6 Constraint Construction

This problem adopts unsupervised learning, and the constraint is that the result needs to satisfy the PDE formula.

Although we are not training in a supervised learning manner, we can still use the supervised constraint `SupervisedConstraint`. Before defining constraints, we need to specify data reading configurations such as file paths for supervised constraints. Because there is no label data in the dataset, we need to use training data as label data when reading data, and pay attention not to use this part of "fake" label data later.

``` py linenums="102"
--8<--
examples/hpinns/holography.py:102:107
--8<--
```

As above, all output labels will read the value of input `x`.

The following are specific contents such as constraints. Pay attention to the "fake" label data given as mentioned above:

``` py linenums="66"
--8<--
examples/hpinns/holography.py:66:127
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraints, where the `"dataset"` field represents the training dataset information used, and each field represents:

1. `name`: Dataset type, here `"IterableMatDataset"` represents `.mat` type dataset read sequentially without batching;
2. `file_path`: Dataset file path;
3. `input_keys`: Input variable name;
4. `label_keys`: Label variable name;
5. `alias_dict`: Variable alias.

The second parameter is the loss function. Here `FunctionalLoss` is a custom loss function class reserved by PaddleScience. This class supports defining the calculation method of loss when writing code, rather than using existing methods such as `MSE`. In this problem, since there are multiple loss terms, multiple loss calculation functions need to be defined, which is also the reason why multiple constraints need to be constructed. For custom loss function code, please refer to [Custom loss and metric](#38-custom-loss-and-metric).

The third parameter is the equation expression, used to describe how to calculate the constraint target. Here fill in `output_expr`. The calculated value will be stored in the output list according to the specified name, so as to ensure that these values can be used when calculating loss.

The fourth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing.

After the constraint is constructed, encapsulate it into a dictionary with the name we just named as the key for subsequent access.

``` py linenums="128"
--8<--
examples/hpinns/holography.py:128:131
--8<--
```

### 3.7 Validator Construction

Similar to constraints, although this problem uses unsupervised learning, `ppsci.validate.SupervisedValidator` can still be used to construct a validator. There are two sampling point regions in this problem, one is a larger complete definition region, and the other is an objective region in the domain. The validator evaluates these two regions separately, so two validators need to be constructed. `opt` corresponds to the objective region, and `val` corresponds to the entire domain.

``` py linenums="133"
--8<--
examples/hpinns/holography.py:133:181
--8<--
```

The evaluation metric `metric` is `FunctionalMetric`, which is a custom metric function class reserved by PaddleScience. This class supports defining the calculation method of metric when writing code, rather than using existing methods such as `MSE`, `L2`, etc. For custom metric function code, please refer to the next section [Custom loss and metric](#38-custom-loss-and-metric).

Other configurations are similar to the settings of [Constraint Construction](#36).

### 3.8 Custom loss and metric

Since this problem adopts unsupervised learning and there is no label data in the data, loss and metric are calculated based on PDE, so custom loss and metric are required. The method is to first define relevant functions, and then pass the function name as a parameter to `FunctionalLoss` and `FunctionalMetric`.

Note that the input and output parameters of custom loss and metric functions need to be consistent with other functions such as `MSE` in PaddleScience, that is, input is model output `output_dict` and other dictionary variables, loss function output is loss value `paddle.Tensor`, metric function output is dictionary `Dict[str, paddle.Tensor]`.

``` py linenums="237"
--8<--
examples/hpinns/functions.py:237:317
--8<--
```

``` py linenums="320"
--8<--
examples/hpinns/functions.py:320:336
--8<--
```

### 3.9 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="183"
--8<--
examples/hpinns/holography.py:183:200
--8<--
```

Since there are multiple training modes in this problem, $[2,1+k]$ complete training and evaluations will be performed according to different modes. For specific code, please refer to the holography.py file in [Complete Code](#4).

### 3.10 Visualization

PaddleScience provides a visualizer, but due to the large number of pictures and complexity of this problem, a visualization function is customized in the code. Visualization can be achieved by calling the custom function.

``` py linenums="279"
--8<--
examples/hpinns/holography.py:279:292
--8<--
```

For custom code, please refer to the plotting.py file in [Complete Code](#4).

## 4. Complete Code

The complete code includes PaddleScience specific implementation process code holography.py, all custom function code functions.py and custom visualization code plotting.py.

``` py linenums="1" title="holography.py"
--8<--
examples/hpinns/holography.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/hpinns/functions.py
--8<--
```

``` py linenums="1" title="plotting.py"
--8<--
examples/hpinns/plotting.py
--8<--
```

## 5. Result Display

Refer to [Problem Definition](#2), the following figure shows the changes in loss during training, changes in parameter lambda and parameter mu with training round k in the augmented Lagrangian method, and the final predicted values of electric field E and permittivity epsilon.

The figure below shows the prediction of electromagnetic wave propagation within a defined square domain. The prediction results are basically consistent with the results of the finite difference frequency domain (FDFD) method.

Loss value changes during training:

<figure markdown>
  ![holograpy_result_6A](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_A.jpg){ loading=lazy }
  <figcaption> Loss value change with iteration during training</figcaption>
</figure>

Objective loss value changes with training round k:
<figure markdown>
  ![holograpy_result_6B](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_B.jpg){ loading=lazy }
  <figcaption> k value corresponds to objective loss value</figcaption>
</figure>

Values of the real and imaginary parts of parameter lambda when k=1, 4, 9:
<figure markdown>
  ![holograpy_result_6C](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_C.jpg){ loading=lazy }
  <figcaption> lambda value when k=1, 4, 9</figcaption>
</figure>

Ratio of parameter lambda to parameter mu changes with training round k:
<figure markdown>
  ![holograpy_result_6D](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_D.jpg){ loading=lazy }
  <figcaption> k value corresponds to lambda/mu value</figcaption>
</figure>

Frequency of occurrence of the ratio of the real parts of parameter lambda and parameter mu with training rounds k=1, 4, 6, 9. The "sharper" the curve, the more uniform the values tend to be, and the better the convergence:
<figure markdown>
  ![holograpy_result_6E](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_E.jpg){ loading=lazy }
  <figcaption> Frequency of real part lambda/mu value when k=1, 4, 6, 9</figcaption>
</figure>

Frequency of occurrence of the ratio of the imaginary parts of parameter lambda and parameter mu with training rounds k=1, 4, 6, 9. The "sharper" the curve, the more uniform the values tend to be, and the better the convergence:
<figure markdown>
  ![holograpy_result_6F](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig6_F.jpg){ loading=lazy }
  <figcaption> Frequency of imaginary part lambda/mu value when k=1, 4, 6, 9</figcaption>
</figure>

Electric field E value:
<figure markdown>
  ![holograpy_result_7C](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig7_C.jpg){ loading=lazy }
  <figcaption> E value</figcaption>
</figure>

Permittivity epsilon value:
<figure markdown>
  ![holograpy_result_7eps](https://paddle-org.bj.bcebos.com/paddlescience/docs/hPINNs/aug_lag_Fig7_eps.jpg){ loading=lazy }
  <figcaption> epsilon value</figcaption>
</figure>

## 6. References

- [PHYSICS-INFORMED NEURAL NETWORKS WITH HARD CONSTRAINTS FOR INVERSE DESIGN](https://arxiv.org/pdf/2102.04626.pdf)

- [Reference Code](https://github.com/lululxvi/hpinn)
