# Volterra integral equation

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6622866?sUid=438690&shared=1&ts=1691582831601" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    python volterra_ide.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python volterra_ide.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/volterra_ide/volterra_ide_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python volterra_ide.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python volterra_ide.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [volterra_ide_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/volterra_ide/volterra_ide_pretrained.pdparams) | loss(L2Rel_Validator): 0.00023 <br> L2Rel.u(L2Rel_Validator): 0.00023 |

## 1. Background Introduction

Volterra integral equation is an integral equation, that is, the equation contains the integral operation of the function to be solved. It has two forms, as shown below

$$
\begin{aligned}
  f(t) &= \int_a^t K(t, s) x(s) d s \\
  x(t) &= f(t)+\int_a^t K(t, s) x(s) d s
\end{aligned}
$$

In the field of mathematics, the Volterra equation can be used to express various multivariate probability distributions and is a powerful tool for multivariate statistical analysis. This makes it very useful when dealing with complex data structures, such as in the field of machine learning. The Volterra equation can also be used to calculate the correlation of attributes of different dimensions and simulate complex dataset structures to provide effective data support for machine learning tasks.

In the field of biology, the Volterra equation is used as a guide for fishery production and is of great significance to ecological balance and environmental protection. In addition, the equation also has applications in disease prevention and control, population statistics, etc. It is worth mentioning that the establishment of the Volterra equation was the first successful attempt to apply mathematics in the field of biology, promoting the emergence and development of the science of biomathematics.

This case takes the second equation as an example and uses deep learning for solving.

## 2. Problem Definition

Assume that there is the following IDE equation:

$$
u(t) = -\dfrac{du}{dt} + \int_{t_0}^t e^{t-s} u(s) d s
$$

Where $u(t)$ is the function to be solved, $-\dfrac{du}{dt}$ corresponds to $f(t)$, and $e^{t-s}$ corresponds to $K(t,s)$.
Therefore, a neural network model can be used, with $t$ as input and $u(t)$ as output, construct differential constraints according to the above equation, and perform unsupervised learning to finally fit the function $u(t)$ to be solved.

In order to facilitate solving in a computer, we move the terms of the above formula, putting the integral term on the left side and the non-integral term on the right side, as shown below:

$$
\int_{t_0}^t e^{t-s} u(s) d s = u(t) + \dfrac{du}{dt}
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the above problem, we determined that the input is $x$ and the output is $u(x)$, so we use, expressed in PaddleScience code as follows:

``` py linenums="39"
--8<--
examples/ide/volterra_ide.py:39:40
--8<--
```

In order to accurately and quickly access the value of specific variables during calculation, we specify here that the input variable name of the network model is `"x"` (i.e. $t$ in the formula), and the output variable name is `"u"`. Then by specifying the number of hidden layers and number of neurons of `MLP`, we instantiate the neural network model `model`.

### 3.2 Computational Domain Construction

The integration domain of the Volterra_IDE problem is $a$ ~ $t$, where `a` is a fixed constant 0, and the range of `t` is 0 ~ 5, so the built-in one-dimensional geometry `TimeDomain` of PaddleScience can be used as the computational domain.

``` py linenums="42"
--8<--
examples/ide/volterra_ide.py:42:43
--8<--
```

### 3.3 Equation Construction

Since Volterra_IDE uses an integral equation, `ppsci.equation.Volterra` built into PaddleScience can be used directly, and specify the required parameters: lower limit of integration `a`, number of discrete points of `t` `num_points`, number of one-dimensional Gaussian integration points `quad_deg`, $K(t,s)$ kernel function `kernel_func`, $u(t) - f(t)$ right side expression of the equation `func`.

``` py linenums="45"
--8<--
examples/ide/volterra_ide.py:45:61
--8<--
```

### 3.4 Constraint Construction

#### 3.4.1 Interior Point Constraint

This paper uses unsupervised learning to constrain the left and right sides of the equation after moving terms to be as equal as possible.

Since the left side of the equation involves integral calculation (actually using Gaussian integration approximate calculation), after sampling multiple `t_i` points in the 0 ~ 5 interval, it is also necessary to calculate the point set used for Gaussian integration, that is, for each `(0, t_i)` interval, calculate the one-to-one corresponding Gaussian integration point set `quad_i` and point weight `weight_i`. PaddleScience adds this step as preprocessing of input data to the code, as shown below

``` py linenums="63"
--8<--
examples/ide/volterra_ide.py:63:117
--8<--
```

#### 3.4.2 Initial Value Constraint

At $t=0$, there are the following initial value conditions:

$$
u(0) = e^{-t} \cosh(t)|_{t=0} = e^{0} \cosh(0) = 1
$$

Therefore, the initial value condition at `t=0` can be added, and the code is as follows

``` py linenums="119"
--8<--
examples/ide/volterra_ide.py:119:137
--8<--
```

After the differential equation constraint and initial value constraint are constructed, encapsulate them into a dictionary with the name we just named as the keyword for subsequent access.

``` py linenums="138"
--8<--
examples/ide/volterra_ide.py:138:142
--8<--
```

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, let the `L-BFGS` optimizer perform one round of optimization, but the number of `max_iters` in one round of optimization can be set to a larger number `15000`.

``` yaml linenums="39"
--8<--
examples/ide/conf/volterra_ide.yaml:39:57
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the commonly used `LBFGS` optimizer is selected.

``` py linenums="144"
--8<--
examples/ide/volterra_ide.py:144:145
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="147"
--8<--
examples/ide/volterra_ide.py:147:161
--8<--
```

Evaluation metric `metric` selects `ppsci.metric.L2Rel`;

Other configurations are similar to the settings in [3.4 Constraint Construction](#34).

### 3.8 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="163"
--8<--
examples/ide/volterra_ide.py:163:181
--8<--
```

### 3.9 Result Visualization

After the model training is completed, we can manually construct 100 points uniformly in the 0 ~ 5 interval as the integration upper limit `t` for evaluation to predict and visualize the results.

``` py linenums="183"
--8<--
examples/ide/volterra_ide.py:183:194
--8<--
```

## 4. Complete Code

``` py linenums="1" title="volterra_ide.py"
--8<--
examples/ide/volterra_ide.py
--8<--
```

## 5. Result Display

The model prediction results are shown below. $t$ is the independent variable, $u(t)$ is the standard solution function of the integral equation, and $\hat{u}(t)$ is the model predicted integral equation solution function

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/Volterra_IDE/Volterra_IDE.png){ loading=lazy }
  <figcaption>Model solution result (orange scatter) and reference result (blue curve)</figcaption>
</figure>

It can be seen that the model's prediction result $\hat{u}(t)$ for the integral equation in the $[0,5]$ interval is basically consistent with the standard solution result $u(t)$.

## 6. References

- [DeepXDE - Antiderivative operator from an unaligned dataset](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py)
- [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval)
- [Volterra integral equation](https://en.wikipedia.org/wiki/Volterra_integral_equation)
