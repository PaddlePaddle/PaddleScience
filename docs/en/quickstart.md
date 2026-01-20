# Quick Start

<a href="https://aistudio.baidu.com/projectdetail/6665190?contributionType=1&sUid=438690&shared=1&ts=1692616326196" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

This article introduces how to use PaddleScience to train a model, solve a class of equation learning and prediction problems, and visualize prediction results through a simple demo and its extended problems.

## 1. Problem Introduction

Suppose we want to use a neural network model to fit the function $u=\sin(x)$ in the interval $x \in [-\pi, \pi]$. In two scenarios where the fitted function is known and unknown, how to fit $u=\sin(x)$ as accurately as possible.

In the first scenario, assuming that the analytical solution of the target function $u$ is known to be $u=\sin(x)$, we adopt the idea of supervised training, directly generating the label dependent variable $u$ using this formula, and training the model together with the independent variable $x$ as supervised data.

In the second scenario, assuming that the analytical solution of the target function $u$ is unknown, but we know that it satisfies a certain differential relationship, we take one of the differential equations satisfying the condition $\dfrac{\partial u} {\partial x}=\cos(x)$ as an example to introduce how to generate data for training.

## 2. Scenario 1

Target fitted function:

$$
u=\sin(x), x \in [-\pi, \pi].
$$

We generate $N$ pairs of data $(x_i, u_i), i=1,...,N$ as supervised data for training.

Before writing the code, we first import the necessary packages.

``` py linenums="1"
--8<--
examples/quick_start/case1.py:1:4
--8<--
```

Then create log and model saving directories for training process recording and saving, which is an operation that needs to be performed before most examples officially start.

``` py linenums="6"
--8<--
examples/quick_start/case1.py:6:13
--8<--
```

Next, officially start writing code.

First define the problem interval. We use `ppsci.geometry.Interval` to define a line segment geometry shape to facilitate subsequent sampling of $x$ on this line segment.

``` py linenums="15"
--8<--
examples/quick_start/case1.py:15:17
--8<--
```

Then define a simple 3-layer MLP model.

``` py linenums="19"
--8<--
examples/quick_start/case1.py:19:20
--8<--
```

The above code indicates that the model accepts the independent variable $x$ as input and outputs the prediction result $\hat{u}$.

Then we define the calculation function of the known $u=\sin(x)$ as a parameter of `ppsci.constraint.InteriorConstraint`, which is used to calculate label data. `InteriorConstraint` means taking the data in the given geometric shape or dataset as input, combined with the given label data, to guide the model for optimization.

``` py linenums="22"
--8<--
examples/quick_start/case1.py:22:47
--8<--
```

Here `interior_constraint` represents a training objective, that is, we hope that in the interval $[-\pi, \pi]$, the model is optimized so that the model's prediction result $\hat{u}$ is as close as possible to its label value $u$.

Next, we can start defining model training related content, such as training epochs, optimizer, and visualizer.

``` py linenums="48"
--8<--
examples/quick_start/case1.py:48:66
--8<--
```

Finally, pass the objects defined above to the training scheduling class `Solver` to start model training.

``` py linenums="67"
--8<--
examples/quick_start/case1.py:67:79
--8<--
```

After training, calculate the L2-relative error with the standard solution using the 1000 points just taken.

``` py linenums="81"
--8<--
examples/quick_start/case1.py:81:86
--8<--
```

Then visualize the prediction results of these 1000 points.

``` py linenums="88"
--8<--
examples/quick_start/case1.py:88:89
--8<--
```

The training record is shown below.

``` log
...
...
ppsci INFO: [Train][Epoch  9/10][Iter  80/100] lr: 0.00200, loss: 0.00663, EQ: 0.00663, batch_cost: 0.00180s, reader_cost: 0.00011s, ips: 17756.64, eta: 0:00:00
ppsci INFO: [Train][Epoch  9/10][Iter  90/100] lr: 0.00200, loss: 0.00598, EQ: 0.00598, batch_cost: 0.00180s, reader_cost: 0.00011s, ips: 17793.97, eta: 0:00:00
ppsci INFO: [Train][Epoch  9/10][Iter 100/100] lr: 0.00200, loss: 0.00547, EQ: 0.00547, batch_cost: 0.00179s, reader_cost: 0.00011s, ips: 17864.08, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  10/100] lr: 0.00200, loss: 0.00079, EQ: 0.00079, batch_cost: 0.00182s, reader_cost: 0.00012s, ips: 17547.05, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  20/100] lr: 0.00200, loss: 0.00075, EQ: 0.00075, batch_cost: 0.00183s, reader_cost: 0.00011s, ips: 17482.92, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  30/100] lr: 0.00200, loss: 0.00077, EQ: 0.00077, batch_cost: 0.00182s, reader_cost: 0.00011s, ips: 17539.51, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  40/100] lr: 0.00200, loss: 0.00074, EQ: 0.00074, batch_cost: 0.00182s, reader_cost: 0.00011s, ips: 17587.51, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  50/100] lr: 0.00200, loss: 0.00071, EQ: 0.00071, batch_cost: 0.00182s, reader_cost: 0.00011s, ips: 17563.59, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  60/100] lr: 0.00200, loss: 0.00070, EQ: 0.00070, batch_cost: 0.00182s, reader_cost: 0.00011s, ips: 17604.60, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  70/100] lr: 0.00200, loss: 0.00074, EQ: 0.00074, batch_cost: 0.00181s, reader_cost: 0.00011s, ips: 17699.28, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  80/100] lr: 0.00200, loss: 0.00077, EQ: 0.00077, batch_cost: 0.00180s, reader_cost: 0.00011s, ips: 17764.92, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  90/100] lr: 0.00200, loss: 0.00075, EQ: 0.00075, batch_cost: 0.00180s, reader_cost: 0.00011s, ips: 17795.87, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter 100/100] lr: 0.00200, loss: 0.00071, EQ: 0.00071, batch_cost: 0.00179s, reader_cost: 0.00011s, ips: 17872.00, eta: 0:00:00
```

After training, calculate the L2-relative error with the standard solution using the 1000 points just taken.

``` py linenums="81"
--8<--
examples/quick_start/case1.py:81:86
--8<--
```

It can be seen that using the standard solution to supervise the training model, there is still good prediction ability near the standard solution, and the L2-relative error is 0.02677.

The prediction result visualization is shown below.

![u=sin(x) prediction](../images/quickstart/u_pred_case1.png)

The complete code for Scenario 1 is shown below.

``` py linenums="1" title="examples/quick_start/case1.py"
--8<--
examples/quick_start/case1.py
--8<--
```

## 3. Scenario 2

It can be seen that the supervised training method in Scenario 1 can solve the function fitting problem well, but in general we cannot know the analytical expression of the fitted function itself, so we cannot directly construct supervised data for the dependent variable.

Although the analytical formula cannot be calculated to directly construct supervised data, it is often possible to use relevant mathematical knowledge to derive a certain mathematical relationship that the target fitting function conforms to, and achieve the purpose of optimizing the model by means of "indirect supervision" by training the model to satisfy this mathematical relationship.

Suppose we no longer use the prior formula $u=\sin(x)$, and thus cannot calculate the label data $u$. Therefore, the following equation system is used, which contains a partial differential equation and boundary conditions:

$$
\begin{cases}
\begin{aligned}
    \dfrac{\partial u} {\partial x} &= \cos(x) \\
    u(-\pi) &= 2
\end{aligned}
\end{cases}
$$

Construct data pairs $(x_i, \cos(x_i)), i=1,...,N$.
This means that we can still keep the input and output of the model unchanged, but the optimization objective becomes: let $\dfrac{\partial \hat{u}} {\partial x}$ be as close as possible to $\cos(x)$, and $\hat{u}(-\pi)$ should also be as close as possible to $2$.

Based on the above theory, we can obtain the code for Scenario 2 by slightly rewriting the code for Scenario 1.

First, since we need to use the first-order differential operation, we need to import the first-order differential API at the beginning of the code.

``` py linenums="1" hl_lines="4"
--8<--
examples/quick_start/case2.py:1:5
--8<--
```

Then add a differential label value calculation function below the original label calculation function.

``` py linenums="28" hl_lines="4"
--8<--
examples/quick_start/case2.py:28:30
--8<--
```

Then change the constraint condition `interior_constraint` from constraining "model output" to constraining "first-order differential of model output with respect to input".

``` py linenums="33" hl_lines="4"
--8<--
examples/quick_start/case2.py:33:49
--8<--
```

Considering that in general cases, the solution of partial differential equations will have undetermined coefficients, which need to be determined by definite conditions (initial (boundary) value conditions), an additional boundary condition constraint `bc_constraint` needs to be added after the `interior_constraint` construction code, as shown below.

``` py linenums="50"
--8<--
examples/quick_start/case2.py:50:65
--8<--
```

1. Corresponding boundary condition $u(x_0)=sin(x_0)+2$

Then add the boundary constraint `bc_constraint` to `constraint`.

``` py linenums="66" hl_lines="4"
--8<--
examples/quick_start/case2.py:66:70
--8<--
```

Similarly, modify the standard solution drawn by Visualizer to $sin(x)+2$.

``` py linenums="77" hl_lines="5"
--8<--
examples/quick_start/case2.py:77:89
--8<--
```

Execute training after modification.

``` py linenums="91"
--8<--
examples/quick_start/case2.py:91:102
--8<--
```

The training log is shown below.

``` log
...
...
ppsci INFO: [Train][Epoch  9/10][Iter  90/100] lr: 0.00200, loss: 0.00176, EQ: 0.00087, BC: 0.00088, batch_cost: 0.00346s, reader_cost: 0.00024s, ips: 9527.80, eta: 0:00:00
ppsci INFO: [Train][Epoch  9/10][Iter 100/100] lr: 0.00200, loss: 0.00170, EQ: 0.00087, BC: 0.00083, batch_cost: 0.00349s, reader_cost: 0.00024s, ips: 9452.07, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  10/100] lr: 0.00200, loss: 0.00107, EQ: 0.00072, BC: 0.00035, batch_cost: 0.00350s, reader_cost: 0.00025s, ips: 9424.75, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  20/100] lr: 0.00200, loss: 0.00116, EQ: 0.00083, BC: 0.00033, batch_cost: 0.00350s, reader_cost: 0.00025s, ips: 9441.33, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  30/100] lr: 0.00200, loss: 0.00103, EQ: 0.00079, BC: 0.00024, batch_cost: 0.00355s, reader_cost: 0.00025s, ips: 9291.90, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  40/100] lr: 0.00200, loss: 0.00108, EQ: 0.00078, BC: 0.00030, batch_cost: 0.00353s, reader_cost: 0.00025s, ips: 9348.09, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  50/100] lr: 0.00200, loss: 0.00163, EQ: 0.00082, BC: 0.00082, batch_cost: 0.00350s, reader_cost: 0.00024s, ips: 9416.24, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  60/100] lr: 0.00200, loss: 0.00160, EQ: 0.00083, BC: 0.00077, batch_cost: 0.00353s, reader_cost: 0.00024s, ips: 9345.73, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  70/100] lr: 0.00200, loss: 0.00150, EQ: 0.00082, BC: 0.00068, batch_cost: 0.00351s, reader_cost: 0.00024s, ips: 9393.89, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  80/100] lr: 0.00200, loss: 0.00146, EQ: 0.00081, BC: 0.00064, batch_cost: 0.00350s, reader_cost: 0.00024s, ips: 9424.81, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter  90/100] lr: 0.00200, loss: 0.00138, EQ: 0.00081, BC: 0.00058, batch_cost: 0.00349s, reader_cost: 0.00024s, ips: 9444.12, eta: 0:00:00
ppsci INFO: [Train][Epoch 10/10][Iter 100/100] lr: 0.00200, loss: 0.00133, EQ: 0.00079, BC: 0.00054, batch_cost: 0.00349s, reader_cost: 0.00024s, ips: 9461.54, eta: 0:00:00
```

After training, calculate the L2-relative error with the standard solution using the 1000 points just taken.

``` py linenums="104"
--8<--
examples/quick_start/case2.py:104:109
--8<--
```

It can be seen that the model trained by the differential equation still has good prediction ability near the standard solution, and the L2-relative error is 0.00564.

The prediction result visualization is shown below.

![u=sin(x)+2 prediction](../images/quickstart/u_pred_case2.png)

It can be found that the model trained by using the differential relationship still has good predictive ability, and combined with the definite condition, it can learn the correct solution model that conforms to both the differential equation and the definite condition.

The complete code for Scenario 2 is shown below.

``` py linenums="1"
--8<--
examples/quick_start/case2.py
--8<--
```
