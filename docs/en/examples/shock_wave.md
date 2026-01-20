# Shock Wave

<a href="https://aistudio.baidu.com/projectdetail/6755993?contributionType=1&sUid=438690&shared=1&ts=1694949960479" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    === "Ma=2.0"

        ``` sh
        python shock_wave.py
        ```
    === "Ma=0.728"

        ``` sh
        python shock_wave.py -cn=shock_wave_Ma0.728
        ```

=== "Model Prediction Command"

    === "Ma=2.0"

        ``` sh
        python shock_wave.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/shockwave/shock_wave_Ma2_pretrained.pdparams
        ```
    === "Ma=0.728"

        ``` sh
        python shock_wave.py -cn=shock_wave_Ma0.728 mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/shockwave/shock_wave_Ma0728_pretrained.pdparams
        ```

=== "Model Export Command"

    === "Ma=2.0"

        ``` sh
        python shock_wave.py mode=export
        ```
    === "Ma=0.728"

        ``` sh
        python shock_wave.py -cn=shock_wave_Ma0.728 mode=export
        ```

=== "Model Inference Command"

    === "Ma=2.0"

        ``` sh
        python shock_wave.py mode=infer
        ```
    === "Ma=0.728"

        ``` sh
        python shock_wave.py -cn=shock_wave_Ma0.728 mode=infer
        ```

## 1. Background Introduction

Shock waves are a phenomenon frequently found in nature and engineering applications. They not only widely exist in compressible flows in the aerospace field, but also appear in other fields such as theoretical and applied physics and engineering applications. In supersonic and hypersonic flows, the appearance of shock waves will have a significant impact on the overall characteristics of fluid flow. The shock wave capturing problem has been developed in the CFD field for decades. Shock wave capturing methods based on the mathematical theory of weak solutions have developed rapidly due to their simplicity and ease of implementation, and have been widely used in numerical simulations of complex supersonic and hypersonic flows.

This case optimizes the PINN-WE model so that it can be applied to flow field simulations with strong shock waves such as supersonic and hypersonic flows.

The PINN-WE model reduces the fitting of strong gradient regions during PINN optimization through loss function weighting, avoiding the shock wave overfitting problem caused by strong gradients in shock wave regions. It has achieved good results in one-dimensional Euler problems and two-dimensional problems with weak shock waves. However, in supersonic two-dimensional flow fields, this model did not achieve very good results. In experiments, it was also found that the model often produces non-physical prediction results such as shock wave position deviation and asymmetric shock wave shape. Therefore, aiming at this problem of the above PINN-WE model, this case proposes the idea of progressive weighting, abandoning the idea of emphasizing gradients during the optimization process, but innovatively gradually strengthening the influence of gradient weights on model optimization, so that the model can obtain a better and physically consistent shock wave position during the optimization process.

## 2. Problem Definition

This problem simulates a cylindrical bow shock wave in a two-dimensional supersonic flow field, involving the two-dimensional Euler equations, as shown below:

$$
\begin{array}{cc}
  \dfrac{\partial \hat{U}}{\partial t}+\dfrac{\partial \hat{F}}{\partial \xi}+\dfrac{\partial \hat{G}}{\partial \eta}=0 \\
  \text { Where, } \quad
  \begin{cases}
    \hat{U}=J U \\
    \hat{F}=J\left(F \xi_x+G \xi_y\right) \\
    \hat{G}=J\left(F \eta_x+G \eta_y\right)
  \end{cases} \\
  U=\left(\begin{array}{l}
  \rho \\
  \rho u \\
  \rho v \\
  E
  \end{array}\right), \quad F=\left(\begin{array}{l}
  \rho u \\
  \rho u^2+p \\
  \rho u v \\
  (E+p) u
  \end{array}\right), \quad G=\left(\begin{array}{l}
  \rho v \\
  \rho v u \\
  \rho v^2+p \\
  (E+p) v
  \end{array}\right)
\end{array}
$$

Free stream conditions $\rho_{\infty}=1.225 \mathrm{~kg} / \mathrm{m}^3$ ; $P_{\infty}=1 \mathrm{~atm}$

The overall process is shown below:

![computation_progress](https://paddle-org.bj.bcebos.com/paddlescience/docs/ShockWave/computation_progress.png)

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the ShockWave problem, given time $t$ and position coordinates $(x,y)$, the model is responsible for predicting the corresponding four physical quantities: $x$ direction velocity, $y$ direction velocity, pressure, and density $(u,v,p,\rho)$. Therefore, we use a relatively simple MLP (Multilayer Perceptron) here to represent the mapping function $g: \mathbb{R}^3 \to \mathbb{R}^4$ from $(t,x,y)$ to $(u,v,p,\rho)$, that is:

$$
u,v,p,\rho = g(t,x,y)
$$

In the above formula, $g$ is the MLP model itself, expressed in PaddleScience code as follows

``` py linenums="254"
--8<--
examples/shock_wave/shock_wave.py:254:255
--8<--
```

In order to accurately and quickly access the value of specific variables during calculation, we specify here that the input variable names of the network model are `("t", "x", "y")` and the output variable names are `("u", "v", "p", "rho")`. These names are consistent with subsequent code.

Then by specifying the number of layers, number of neurons, and activation function of the MLP, we instantiate a neural network model `model` with 9 hidden layers, 90 neurons per layer, using "tanh" as the activation function.

### 3.2 Equation Construction

This case involves two-dimensional Euler equations and equations on the boundary, as shown below

``` py linenums="31"
--8<--
examples/shock_wave/shock_wave.py:31:217
--8<--
```

``` py linenums="257"
--8<--
examples/shock_wave/shock_wave.py:257:258
--8<--
```

### 3.3 Computational Domain Construction

The computational domain of this case is 0 ~ 0.4 unit time, a rectangular area with a length of 1.5 and a width of 2.0, containing a circle with center coordinates [1, 1] and radius 0.25. The code is as follows

``` yaml linenums="31"
--8<--
examples/shock_wave/conf/shock_wave_Ma2.0.yaml:31:43
--8<--
```

### 3.4 Constraint Construction

#### 3.4.1 Interior Point Constraint

We apply the Euler equations to the interior points of the computational domain, and use the Latin HyperCube Sampling (LHS) method to sample a total of `N_INTERIOR` training points. The code is as follows:

``` yaml linenums="38"
--8<--
examples/shock_wave/conf/shock_wave_Ma2.0.yaml:38:38
--8<--
```

``` py linenums="260"
--8<--
examples/shock_wave/shock_wave.py:260:276
--8<--
```

``` py linenums="339"
--8<--
examples/shock_wave/shock_wave.py:339:352
--8<--
```

#### 3.4.2 Boundary Constraint

We apply boundary conditions to the boundary points of the computational domain, and also use the Latin HyperCube Sampling (LHS) method to sample a total of `N_BOUNDARY` training points on the boundary. The code is as follows:

``` yaml linenums="39"
--8<--
examples/shock_wave/conf/shock_wave_Ma2.0.yaml:39:39
--8<--
```

``` py linenums="278"
--8<--
examples/shock_wave/shock_wave.py:278:311
--8<--
```

``` py linenums="365"
--8<--
examples/shock_wave/shock_wave.py:365:389
--8<--
```

#### 3.4.3 Initial Value Constraint

We apply boundary conditions to the points at the initial time of the computational domain, and also use the Latin HyperCube Sampling (LHS) method to sample a total of `N_BOUNDARY` training points in the computational domain at the initial time. The code is as follows:

``` py linenums="313"
--8<--
examples/shock_wave/shock_wave.py:313:337
--8<--
```

``` py linenums="353"
--8<--
examples/shock_wave/shock_wave.py:353:364
--8<--
```

After the above three constraints are constructed, they need to be wrapped into a dictionary to facilitate subsequent passing as parameters

``` py linenums="390"
--8<--
examples/shock_wave/shock_wave.py:390:395
--8<--
```

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, we use 100 training epochs.

``` yaml linenums="59"
--8<--
examples/shock_wave/conf/shock_wave_Ma2.0.yaml:59:59
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `L-BFGS` optimizer is selected and `max_iter` is set to 100.

``` py linenums="397"
--8<--
examples/shock_wave/shock_wave.py:397:400
--8<--
```

### 3.7 Model Training and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order.

``` py linenums="402"
--8<--
examples/shock_wave/shock_wave.py:402:418
--8<--
```

This case needs to calculate the weight coefficient `relu` in PDE and BC equations based on the epoch value of each training round. Therefore, after the solver instantiation is completed, it needs to be additionally passed to the equation itself. The code is as follows:

``` py linenums="419"
--8<--
examples/shock_wave/shock_wave.py:419:422
--8<--
```

Finally, start training:

``` py linenums="424"
--8<--
examples/shock_wave/shock_wave.py:424:425
--8<--
```

After training, we visualize the shock wave with a resolution of 600x600 in the computational domain at the last moment, totaling 360,000 points. The code is as follows:

``` py linenums="447"
--8<--
examples/shock_wave/shock_wave.py:447:518
--8<--
```

## 4. Complete Code

``` py linenums="1" title="shock_wave.py"
--8<--
examples/shock_wave/shock_wave.py
--8<--
```

## 5. Result Display

This case conducted experiments for two different parameter configurations: $Ma=2.0$ and $Ma=0.728$. The results are as follows:

=== "Ma=2.0"

    <figure markdown>
      ![Ma_2.0](https://paddle-org.bj.bcebos.com/paddlescience/docs/ShockWave/shock_wave(Ma_2.000).png){ loading=lazy }
      <figcaption> Prediction results of x-direction velocity u, y-direction velocity v, pressure p, and density rho when Ma=2.0</figcaption>
    </figure>

=== "Ma=0.728"

    <figure markdown>
      ![Ma_0.728](https://paddle-org.bj.bcebos.com/paddlescience/docs/ShockWave/shock_wave(Ma_0.728).png){ loading=lazy }
      <figcaption> Prediction results of x-direction velocity u, y-direction velocity v, pressure p, and density rho when Ma=0.728</figcaption>
    </figure>

## 6. References

- [Compressible PINN - AIStudio](https://aistudio.baidu.com/projectdetail/5528154)
- [Discontinuity computing with physics-informed neural network](https://arxiv.org/abs/2206.03864)
