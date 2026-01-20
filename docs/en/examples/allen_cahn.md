# Allen-Cahn

<a href="https://aistudio.baidu.com/projectdetail/7927786" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat --create-dirs -o ./dataset/allen_cahn.mat
    # Using Adam optimizer
    python allen_cahn_piratenet.py
    # Using SOAP optimizer
    python allen_cahn_piratenet.py TRAIN.optim=soap TRAIN.lr_scheduler.warmup_epoch=5
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat --create-dirs -o ./dataset/allen_cahn.mat
    # Using Adam pretrained model
    python allen_cahn_piratenet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_pretrained.pdparams
    # Using SOAP pretrained model
    python allen_cahn_piratenet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_soap_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python allen_cahn_piratenet.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat --create-dirs -o ./dataset/allen_cahn.mat
    python allen_cahn_piratenet.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [allen_cahn_piratenet_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_pretrained.pdparams) | L2Rel.u: 1.2e-05 |
| [allen_cahn_piratenet_soap_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_soap_pretrained.pdparams) | L2Rel.u: 6.8e-6 |

## 1. Background Introduction

The Allen-Cahn equation (sometimes called the model equation or phase field equation) is a mathematical model commonly used to describe the interface evolution between two different phases. This equation was first proposed by Samuel Allen and John Cahn in the 1970s to describe the process of phase separation in alloys. The Allen-Cahn equation is a non-linear partial differential equation, and its general form can be written as:

$$ \frac{\partial u}{\partial t} = \varepsilon^2 \Delta u - F'(u) $$

Here:

- $u(\mathbf{x},t)$ is a field variable representing a physical quantity, such as the composition concentration of an alloy or the order parameter in a crystal.
- $t$ represents time.
- $\mathbf{x}$ represents spatial position.
- $\Delta$ is the Laplace operator, corresponding to the second-order partial derivative of the spatial variable (i.e. $\Delta u = \nabla^2 u$), used to describe the spatial diffusion process.
- $\varepsilon$ is a small positive parameter, which is related to the width of the phase interface.
- $F(u)$ is a bistable potential energy function, usually taken as $F(u) = \frac{1}{4}(u^2-1)^2$, which makes $F'(u) = u^3 - u$ its derivative, representing the non-linear reaction term responsible for driving the system to a stable state.

The $F'(u)$ term in this equation makes two stable equilibrium states near $u=1$ and $u=-1$, corresponding to different physical phases. The $\varepsilon^2 \Delta u$ term describes the diffusion effect caused by the curvature of the phase interface, which causes the interface to tend to reduce curvature. Therefore, the Allen-Cahn equation describes the phase transition due to the influence of phase interface curvature and potential energy.

In practical applications, the equation may also include boundary conditions and initial conditions to facilitate numerical simulation and analysis of specific problems. For example, in specific physical problems, there may be Neumann boundary conditions (zero derivative, indicating no flux across the boundary) or Dirichlet boundary conditions (fixed boundary values).

This case solves the following Allen-Cahn equation:

$$
\begin{aligned}
    & u_t - 0.0001 u_{xx} + 5 u^3 - 5 u  = 0,\quad t \in [0, 1],\ x\in[-1, 1],\\
    &u(x,0) = x^2 \cos(\pi x),\\
    &u(t, -1) = u(t, 1),\\
    &u_x(t, -1) = u_x(t, 1).
\end{aligned}
$$

## 2. Problem Definition

According to the above equation, it can be seen that the calculation domain is $[0, 1]\times [-1, 1]$, containing one initial condition: $u(x,0) = x^2 \cos(\pi x)$, and two periodic boundary conditions: $u(t, -1) = u(t, 1)$, $u_x(t, -1) = u_x(t, 1)$.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the Allen-Cahn problem, each known coordinate point $(t, x)$ corresponds to the unknown quantity $(u)$ to be solved.
Here, PirateNet is used to represent the mapping function $f: \mathbb{R}^2 \to \mathbb{R}^1$ from $(t, x)$ to $(u)$, namely:

$$
u = f(t, x)
$$

In the above formula, $f$ is the PirateNet model itself, expressed in PaddleScience code as follows

``` py linenums="63"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:63:64
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, the input variable name of the network model is specified as `("t", "x")` and the output variable name is `("u")`, these names are consistent with the subsequent code.

Then by specifying the number of layers and neurons of PirateNet, a neural network model `model` with 3 PiraBlocks is instantiated, where the number of hidden layer neurons in each PiraBlock is 256, and `tanh` is used as the activation function.

``` yaml linenums="34"
--8<--
examples/allen_cahn/conf/allen_cahn_piratenet.yaml:34:40
--8<--
```

### 3.2 Equation Construction

The Allen-Cahn differential equation can be expressed by the following code:

``` py linenums="66"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:66:67
--8<--
```

### 3.3 Computational Domain Construction

The computational domain of this problem is $[0, 1]\times [-1, 1]$, and the data used for training has been generated in advance and saved in `./dataset/allen_cahn.mat`. Read and generate discrete points within the computational domain.

``` py linenums="69"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:69:81
--8<--
```

### 3.4 Constraint Construction

#### 3.4.1 Interior Point Constraint

Taking `SupervisedConstraint` acting on interior points as an example, the code is as follows:

``` py linenums="94"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:94:110
--8<--
```

The first parameter of `SupervisedConstraint` is the data configuration used for training. Since we use real-time randomly generated data instead of fixed data points, we fill in the custom input data/label generation function;

The second parameter is the equation expression, so pass in the Allen-Cahn equation object;

The third parameter is the loss function. Here, the `CausalMSELoss` function is selected, which will re-weight different time windows according to `causal` and `tol` parameters, and can better optimize transient problems;

The fourth parameter is the name of the constraint condition. Each constraint condition needs to be named to facilitate subsequent indexing. Here it is named "PDE".

#### 3.4.2 Periodic Boundary Constraint

Here we use the hard-constraint method. In the neural network model, periodic functions such as cos and sin are used to periodically process the input data, so that $u_{\theta}$ directly satisfies the periodic property of the equation mathematically.
According to the equation, the period of function $u(t, x)$ on the $x$ axis is 2, so set this period in the model configuration.

``` yaml linenums="41"
--8<--
examples/allen_cahn/conf/allen_cahn_piratenet.yaml:41:42
--8<--
```

#### 3.4.3 Initial Value Constraint

The third constraint condition is the initial value constraint, the code is as follows:

``` py linenums="112"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:112:125
--8<--
```

After the differential equation constraint and initial value constraint are constructed, encapsulate them into a dictionary with the names just given as keys for subsequent access.

``` py linenums="126"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:126:130
--8<--
```

### 3.5 Hyperparameter Setting

Next, you need to specify the number of training epochs and learning rate. Here, based on experimental experience, 300 training epochs and an initial learning rate of 0.001 are used.

``` yaml linenums="50"
--8<--
examples/allen_cahn/conf/allen_cahn_piratenet.yaml:50:63
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and the ExponentialDecay learning rate adjustment strategy commonly used in machine learning is used together.

``` py linenums="132"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:132:136
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator.

``` py linenums="146"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:146:164
--8<--
```

### 3.8 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="166"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py:166:192
--8<--
```

## 4. Complete Code

``` py linenums="1" title="allen_cahn_piratenet.py"
--8<--
examples/allen_cahn/allen_cahn_piratenet.py
--8<--
```

## 5. Result Display

$201\times501$ points are uniformly sampled on the computational domain, and their prediction results and analytical solutions are shown in the figure below.

<figure markdown>
  ![allen_cahn_piratenet.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/AllenCahn/allen_cahn_piratenet_ac.png){ loading=lazy }
  <figcaption> Left: PaddleScience prediction result, Middle: Analytical solution result, Right: Difference between the two</figcaption>
</figure>

It can be seen that for the function $u(t, x)$, the prediction result of the model is basically consistent with the result of the analytical solution.

## 6. References

- [PIRATENETS: PHYSICS-INFORMED DEEP LEARNING WITHRESIDUAL ADAPTIVE NETWORKS](https://arxiv.org/pdf/2402.00326.pdf)
- [Allen-Cahn equation](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/examples/allen_cahn/README.md)
- [Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective](https://arxiv.org/abs/2502.00604)
