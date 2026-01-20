# Euler Beam

=== "Model Training Command"

    ``` sh
    python euler_beam.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python euler_beam.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python euler_beam.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [euler_beam_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams) | loss(L2Rel_Metric): 0.00000<br>L2Rel.u(L2Rel_Metric): 0.00058 |

## 1. Problem Definition

Euler Beam Formula:

$$
\dfrac{\partial^{4} u}{\partial x^{4}} + 1 = 0, x \in [0, 1]
$$

Boundary Conditions:

$$
u''(1)=0, u'''(1)=0
$$

Dirichlet Condition:

$$
u(0)=0
$$

Neumann Boundary Condition:

$$
u'(0)=0
$$

## 2. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 2.1 Model Construction

In the Euler Beam problem, each known coordinate point $x$ has a corresponding unknown quantity $u$ to be solved. We use a relatively simple MLP (Multilayer Perceptron) here to represent the mapping function $f: \mathbb{R}^1 \to \mathbb{R}^1$ from $x$ to $u$, i.e.:

$$
u = f(x)
$$

In the above formula, $f$ is the MLP model itself, expressed in PaddleScience code as follows:

``` py linenums="24"
--8<--
examples/euler_beam/euler_beam.py:24:25
--8<--
```

The parameters used to initialize the model are configured through the configuration file:

``` yaml linenums="38"
--8<--
examples/euler_beam/conf/euler_beam.yaml:38:43
--8<--
```

Then by specifying the number of layers and neurons of MLP, we instantiated a neural network model `model` with 3 hidden layers and 20 neurons per layer.

### 2.2 Equation Construction

The equation construction of Euler Beam can directly use the `Biharmonic` built in PaddleScience, specifying the parameter `dim` of this class as 1, `q` as -1, and `D` as 1.

``` py linenums="30"
--8<--
examples/euler_beam/euler_beam.py:30:31
--8<--
```

### 2.3 Computational Domain Construction

In this article, the Euler Beam problem acts on a one-dimensional area of (0.0, 1.0), so the spatial geometry `Interval` built in PaddleScience can be used directly as the computational domain.

``` py linenums="27"
--8<--
examples/euler_beam/euler_beam.py:27:28
--8<--
```

### 2.4 Constraint Construction

In this case, we used two constraints to guide the training of the model in the computational domain, namely the equation constraint acting on the sampling point and the constraint acting on the boundary point.

Before defining constraints, you need to specify the number of sampling points for each constraint, indicating the number of sampled data for each constraint in its corresponding computational domain, as well as general sampling configuration.

``` yaml linenums="45"
--8<--
examples/euler_beam/conf/euler_beam.yaml:45:55
--8<--
```

#### 2.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on internal points as an example, the code is as follows:

``` py linenums="33"
--8<--
examples/euler_beam/euler_beam.py:33:47
--8<--
```

#### 2.4.2 Boundary Constraint

Similarly, we also need to construct boundary constraints. However, unlike constructing `InteriorConstraint`, since the action area is the boundary, we use the `BoundaryConstraint` class, code as follows:

``` py linenums="48"
--8<--
examples/euler_beam/euler_beam.py:48:61
--8<--
```

### 2.5 Hyperparameter Setting

Next, we need to specify the number of training epochs in the configuration file. Here, based on experimental experience, we use 10,000 training epochs, with an evaluation interval of 1,000 epochs.

``` yaml linenums="45"
--8<--
examples/euler_beam/conf/euler_beam.yaml:45:51
--8<--
```

### 2.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="68"
--8<--
examples/euler_beam/euler_beam.py:68:69
--8<--
```

### 2.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="77"
--8<--
examples/euler_beam/euler_beam.py:77:90
--8<--
```

### 2.8 Visualizer Construction

During model evaluation, if the evaluation result is data that can be visualized, we can choose a suitable visualizer to visualize the output result.

The output data in this article is a graph, so we only need to save the evaluated output data as a **png** file. The code is as follows:

``` py linenums="92"
--8<--
examples/euler_beam/euler_beam.py:92:105
--8<--
```

### 2.9 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="107"
--8<--
examples/euler_beam/euler_beam.py:107:132
--8<--
```

## 3. Complete Code

``` py linenums="1" title="euler_beam.py"
--8<--
examples/euler_beam/euler_beam.py
--8<--
```

## 4. Result Display

The trained model is used to predict a total of `NPOINT_TOTAL` points $x_i$ uniformly taken from the above computational domain. The prediction results are shown below. In the image, the abscissa is $x$, and the ordinate is the corresponding predicted result $u$.

<figure markdown>
  ![euler_beam](https://paddle-org.bj.bcebos.com/paddlescience/docs/euler_beam/euler_beam.png){ loading=lazy }
  <figcaption>Model prediction result</figcaption>
</figure>
