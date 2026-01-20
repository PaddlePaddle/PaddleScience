# DeepONet

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6566389?sUid=438690&shared=1&ts=1690775701017" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/deeponet/antiderivative_unaligned_train.npz -o antiderivative_unaligned_train.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/deeponet/antiderivative_unaligned_test.npz -o antiderivative_unaligned_test.npz
    python deeponet.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/deeponet/antiderivative_unaligned_train.npz -o antiderivative_unaligned_train.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/deeponet/antiderivative_unaligned_test.npz -o antiderivative_unaligned_test.npz
    python deeponet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/deeponet/deeponet_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python deeponet.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python deeponet.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [deeponet_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/deeponet/deeponet_pretrained.pdparams) | loss(G_eval): 0.00003<br>L2Rel.G(G_eval): 0.01799 |

## 1. Background Introduction

Based on the universal approximation theorem for operators, neural networks can approximate not just functions, but also nonlinear operators that map one function space to another. This is the core concept of "operator learning."

DeepONet, a prominent operator learning framework, demonstrates significant potential across diverse fields:

- **Fluid Dynamics**: Solving partial differential equations (PDEs) like the Navier-Stokes equations for aerodynamics and climate modeling.
- **Computer Vision**: Learning complex mappings for image classification, segmentation, and medical analysis.
- **Signal Processing**: Applications in denoising, compression, and restoration for communications and radar.
- **Control Systems**: Modeling system dynamics for predictive control and optimization.
- **Finance & Environment**: Risk assessment, market forecasting, and climate prediction.

While DeepONet is versatile, successful application requires domain-specific adaptation and optimization.

## 2. Problem Definition

Consider the following Ordinary Differential Equation (ODE) system:

$$
\begin{cases}
\frac{d}{dx} \mathbf{s}(x) = \mathbf{g}(\mathbf{s}(x), u(x), x) \\
\mathbf{s}(a) = s_0
\end{cases}
$$

Here, $u \in V$ (continuous on $[a, b]$) is the input signal, and the solution $\mathbf{s}: [a,b] \rightarrow \mathbb{R}^K$ is the output. We define an operator $G$ such that $\mathbf{s}(x) = (G u)(x)$. This can be expressed in integral form:

$$
(G u)(x) = s_0 + \int_a^x \mathbf{g}((G u)(t), u(t), t) dt
$$

Our goal is to train a neural network that takes the function $u$ and a coordinate $x$ as inputs and predicts the value $(G u)(x)$. Essentially, we aim to learn the operator $G$.

**Note**: In this specific example, $G$ acts as an integral operator (antiderivative) with the initial condition $(G u)(0)=0$.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

This case dataset uses the dataset provided by the DeepXDE official documentation. One npz file already contains the training set and validation set. [Download Address](https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2Fdeepxde%2Fdeeponet%5Fantiderivative%5Funaligned)

The data file description is as follows:

`antiderivative_unaligned_train.npz`

|Field Name | Description |
|:----:|:---------:|
|X_train0 |Training input data corresponding to $u$, shape is (10000, 100) |
|X_train1 |Training input data corresponding to $y$, shape is (10000, 1) |
|y_train |Training label data corresponding to $G(u)$, shape is (10000,1) |

`antiderivative_unaligned_test.npz`

|Field Name | Description |
|:----:|:---------:|
|X_test0 |Test input data corresponding to $u$, shape is (100000, 100) |
|X_test1 |Test input data corresponding to $y$, shape is (100000, 1) |
|y_test |Test label data corresponding to $G(u)$, shape is (100000,1) |

### 3.2 Model Construction

The inputs are the function $u$ and the coordinate $y$, and the output is the value $G(u)(y)$. Following the DeepONet architecture, we employ a Branch Net (for $u$) and a Trunk Net (for $y$).

``` py linenums="27"
--8<--
examples/operator_learning/deeponet.py:27:27
--8<--
```

We specify input keys as `u` and `y`, and the output key as `G`. The `DeepONet` model is instantiated by configuring the number of sensors, feature channels, hidden layers, neurons, and activation functions.

### 3.3 Constraint Construction

We use supervised learning to train the model. First, we configure the data loader, specifying file paths, input/label keys, and aliases.

``` py linenums="30"
--8<--
examples/operator_learning/deeponet.py:30:38
--8<--
```

#### 3.3.1 Supervised Constraint

Since we train in a supervised manner, supervised constraint `SupervisedConstraint` is used here:

``` py linenums="40"
--8<--
examples/operator_learning/deeponet.py:40:44
--8<--
```

- **Dataloader**: Uses `train_dataloader_cfg`.
- **Loss**: MSE with `reduction="mean"`.
- **Target**: The model output `G`.

The constraint is then stored in a dictionary.

``` py linenums="45"
--8<--
examples/operator_learning/deeponet.py:45:46
--8<--
```

### 3.4 Hyperparameter Setting

We set the training epochs to 10,000 and the evaluation interval to 500 epochs.

``` yaml linenums="49"
--8<--
examples/operator_learning/conf/deeponet.yaml:49:55
--8<--
```

### 3.5 Optimizer Construction

We use the `Adam` optimizer with a learning rate of `0.001`.

``` py linenums="48"
--8<--
examples/operator_learning/deeponet.py:48:49
--8<--
```

### 3.6 Validator Construction

To monitor performance, we construct a `SupervisedValidator` for periodic evaluation on the test set.

``` py linenums="51"
--8<--
examples/operator_learning/deeponet.py:51:60
--8<--
```

For evaluation metric `metric`, select `ppsci.metric.L2Rel`.

Other configurations are similar to the settings of [Constraint Construction](#33).

### 3.7 Model Training and Evaluation

With all components configured, we pass them to `ppsci.solver.Solver` to commence training and evaluation.

``` py linenums="71"
--8<--
examples/operator_learning/deeponet.py:71:90
--8<--
```

### 3.8 Result Visualization

Post-training, we verify the model by constructing 9 synthetic $u-G(u)$ function pairs. We discretize $u$ and $y$, predict $G(u)(y)$, and compare the results with the analytical solutions.

``` py linenums="92"
--8<--
examples/operator_learning/deeponet.py:92:151
--8<--
```

## 4. Complete Code

``` py linenums="1" title="deeponet.py"
--8<--
examples/operator_learning/deeponet.py
--8<--
```

## 5. Result Display

<figure markdown>
  ![result0.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_0_result.png){ loading=lazy }
  ![result1.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_1_result.png){ loading=lazy }
  ![result2.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_2_result.png){ loading=lazy }
  ![result3.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_3_result.png){ loading=lazy }
  ![result4.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_4_result.png){ loading=lazy }
  ![result5.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_5_result.png){ loading=lazy }
  ![result6.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_6_result.png){ loading=lazy }
  ![result7.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_7_result.png){ loading=lazy }
  ![result8.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepONet/func_8_result.png){ loading=lazy }
</figure>

## 6. References

- [DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://export.arxiv.org/pdf/1910.03193.pdf)
- [DeepXDE - Antiderivative operator from an unaligned dataset](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html)
