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

According to the universal approximation theorem in the field of machine learning, a neural network model can not only fit the functional mapping relationship from input data to output data, but also be extended to fit the mapping relationship between functions, which is called "operator" learning.

Therefore, DeepONet has considerable potential in various fields. Here are some possible application areas:

1. **Fluid Dynamics Simulation**: DeepONet can be used for numerical solution of fluid dynamics equations, such as Navier-Stokes equations. This makes DeepONet directly applicable in fields such as aerodynamics, fluid machinery, and climate simulation.
2. **Image Processing and Computer Vision**: DeepONet can learn features in images and be used for tasks such as classification, segmentation, and detection. For example, it can be used for medical image analysis, including disease detection and prognosis prediction.
3. **Signal Processing**: DeepONet can be used for various signal processing tasks, such as denoising, compression, and restoration. In fields such as communications, radar, and sonar, DeepONet has potential applications.
4. **Control Systems**: DeepONet can be used for the design and optimization of control systems. For example, it can learn the dynamic behavior of the system and be used to predict and control the future behavior of the system.
5. **Finance**: DeepONet can be used for financial forecasting and analysis, such as stock price prediction, risk assessment, credit risk analysis, etc.
6. **Human-Computer Interaction**: DeepONet can be used for tasks such as speech recognition, natural language processing, and gesture recognition, making human-computer interaction more intelligent and natural.
7. **Environmental Science**: DeepONet can be used for tasks such as climate model prediction, ecosystem simulation, and environmental pollution detection.

It should be noted that although DeepONet has potential applications in many fields, each field has its unique problems and challenges. When applying DeepONet to a specific field, a deep understanding of the problems in that field is required, and model adjustments and optimizations may be needed for that field.

## 2. Problem Definition

Assume there is the following ODE system:

$$
\begin{equation}
\left\{\begin{array}{l}
\frac{d}{d x} \mathbf{s}(x)=\mathbf{g}(\mathbf{s}(x), u(x), x) \\
\mathbf{s}(a)=s_0
\end{array}\right.
\end{equation}
$$

where $u \in V$ (and $u$ is continuous on $[a, b]$) serves as the input signal, and $\mathbf{s}: [a,b] \rightarrow \mathbb{R}^K$ is the solution of this equation, serving as the output signal.
Therefore, an operator $G$ can be defined, which satisfies:

$$
\begin{equation}
(G u)(x)=s_0+\int_a^x \mathbf{g}((G u)(t), u(t), t) d t
\end{equation}
$$

Therefore, a neural network model can be used, with $u$ and $x$ as inputs and $G(u)(x)$ as output, to perform supervised training to fit the $G$ operator itself.

Note: According to the above formula, it can be found that the operator $G$ is an integral operator "$\int$", which acts on a given function $u$ to obtain its original function $G(u)$ under a certain initial value condition (in this problem, the initial value condition is $G(u)(0)=0$).

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

In the above problem, we determined that the input is $u$ and $y$, and the output is $G(u)$. According to the DeepONet paper, we use `DeepONet` containing branch and trunk sub-networks to create the network model, expressed in PaddleScience code as follows:

``` py linenums="27"
--8<--
examples/operator_learning/deeponet.py:27:27
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, we specify the input variable name of the network model as `u` and `y` and the output variable name as `G`. Then by specifying the number of SENSORS, number of feature channels, number of hidden layers, number of neurons and activation functions of sub-networks of `DeepONet`, we instantiated the `DeepONet` neural network model `model`.

### 3.3 Constraint Construction

This article uses supervised learning to constrain the model output $G(u)$.

Before defining constraints, data reading configuration such as file path needs to be specified for supervised constraint, including file path, input data field name, label data field name, alias dictionary before and after data conversion.

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

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraint, here fill in `train_dataloader_cfg` instantiated in [3.4 Constraint Construction](#34) chapter;

The second parameter is the loss function. Here we choose the commonly used MSE function, and `reduction` is the default value `"mean"`, that is, we will sum and average the loss terms generated by all data points involved in the calculation;

The third parameter is the equation expression, used to describe how to calculate the constraint target. Here we only need to get the output corresponding to the output field `G` from the output dictionary;

After the supervised constraint is constructed, encapsulate it into a dictionary with the names we just named as keys for subsequent access.

``` py linenums="45"
--8<--
examples/operator_learning/deeponet.py:45:46
--8<--
```

### 3.4 Hyperparameter Setting

Next, we need to specify the number of training epochs and learning rate. Here, based on experimental experience, we use 10,000 training epochs, and evaluate the model accuracy every 500 epochs.

``` yaml linenums="49"
--8<--
examples/operator_learning/conf/deeponet.yaml:49:55
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and the learning rate is set to `0.001`.

``` py linenums="48"
--8<--
examples/operator_learning/deeponet.py:48:49
--8<--
```

### 3.6 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator.

``` py linenums="51"
--8<--
examples/operator_learning/deeponet.py:51:60
--8<--
```

For evaluation metric `metric`, select `ppsci.metric.L2Rel`.

Other configurations are similar to the settings of [Constraint Construction](#33).

### 3.7 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="71"
--8<--
examples/operator_learning/deeponet.py:71:90
--8<--
```

### 3.8 Result Visualization

After the model training is completed, we can manually construct $u$ and $y$ and discretize them within an appropriate range to obtain corresponding input data, then predict $G(u)(y)$, and plot the image together with the standard solution of $G(u)$ for comparison. (Here we constructed 9 sets of $u-G(u)$ function pairs) for testing

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
