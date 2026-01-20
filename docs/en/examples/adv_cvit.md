# CVit (Advection)

<a href="https://aistudio.baidu.com/projectdetail/8141430" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

!!! note

    Before running the model, please download the two files `adv_a0.npy` and `adv_aT.npy` from [Zhengyu-Huang/Operator-Learning](https://github.com/Zhengyu-Huang/Operator-Learning/tree/main/data) and place them in the `./examples/adv/data/` folder.

=== "Model Training Command"

    ``` sh
    python adv_cvit.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python adv_cvit.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/adv_cvit_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python adv_cvit.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    python adv_cvit.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [adv_cvit_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/adv_cvit_pretrained.pdparams) | L2 error(mean): 0.028<br>L2 error(median): 0.022<br>L2 error(max): 0.166<br>L2 error(min): 0.0015 |

## 1. Background Introduction

Current models used in the sciml field are quite different from advanced models in the CV and NLP fields, and do not make good use of the advantages provided by these advanced models. Therefore, the authors first proposed a unified perspective of operator learning, summarizing DeepONet, FNO, GNO and other models according to Global conditioning and Local Conditioning respectively, and then designed a Global conditioning model CVit based on the Transformer structure widely used in CV and NLP fields. Compared with previous operator learning models, it has fewer parameters and higher accuracy.

The model structure is shown in the figure below:

<img src="https://github.com/PredictiveIntelligenceLab/cvit/raw/main/figures/cvit_arch.png" alt="Cvit" width="800">

## 2. Problem Definition

As an operator learning model, CVit takes the input function $u$ and the query coordinate $y$ of function $s$ as input, and outputs the function value $s(y)$ at the query point $y$ of the function after operator mapping.

This problem solves the following equation:

Formulation The 1D advection equation in $\Omega=[0,1)$ is

$$
\begin{aligned}
& \frac{\partial u}{\partial t}+c \frac{\partial u}{\partial x}=0 \quad x \in \Omega, \\
& u(0)=u_0
\end{aligned}
$$

where $c=1$ is the constant advection speed, and periodic boundary conditions are imposed. We are interested in the map from the initial $u_0$ to solution $u(\cdot, T)$ at $T=0.5$. The initial condition $u_0$ is assumed to be

$$
u_0=-1+2 \mathbb{1}\left\{\tilde{u_0} \geq 0\right\}
$$

where $\widetilde{u_0}$ a centered Gaussian

$$
\widetilde{u_0} \sim \mathbb{N}(0, \mathrm{C}) \quad \text { and } \quad \mathrm{C}=\left(-\Delta+\tau^2\right)^{-d} \text {; }
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In this problem, for each function $u$, after being mapped to $s$ by the operator learning model, there is a corresponding label $s(y)$ on $y$, so here CVit is used to represent the mapping relationship from $(u, y)$ to $s(y)$:

$$
s(y) = G(u)(y)
$$

In the above formula, $G(u)$ is the CVit model itself, expressed in PaddleScience code as follows

``` py linenums="55"
--8<--
examples/adv/adv_cvit.py:55:56
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, the input variable name of the network model is specified as `("u", "y")` and the output variable name is `("s")`, these names are consistent with the subsequent code.

Then by specifying the hyperparameters such as input dimension, coordinate dimension, output dimension, and number of model layers of CVit, a `model` can be instantiated.

``` yaml linenums="34"
--8<--
examples/adv/conf/adv_cvit.yaml:34:54
--8<--
```

### 3.2 Data Preparation

The data in this problem is stored in `adv_a0.py` and `adv_aT.py` files. After randomly shuffling the data, the first 20000 data are taken as training data, and the last 10000 are test data.

``` py linenums="27"
--8<--
examples/adv/adv_cvit.py:27:76
--8<--
```

### 3.3 Constraint Construction

#### 3.3.1 Supervised Constraint

During training, `batch_size` groups of data from $u$ are randomly selected, and `query_point` $y$ coordinates are randomly selected at the same time, thus constituting training input data. Label data is randomly selected from $s$ with the same `batch_size` x `query_point` label points.

``` py linenums="83"
--8<--
examples/adv/adv_cvit.py:83:115
--8<--
```

The first parameter of `SupervisedConstraint` is the data configuration used for training. We use `ContinuousNamedArrayDataset` as the dataset type, and pass in custom `gen_input_batch_train` and `gen_label_batch_train` to complete the random selection process of the above training input and label samples;

The second parameter is the calculation expression of the constraint. We only need to calculate $s$, so we fill in an anonymous expression that directly takes out the model output result "s" without any processing;

The third parameter is the loss function, here `MSELoss` function is selected;

The fourth parameter is the name of the constraint condition. Each constraint condition needs to be named to facilitate subsequent indexing. Here it is named "Sup".

### 3.4 Hyperparameter Setting

Next, you need to specify the number of training epochs and learning rate. Here, based on experimental experience, 200,000 training epochs are used. The initial learning rate is 0.0001, the global gradient clipping coefficient is 1.0, the weight decay is 1e-5, and model averaging EMA is performed every 1 training epoch.

``` yaml linenums="56"
--8<--
examples/adv/conf/adv_cvit.yaml:56:79
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `AdamW` optimizer is selected, and the ExponentialDecay learning rate adjustment strategy commonly used in machine learning is used together.

``` py linenums="117"
--8<--
examples/adv/adv_cvit.py:117:125
--8<--
```

### 3.6 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="127"
--8<--
examples/adv/adv_cvit.py:127:145
--8<--
```

## 4. Complete Code

``` py linenums="1" title="adv_cvit.py"
--8<--
examples/adv/adv_cvit.py
--8<--
```

## 5. Result Display

The prediction results, reference results and absolute value errors on the test set are shown in the figure below.

<figure markdown>
  ![adv_cvit.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/adv_cvit.png){ loading=lazy }
  <figcaption> CVit fitting results for signals (here showing the 16.6% of results with poorer fitting errors, the overall average error of the test set is 2.8%) </figcaption>
</figure>

## 6. References

- [Bridging Operator Learning and Conditioned Neural Fields: A Unifying Perspective](https://arxiv.org/abs/2405.13998)
- [PredictiveIntelligenceLab/cvit/adv](https://github.com/PredictiveIntelligenceLab/cvit/blob/main/adv/README.md)
- [The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks](https://arxiv.org/abs/2203.13181)
