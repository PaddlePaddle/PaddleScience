# CVit(Navier-Stokes)

<a href="https://aistudio.baidu.com/projectdetail/8141482" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # download data
    git lfs install
    git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D
    python ns_cvit.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # download data
    git lfs install
    git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D
    python ns_cvit.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/ns_cvit_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    pip install einops==0.8.1 # Check if einops has been updated to 0.8.1, see: https://github.com/arogozhnikov/einops/pull/353
    python ns_cvit.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    pip install einops==0.8.1 # Check if einops has been updated to 0.8.1, see: https://github.com/arogozhnikov/einops/pull/353
    # download data
    git lfs install
    git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D
    python ns_cvit.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [ns_cvit_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/ns_cvit_pretrained.pdparams) | 4-step l2_error: 0.0396 |

## 1. Background Introduction

At this stage, the models used in the sciml field are quite different from advanced models in the CV and NLP fields, and do not make good use of the advantages provided by these advanced models. Therefore, the author of the paper first proposed a unified perspective of operator learning, summarized models such as DeepONet, FNO, and GNO according to Global conditioning and Local Conditioning respectively, and then designed a Global conditioning model CVit based on the Transformer structure widely used in CV and NLP fields. Compared with previous operator learning models, it has fewer parameters and higher accuracy.

The model structure is shown in the figure below:

<img src="https://github.com/PredictiveIntelligenceLab/cvit/raw/main/figures/cvit_arch.png" alt="Cvit" width="800">

## 2. Problem Definition

As an operator learning model, CVit takes the input function $u$ and the query coordinate $y$ of the function $s$ as input, and outputs the function value $s(y)$ at the query point $y$ after operator mapping.

This problem is based on the incompressible buoyancy-driven flow in a fixed square cavity, solving the following equation:

Formulation We consider the vorticity-stream $(\omega-\psi)$ formulation of the incompressible Navier-Stokes equations on a two-dimensional periodic domain, $D=D_u=D_v=[0,2 \pi]^2$ :

$$
\begin{aligned}
& \frac{\partial \omega}{\partial t}+(v \cdot \nabla) \omega-v \Delta \omega=f^{\prime} \\
& \omega=-\Delta \psi \quad \int_D \psi=0, \\
& v=\left(\frac{\partial \psi}{\partial x_2},-\frac{\partial \psi}{\partial x_1}\right)
\end{aligned}
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

``` py linenums="80"
--8<--
examples/ns/ns_cvit.py:80:81
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, here we specify the input variable name of the network model as `("u", "y")` and the output variable name as `("s")`, these names are consistent with the subsequent code.

Then by specifying hyperparameters such as input dimension, coordinate dimension, output dimension, and number of model layers of CVit, a `model` can be instantiated

``` yaml linenums="40"
--8<--
examples/ns/conf/ns_cvit_small_8x8.yaml:40:59
--8<--
```

### 3.2 Data Preparation

The data slices in this problem are stored in `NavierStokes2D/*.h5` files, divided into training and test sets, and their data contents are shown in the table below (this information will be printed during runtime).

| File Name | File Quantity | Data Shape | Input Shape | Label Shape |
| :-- | :-- | :-- | :-- | :-- |
| NavierStokes2D_train_*.h5 | 52 |`[1000, 14, 128, 128, 3]` | `[4000, 10, 128, 128, 3]` | `[4000, 1, 128, 128, 3]` |
| NavierStokes2D_test_*.h5 | 41 | `[5200, 14, 128, 128, 3]` | `[20800, 10, 128, 128, 3]` | `[20800, 1, 128, 128, 3]` |

The data reading function is as follows:

``` py linenums="27"
--8<--
examples/ns/ns_cvit.py:27:76
--8<--
```

During training and testing, the previous 10 moments are used to predict the next moment, and during testing, 4 consecutive moments will be predicted in an autoregressive form.

### 3.3 Constraint Construction

#### 3.3.1 Supervised Constraint

During training, `batch_size` groups of data from $u$ and `query_point` $y$ coordinates are randomly selected to form training input data, and label data is randomly selected from $s$ with the same `batch_size` x `query_point` label points.

``` py linenums="104"
--8<--
examples/ns/ns_cvit.py:104:145
--8<--
```

The first parameter of `SupervisedConstraint` is the data configuration for training. We use `NamedArrayDataset` as the dataset type, and pass in custom `random_query` as `transforms` to complete the above sample random selection process;

The second parameter is the calculation expression of the constraint. We only need to calculate $s$, so fill in an anonymous expression that does not do any processing and directly takes out the model output result "s";

The third parameter is the loss function, here `MSELoss` function is selected;

The fourth parameter is the name of the constraint condition. Each constraint condition needs to be named for subsequent indexing. Here it is named "Sup".

### 3.4 Hyperparameter Setting

Next, the number of training epochs and learning rate need to be specified. Here, based on experimental experience, 200 training epochs are used, the initial learning rate is 0.001, the number of warm-up epochs is 5, the global gradient clipping coefficient is 1.0, and the weight decay is 1e-5.

``` yaml linenums="61"
--8<--
examples/ns/conf/ns_cvit_small_8x8.yaml:61:79
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and used in conjunction with the ExponentialDecay learning rate adjustment strategy commonly used in machine learning.

``` py linenums="147"
--8<--
examples/ns/ns_cvit.py:147:155
--8<--
```

### 3.6 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator.

``` py linenums="157"
--8<--
examples/ns/ns_cvit.py:157:202
--8<--
```

In the process, we used the custom evaluation function `l2_err_func` to evaluate the 2-norm error of all samples and three output physical quantities on the test set.

### 3.7 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="204"
--8<--
examples/ns/ns_cvit.py:204:213
--8<--
```

## 4. Complete Code

``` py linenums="1" title="ns_cvit.py"
--8<--
examples/ns/ns_cvit.py
--8<--
```

## 5. Result Display

The prediction results, reference results and absolute errors on the test set are shown in the figure below.

<figure markdown>
  ![ns_u.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/ns_u.png){ loading=lazy }
  <figcaption> The left side is the prediction result of CVit for physical quantity u, the middle is the reference result of physical quantity u, and the right side is the difference between the two</figcaption>
  ![ns_ux.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/ns_ux.png){ loading=lazy }
  <figcaption> The left side is the prediction result of CVit for physical quantity ux, the middle is the reference result of physical quantity ux, and the right side is the difference between the two</figcaption>
  ![ns_uy.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/cvit/ns_uy.png){ loading=lazy }
  <figcaption> The left side is the prediction result of CVit for physical quantity uy, the middle is the reference result of physical quantity uy, and the right side is the difference between the two</figcaption>
</figure>

It can be seen that the three predicted physical quantities of the model are basically consistent with the reference results. Through autoregression, the average error of continuous inference for 4 steps is 0.039%.

## 6. References

- [Bridging Operator Learning and Conditioned Neural Fields: A Unifying Perspective](https://arxiv.org/abs/2405.13998)
- [PredictiveIntelligenceLab/cvit/ns](https://github.com/PredictiveIntelligenceLab/cvit/blob/main/ns/README.md)
- [The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks](https://arxiv.org/abs/2203.13181)
