# 3D-Brusselator

<a href="https://aistudio.baidu.com/projectdetail/8347444" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -P data -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz --create-dirs -o data/brusselator3d_dataset.npz
    python brusselator3d.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -P data -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz --create-dirs -o data/brusselator3d_dataset.npz
    python brusselator3d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/Brusselator3D/brusselator3d_pretrained.pdparams
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [brusselator3d_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/Brusselator3D/brusselator3d_pretrained.pdparams) | loss(sup_validator): 14.51938<br>L2Rel.output(sup_validator): 0.07354 |

## 1. Background Introduction

This case introduces the Laplace Neural Operator (LNO) to build a deep learning network, which utilizes the Laplace transform to decompose the input space. Unlike the Fourier Neural Operator (FNO), LNO can handle non-periodic signals, consider transient responses, and exhibit exponential convergence. It combines the pole-residue relationship between input and output spaces, thereby achieving greater interpretability and improved generalization capabilities. A single Laplace layer in LNO approximates the accuracy of four Fourier modules in FNO, and for non-linear reaction-diffusion systems, the error of LNO is smaller than that of FNO.

This case studies the application of LNO network on the Brusselator reaction-diffusion system.

## 2. Problem Definition

Reaction-diffusion systems describe the changes in the concentration of chemical substances or particles over time and space, and are commonly used in chemistry, biology, geology, and physics. The diffusion-reaction equation can be expressed as:

$$D\frac{\partial^2 y}{\partial x^2}+ky^2-\frac{\partial y}{\partial t}=f(x,t)$$

Where $y(x,t)$ represents the concentration of chemical substances or particles at position x and time t, $f(x,t)$ is the source term, $D$ is the diffusion coefficient, and $k$ is the reaction rate.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset is provided by the original code of the LNO paper, which contains training set input and label data, validation set input and label data. The data is stored in a `.npz` file and needs to be read before training.

Before running the code for this problem, please download the [dataset](https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz) and store it in the corresponding path:

``` yaml linenums="39"
--8<--
examples/brusselator3d/conf/brusselator3d.yaml:39:40
--8<--
```

### 3.2 Model Construction

<figure markdown>
  ![LNO](https://paddle-org.bj.bcebos.com/paddlescience/docs/Brusselator3D/lno.png){ loading=lazy style="margin:0 auto"}
  <figcaption> (a) Overall LNO architecture (b) Laplace layer</figcaption>
</figure>

The above figure shows the overall LNO architecture and the schematic diagram of the Laplace layer. After the input data enters the network, it is first lifted to a higher dimension through a shallow neural network $P$, then undergoes local linear transformation $W$ on the one hand, and applies the Laplace layer on the other hand. The results of these two paths are then summed, and finally returned to the target dimension through a shallow neural network $Q$.

In the Laplace layer, the top row represents applying the pole-residue method to calculate the transient response residue $\gamma_{n}$ based on the system pole $\mu_{n}$ and residue $\beta_{n}$, representing the transient response in the Laplace domain. The bottom row represents applying the pole-residue method to calculate the steady-state response residue $i\lambda_{l}$ based on the input pole $i\omega_{l}$ and residue $i\alpha_{l}$, representing the steady-state response in the Laplace domain.

For specific code, please refer to the `lno.py` file in [Complete Code](#4).

Before building the network, it is necessary to use `linespace` to clarify the length of each dimension according to the parameter settings, so that the LNO network can initialize $\lambda$. Expressed in PaddleScience code as follows:

``` py linenums="120"
--8<--
examples/brusselator3d/brusselator3d.py:120:128
--8<--
```

In addition, if `use_grid` in the model parameters is set to `True`, no preprocessing is required, and the model will automatically generate and add a grid. If it is `False`, the grid needs to be manually added to the data during data processing, and then input into the model:

``` py linenums="114"
--8<--
examples/brusselator3d/brusselator3d.py:114:118
--8<--
```

### 3.3 Parameter and Hyperparameter Setting

We need to specify problem-related parameters, such as dataset path, length of each dimension, etc.

``` yaml linenums="32"
--8<--
examples/brusselator3d/conf/brusselator3d.yaml:32:40
--8<--
```

In addition, parameters such as training epochs and `batch_size` need to be specified in the configuration file.

``` yaml linenums="54"
--8<--
examples/brusselator3d/conf/brusselator3d.yaml:54:58
--8<--
```

### 3.4 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `AdamW` optimizer is selected, and the StepDecay learning rate adjustment strategy commonly used in machine learning is used together.

The `AdamW` optimizer is an improvement based on the `Adam` optimizer, used to solve the problem of L2 regularization failure in the `Adam` optimizer.

``` py linenums="130"
--8<--
examples/brusselator3d/brusselator3d.py:130:134
--8<--
```

### 3.5 Constraint Construction

This problem uses supervised learning for training, and there is only a supervised constraint `SupervisedConstraint`. The code is as follows:

``` py linenums="136"
--8<--
examples/brusselator3d/brusselator3d.py:136:161
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of the supervised constraint, where the `dataset` field represents the training dataset information used, and each field represents:

1. `name`: Dataset type, here `NamedArrayDataset` means dataset read from Array;
2. `input`: Input data of Array type;
3. `label`: Label data of Array type;

The `batch_size` field represents the size of the batch;

The `sampler` field represents the sampling method, where each field represents:

1. `name`: Sampler type, here `BatchSampler` means batch sampler;
2. `drop_last`: Whether to discard the last samples that cannot make up a mini-batch, set to False;
3. `shuffle`: Whether to shuffle the order when generating sample indices, set to True;

The `num_workers` field represents the number of threads when loading input;

The second parameter is the loss function. Here, the commonly used L2Rel loss function is selected, and reduction is set to "sum", that is, the loss terms generated by all data points involved in the calculation are summed;

The third parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing.

``` py linenums="157"
--8<--
examples/brusselator3d/brusselator3d.py:157:157
--8<--
```

### 3.6 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so a validator needs to be built:

``` py linenums="163"
--8<--
examples/brusselator3d/brusselator3d.py:163:187
--8<--
```

Most parameter meanings are similar to those in the constraint, with the following differences:

The third parameter is the output transcription formula `output_expr`, which specifies the key and value of the final input data;

The fourth parameter is the error evaluation function. Here, the L2Rel Error function is selected, and reduction is not set, which is the default value "mean", averaging the Error generated by all data points involved in the calculation.

### 3.7 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="189"
--8<--
examples/brusselator3d/brusselator3d.py:189:202
--8<--
```

## 4. Complete Code

``` py linenums="1" title="brusselator3d.py"
--8<--
examples/brusselator3d/brusselator3d.py
--8<--
```

``` py linenums="1" title="lno.py"
--8<--
ppsci/arch/lno.py
--8<--
```

## 5. Result Display

The following shows the prediction results and labels on the validation set.

<figure markdown>
  ![brusselator3d_compare.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/Brusselator3D/pretrained_result.png){ loading=lazy }
  <figcaption>Blue line is prediction result, yellow line is label</figcaption>
</figure>

It can be seen that the model prediction results are basically consistent with the labels.

## 6. References

- [LNO: Laplace Neural Operator for Solving Differential Equations](https://arxiv.org/abs/2303.10528)

- [Reference Code](https://github.com/qianyingcao/Laplace-Neural-Operator/tree/main/3D_Brusselator)
