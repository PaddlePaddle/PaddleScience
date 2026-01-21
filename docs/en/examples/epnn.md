# EPNN

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat --create-dirs -o ./datasets/dstate-16-plas.dat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat --create-dirs -o ./datasets/dstress-16-plas.dat
    python epnn.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat --create-dirs -o ./datasets/dstate-16-plas.dat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat --create-dirs -o ./datasets/dstress-16-plas.dat
    python epnn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/epnn/epnn_pretrained.pdparams
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [epnn_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/epnn/epnn_pretrained.pdparams) | error(total): 3.96903<br> error(error_elasto): 0.65328<br> error(error_plastic): 3.04176<br> error(error_stress): 0.27399 |

## 1. Background Introduction

Here we mainly reproduce the Physics-Informed Neural Network (PINN) surrogate model of the Elasto-Plastic Neural Network (EPNN). Incorporating these physics into the architecture of neural networks can train the network more effectively while using less data for training, and at the same time enhance inference capabilities for loading regimes outside the training data. The architecture of EPNN is model and material agnostic, meaning it can adapt to various types of elastoplastic materials, including geomaterials and metals; and experimental data can be used directly to train the network. To demonstrate the robustness of the proposed architecture, we apply its general framework to the elastoplastic behavior of sand. EPNN outperforms conventional neural network architectures in predicting unobserved strain-controlled loading paths for sands of different initial densities.

## 2. Problem Definition

In a neural network, information flows through connected neurons. The "strength" of each link in a neural network is determined by a variable weight:

$$
z_l^{\mathrm{i}}=W_{k l}^{\mathrm{i}-1, \mathrm{i}} a_k^{\mathrm{i}-1}+b^{\mathrm{i}-1}, \quad k=1: N^{\mathrm{i}-1} \quad \text { or } \quad \mathbf{z}^{\mathrm{i}}=\mathbf{a}^{\mathrm{i}-1} \mathbf{W}^{\mathrm{i}-1, \mathrm{i}}+b^{\mathrm{i}-1} \mathbf{I}
$$

Where $b$ is the bias term; $N$ is the number of neurons in different layers; $I$ refers to the unit vector where all elements are 1.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the EPNN problem, build the network, expressed in PaddleScience code as follows

``` py linenums="371"
--8<--
examples/epnn/functions.py:371:391
--8<--
```

EPNN parameters `input_keys` are input field names, `output_keys` are output field names, `node_sizes` are node size lists, `activations` are activation function string lists, and `drop_p` is node dropout probability.

### 3.2 Data Generation

This case involves reading data generation, as shown below

``` py linenums="36"
--8<--
examples/epnn/epnn.py:36:41
--8<--
```

``` py linenums="306"
--8<--
examples/epnn/functions.py:306:321
--8<--
```

Here, Data is used to read files to construct the data class, then get_shuffled_data is used to shuffle the data, then the number of shuffled data itrain to be obtained is calculated, and finally get is used to obtain 10 groups of data with the quantity of itrain for each group.

### 3.3 Constraint Construction

Set training dataset and loss calculation function, return fields, code is as follows:

``` py linenums="63"
--8<--
examples/epnn/epnn.py:63:86
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of the supervised constraint. The `"dataset"` field in the configuration represents the training dataset information used, and its various fields represent:

1. `name`: Dataset type, here `"NamedArrayDataset"` means sequentially read dataset;
2. `input`: Input dataset;
3. `label`: Label dataset;

The second parameter is the loss function, here the custom function `train_loss_func` is used.

The third parameter is the equation expression, used to describe how to calculate the constraint target. The calculated value will be stored in the output list according to the specified name, so as to ensure that these values can be used when calculating loss.

The fourth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing.

After the constraint is constructed, encapsulate it into a dictionary with the name we just named as the key for subsequent access.

### 3.4 Validator Construction

Similar to constraints, this problem uses `ppsci.validate.SupervisedValidator` to build a validator. The parameter meanings are also similar to [Constraint Construction](#33-constraint-construction). The only difference is the evaluation metric `metric`. The code is as follows:

``` py linenums="88"
--8<--
examples/epnn/epnn.py:88:103
--8<--
```

### 3.5 Hyperparameter Setting

Next we need to specify the number of training epochs. Here we use 10000 training epochs based on experimental experience. iters_per_epoch is 1.

``` yaml linenums="40"
--8<--
examples/epnn/conf/epnn.yaml:40:41
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, and combined with the ExponentialDecay learning rate adjustment strategy commonly used in machine learning.

Since multiple models are used, multiple optimizers need to be set. For the EPNN network part, `Adam` optimizer needs to be set.

``` py linenums="395"
--8<--
examples/epnn/functions.py:395:404
--8<--
```

Then for the added gkratio parameter, another optimizer needs to be set.

``` py linenums="406"
--8<--
examples/epnn/functions.py:406:413
--8<--
```

Optimizers optimize in order, code summarized as:

``` py linenums="395"
--8<--
examples/epnn/functions.py:395:413
--8<--
```

### 3.7 Custom loss

Since this problem includes unsupervised learning and there is no label data in the data, the loss is calculated based on the returned data of the model, so a custom loss is required. The method is to first define relevant functions, and then pass the function name as a parameter to `FunctionalLoss` and `FunctionalMetric`.

Note that the input and output parameters of the custom loss function need to be consistent with other functions such as `MSE` in PaddleScience, that is, the input is the model output `output_dict` and other dictionary variables, and the loss function output is the loss value `paddle.Tensor`.

The relevant custom loss function is calculated using `MAELoss`, code is

``` py linenums="114"
--8<--
examples/epnn/functions.py:114:126
--8<--
```

### 3.8 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order.

``` py linenums="106"
--8<--
examples/epnn/epnn.py:106:118
--8<--
```

Set eval_during_train to True during model training, and evaluation will be performed after each training.

``` yaml linenums="43"
--8<--
examples/epnn/conf/epnn.yaml:43:43
--8<--
```

Finally start training:

``` py linenums="121"
--8<--
examples/epnn/epnn.py:121:121
--8<--
```

## 4. Complete Code

``` py linenums="1" title="epnn.py"
--8<--
examples/epnn/epnn.py
--8<--
```

## 5. Result Display

The EPNN case was experimented with the parameter configuration of epoch=10000, and the result returned Loss was 0.00471.

The figures below are the Loss, Training error, and Cross validation error graphs for different epochs:

<figure markdown>
  ![loss_trend](https://paddle-org.bj.bcebos.com/paddlescience/docs/EPNN/loss_trend.png){ loading=lazy }
  <figcaption> Training loss graph </figcaption>
</figure>

## 6. References

- [A physics-informed deep neural network for surrogate modeling in classical elasto-plasticity](https://arxiv.org/abs/2204.12088)

- [Reference Code](https://github.com/meghbali/ANNElastoplasticity)
