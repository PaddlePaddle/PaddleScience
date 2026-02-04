# Extended Physics-Informed Neural Networks (XPINNs)

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat -P ./data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat --create-dirs -o ./data/XPINN_2D_PoissonEqn.mat
    python xpinn.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat -P ./data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat --create-dirs -o ./data/XPINN_2D_PoissonEqn.mat
    python xpinn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/XPINN/xpinn_pretrained.pdparams
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [xpinn_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/XPINN/xpinn_pretrained.pdparams) | L2Rel.l2_error: 0.04226 |

## 1. Background Introduction

Solving partial differential equations (PDEs) is a fundamental physical problem. With the rapid development of artificial intelligence technology, using deep learning to solve partial differential equations has become a new research trend. [XPINNs (Extended Physics-Informed Neural Networks)](https://doi.org/10.4208/cicp.OA-2020-0164) is a generalized spatiotemporal domain decomposition method applicable to Physics-Informed Neural Networks (PINNs) to solve nonlinear partial differential equations on arbitrarily complex geometric domains.

XPINNs effectively improves the parallelism of the model through generalized spatiotemporal domain decomposition, and supports highly irregular, convex/non-convex spatiotemporal domain decomposition, and the interface conditions are simple. XPINNs can be extended to any type of partial differential equation, regardless of the physical properties of the equation.

Accurately solving high-dimensional complex equations has become one of the biggest challenges in scientific computing. The advantages of XPINNs make it a suitable method for simulating complex equations.

## 2. Problem Definition

2D Poisson Equation:

$$ \Delta u = f(x, y),  x,y \in \Omega \subset R^2$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Download

As shown in the figure below, the dataset contains data for three subregions of the computational domain: the boundary and residual points of the red region; the interface of the yellow region; and the interface of the green region.

<figure markdown>
  ![](https://ai-studio-static-online.cdn.bcebos.com/27ef9bddb0604ef58007f9be6a3364ac0336f476ac894233a6f6b1c97ab68c5c)
  <figcaption>Three subregions of 2D Poisson Equation</figcaption>
</figure>

The boundary expression of the computational domain is as follows.

$$ \gamma =1.5+0.14 sin(4θ)+0.12 cos(6θ)+0.09 cos(5θ), θ \in [0,2π) $$

The interface expressions of the red region and the yellow region are as follows.

$$ \gamma_1 =0.5+0.18 sin(3θ)+0.08 cos(2θ)+0.2 cos(5θ), θ \in [0,2π)$$

$$ \gamma_2 =0.34+0.04 sin(5θ)+0.18 cos(3θ)+0.1 cos(6θ), θ \in [0,2π) $$

Execute the following command to download and unzip the dataset.

``` sh
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat -P ./data/
```

### 3.2 Model Construction

In this problem, we use the neural network `MLP` as the model, and define three `MLP`s in the model code as models for the three subregions respectively.

``` py linenums="301"
--8<--
examples/xpinn/xpinn.py:301:302
--8<--
```

When training the model, we will use the XPINN method to calculate the model loss for each subregion separately.

<figure markdown>
  ![](https://ai-studio-static-online.cdn.bcebos.com/d30ac172809343c5ac9d2b44d3657efd8e30949fd8f44174bf6221e14c31f6bf)
  <figcaption>Training process of XPINN subnetwork</figcaption>
</figure>

### 3.3 Constraint Construction

In this case, we use a supervised dataset to train the model, so we need to construct supervised constraints.

Before defining constraints, we need to specify relevant configurations such as the dataset path and store this information in the corresponding YAML file, as shown below.

``` yaml linenums="44"
--8<--
examples/xpinn/conf/xpinn.yaml:44:45
--8<--
```

Then define the calculation process of the training loss function and call the XPINN method to calculate the loss, as shown below.

``` py linenums="130"
--8<--
examples/xpinn/xpinn.py:130:191
--8<--
```

Finally, construct the supervised constraint as shown below.

``` py linenums="304"
--8<--
examples/xpinn/xpinn.py:304:311
--8<--
```

### 3.4 Hyperparameter Setting

Set training epochs and other parameters, as shown below.

``` yaml linenums="84"
--8<--
examples/xpinn/conf/xpinn.yaml:84:89
--8<--
```

### 3.5 Optimizer Construction

The training process calls the optimizer to update model parameters. The commonly used `Adam` optimizer is selected here.

``` py linenums="337"
--8<--
examples/xpinn/xpinn.py:337:338
--8<--
```

### 3.6 Validator Construction

During the training process, the training status of the current model is usually evaluated using the validation set (test set) at a certain epoch interval. Therefore, `ppsci.validate.SupervisedValidator` is used to construct the validator.

``` py linenums="324"
--8<--
examples/xpinn/xpinn.py:324:335
--8<--
```

The evaluation metric is the L2 relative error value of the prediction result and the real result. Here, a custom metric calculation function needs to be defined, as shown below.

``` py linenums="194"
--8<--
examples/xpinn/xpinn.py:194:219
--8<--
```

### 3.7 Model Training and Evaluation

After completing the above settings, just pass the above instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="340"
--8<--
examples/xpinn/xpinn.py:340:350
--8<--
```

### 3.8 Result Visualization

After training, the program will predict the data in the test set and visualize the results in the form of pictures, as shown below.

``` py linenums="352"
--8<--
examples/xpinn/xpinn.py:352:376
--8<--
```

## 4. Complete Code

``` py linenums="1" title="xpinn.py"
--8<--
examples/xpinn/xpinn.py
--8<--
```

## 5. Result Display

The prediction results, reference results and relative errors of each point in the computational domain are shown below.

<figure markdown>
  ![](https://ai-studio-static-online.cdn.bcebos.com/3f3b0dda860041009c7f87aae099871d85dc9694bd924608afa0af2c6101d37e)
  <figcaption>Comparison of prediction results and reference results</figcaption>
</figure>

It can be seen that the model prediction result is close to the real result. If the number of training epochs is increased, the model accuracy will be further improved.

## 6. References

- [Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations](https://doi.org/10.4208/cicp.OA-2020-0164)
