# FunDiff

<!-- <a href="https://aistudio.baidu.com/projectdetail/7927786" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

!!! warning

    This document only reproduces the turbulence_mass_transfer task in the Fundiff paper.

!!! note

    Please download the tmt.npy dataset file from <https://drive.google.com/drive/folders/1GX5uG_3R-yfuP9nMIk0v7ChuEytYwYPW?usp=drive_link> first.

=== "Model Training Command"

    ``` sh
    python main.py -cn fae.yaml

    python main.py -cn diffusion.yaml FAE.pretrained_model_path=/your/fae/pretrained/model/path
    ```

=== "Model Evaluation Command"

    ``` sh
    python main.py -cn diffusion.yaml mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/fundiff/fundiff_turbulence_mass_transfer_dit_pretrained.pdparams
    ```

<!-- === "Model Export Command"

    None

=== "Model Inference Command"

    None -->

| Pretrained Model | Metrics |
|:--| :--|
| [fundiff_turbulence_mass_transfer_dit_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/fundiff/fundiff_turbulence_mass_transfer_dit_pretrained.pdparams) | Mean relative p error: 0.066<br>Max relative p error: 0.159<br>Min relative p error: 0.029<br>Std relative p error: 0.027<br>Mean relative sdf error: 0.085<br>Max relative sdf error: 0.307<br>Min relative sdf error: 0.022<br>Std relative sdf error: 0.0499 |

## 1. Background Introduction

Recent advances in generative models (especially diffusion models and flow matching) have achieved remarkable success in synthesizing discrete data such as images and videos. However, applying these models to physical applications remains challenging because the physical quantities of interest are continuous functions governed by complex physical laws. This paper introduces $\textbf{FunDiff}$, a novel framework for function space generative models. FunDiff combines latent diffusion processes with functional autoencoder architectures to handle input functions with varying degrees of discretization, generate continuous functions that can be evaluated at arbitrary positions, and seamlessly integrate physical priors. These priors are enforced through architectural constraints or physics-based loss functions, ensuring that generated samples satisfy fundamental physical laws. The authors theoretically establish minimax optimality guarantees for density estimation in function space, showing that diffusion-based estimators can achieve optimal convergence rates under appropriate regularity conditions. Results demonstrate the practical effectiveness of FunDiff in various applications such as fluid dynamics and solid mechanics. Empirical results show that the authors' method is capable of generating physically consistent samples that are highly consistent with the target distribution and exhibit robustness to noisy data and low-resolution data.

## 2. Problem Definition

Given known physical fields $u$ and $v$, solve for physical fields $p$ and $sdf$.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Training FAE

#### 3.1.1 FAE Model Construction

In the FuncDiff model, the FAE module adopts the Perceiver architecture. Its input is the physical field $x$ and query coordinates $coords$, and the output is the value $u$ of a certain physical field at the query coordinates. Therefore, the model construction code is as follows:

``` yaml linenums="36" title="fae.yaml"
--8<--
examples/fundiff/conf/fae.yaml:36:63
--8<--
```

``` py linenums="87" title="main.py"
--8<--
examples/fundiff/main.py:87:95
--8<--
```

#### 3.1.2 Constraint Construction

FAE uses the auto encoder decoder training paradigm, so the label is the input $u$.

``` py linenums="97" title="main.py"
--8<--
examples/fundiff/main.py:97:148
--8<--
```

### 3.2 Training DiT

#### 3.2.1 DiT Model Construction

The model construction of DiT is as follows:

``` yaml linenums="66" title="diffusion.yaml"
--8<--
examples/fundiff/conf/diffusion.yaml:66:76
--8<--
```

``` py linenums="186" title="main.py"
--8<--
examples/fundiff/main.py:186:208
--8<--
```

#### 3.2.2 Constraint Construction

In the FuncDiff model, the training of DiT uses the rectified flow algorithm, and its corresponding mathematical formula is as follows:

$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z}, t, {\epsilon}} \left[ \left\| \hat{\mathbf{v}}_\theta(\mathbf{x}, t) - (\mathbf{z} - \mathbf{x}) \right\|^2 \right]
$$

Its corresponding forward calculation implementation code is as follows:

``` py linenums="44" title="main.py"
--8<--
examples/fundiff/main.py:44:85
--8<--
```

The overall constraint construction is as follows:

``` py linenums="210" title="main.py"
--8<--
examples/fundiff/main.py:210:262
--8<--
```

### 3.3 Hyperparameter Setting

FAE uses 100,000 training steps and an initial learning rate of 0.001.

``` yaml linenums="65" title="fae.yaml"
--8<--
examples/fundiff/conf/fae.yaml:65:80
--8<--
```

DiT uses 100,000 training steps and an initial learning rate of 0.001.

``` yaml linenums="78" title="diffusion.yaml"
--8<--
examples/fundiff/conf/diffusion.yaml:78:92
--8<--
```

### 3.4 Optimizer Construction

The training process will call the optimizer to update model parameters. Both FAE and DiT choose the more commonly used `Adam` optimizer, and combine it with the ExponentialDecay learning rate adjustment strategy commonly used in machine learning.

``` yaml linenums="65" title="fae.yaml"
--8<--
examples/fundiff/conf/fae.yaml:65:80
--8<--
```

``` py linenums="159" title="main.py"
--8<--
examples/fundiff/main.py:159:172
--8<--
```

``` yaml linenums="78" title="diffusion.yaml"
--8<--
examples/fundiff/conf/diffusion.yaml:78:104
--8<--
```

``` py linenums="273" title="main.py"
--8<--
examples/fundiff/main.py:273:286
--8<--
```

### 3.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="174"
--8<--
examples/fundiff/main.py:174:182
--8<--
```

``` py linenums="288"
--8<--
examples/fundiff/main.py:288:296
--8<--
```

## 4. Complete Code

``` py linenums="1" title="main.py"
--8<--
examples/fundiff/main.py
--8<--
```

## 5. Result Display

Evaluated on the test set, and some results are displayed:

<figure markdown>
  ![result_of_sample_2.jpg](https://paddle-org.bj.bcebos.com/paddlescience/docs/fundiff/result_of_sample_2.png){ loading=lazy }
</figure>

It can be seen that for functions $p(x, coord | u,v)$ and $sdf(x, coord | u,v)$, the model's prediction results are basically consistent with the reference results.

## 6. References

- [FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling](https://arxiv.org/abs/2506.07902v1)
- [fundiff github](https://github.com/sifanexisted/fundiff/tree/main)
