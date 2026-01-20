# NLS-MB

<!-- <a href="TODO" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

=== "Model Training Command"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py
    # rogue wave
    python NLS-MB_optical_rogue_wave.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_soliton_pretrained.pdparams
    # rogue wave
    python NLS-MB_optical_rogue_wave.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_rogue_wave_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py mode=export
    # rogue wave
    python NLS-MB_optical_rogue_wave.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # soliton
    python NLS-MB_optical_soliton.py mode=infer
    # rogue wave
    python NLS-MB_optical_rogue_wave.py mode=infer

    ```

| Pretrained Model | Metrics |
|:--| :--|
| [NLS-MB_soliton_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_soliton_pretrained.pdparams) | Residual/loss: 0.00000<br>Residual/MSE.Schrodinger_1: 0.00000<br>Residual/MSE.Schrodinger_2: 0.00000<br>Residual/MSE.Maxwell_1: 0.00000<br>Residual/MSE.Maxwell_2: 0.00000<br>Residual/MSE.Bloch: 0.00000 |
| [NLS-MB_optical_rogue_wave.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_optical_rogue_wave.pdparams) | Residual/loss: 0.00001<br>Residual/MSE.Schrodinger_1: 0.00000<br>Residual/MSE.Schrodinger_2: 0.00000<br>Residual/MSE.Maxwell_1: 0.00000<br>Residual/MSE.Maxwell_2: 0.00000<br>Residual/MSE.Bloch: 0.00000 |

## 1. Background Introduction

Nonlinear localized wave dynamics, as an important branch of nonlinear science, covers basic forms of nonlinear localized waves such as solitons, breathers and rogue waves. Laser mode-locking technology provides an experimental verification platform for these theoretically predicted nonlinear localized waves. Through this technology, people have observed rich nonlinear phenomena such as soliton molecules and rogue waves, further promoting the research of nonlinear localized waves. Currently, research in this field has penetrated into many physical fields such as fluid mechanics, nonlinear optics, Bose-Einstein condensation (BEC), and plasma physics. In the field of optical fibers, the research of nonlinear dynamics is based on the principles of optical fiber optical devices, information processing, material design and signal transmission, and has played a key role in the development of fiber lasers, amplifiers, waveguides and communication technologies. The propagation dynamics of optical pulses in optical fibers are governed by nonlinear partial differential equations (such as the nonlinear Schrödinger equation NLSE). When dispersion and nonlinear effects coexist, these equations are often difficult to solve analytically. Therefore, the split-step Fourier method and its improved versions are widely used to study nonlinear effects in optical fibers. Its advantage lies in simple implementation and high relative accuracy. However, for long-distance and highly nonlinear scenarios, in order to meet accuracy requirements, the step size of the split-step Fourier method must be significantly reduced, which undoubtedly increases computational complexity, resulting in a huge number of grid point sets in the time domain and a long calculation process. PINN shows better performance than data-driven methods with much less data, and the computational complexity (expressed in multiples) is usually two orders of magnitude lower than SFM.

## 2. Problem Definition

In erbium-doped fiber, the propagation properties of optical pulses can be described by the coupled NLS-MB equations, which have the form

$$
\begin{cases}
   \dfrac{\partial E}{\partial x} = i \alpha_1 \dfrac{\partial^2 E}{\partial t ^2} - i \alpha_2 |E|^2 E+2 p \\
   \dfrac{\partial p}{\partial t} = 2 i \omega_0 p+2 E \eta \\
   \dfrac{\partial \eta}{\partial t} = -(E p^* + E^* p)
\end{cases}
$$

Among them, *x*, *t* represent the normalized propagation distance and time respectively, the complex envelope *E* is the slowly varying electric field, *p* is the measure of polarization of the resonant medium, $\eta$ represents the degree of population inversion, and the symbol * represents complex conjugation. $\alpha_1$ is the group velocity dispersion parameter, $\alpha_2$ is the Kerr nonlinearity parameter, and is the offset measuring the resonance frequency. The NLS-MB system was first proposed by Maimistov and Manykin to describe the propagation of ultrashort pulses in Kerr nonlinear media. This system also plays an important role in solving the problem that optical fiber loss limits its transmission distance. In this equation, it describes the mixed state of self-induced transparency solitons and NLS solitons, called SIT-NLS solitons. These two types of solitons can coexist, and there have been many studies on their application in optical fiber communications.

### 2.1 Optical soliton

In the anomalous dispersion region of optical fibers, due to the interaction of dispersion and nonlinear effects, a very compelling phenomenon can be produced - optical solitons. "Soliton" is a special wave packet that can transmit long distances without deformation. Solitons have been widely studied in many branches of physics. The solitons in optical fibers discussed in this case not only have basic theoretical research value, but also have practical applications in optical fiber communications.

$$
\begin{gathered}
  E(x,t) = \frac{{2\exp ( - 2it)}}{{\cosh (2t + 6x)}},  \\
  p(x,t) = \frac{{\exp ( - 2it)\left\{ {\exp ( - 2t - 6x) - \exp (2t + 6x)} \right\}}}{{\cosh {{(2t + 6x)}^2}}},  \\
  \eta (x,t) = \frac{{\cosh {{(2t + 6x)}^2} - 2}}{{\cosh {{(2t + 6x)}^2}}}.
\end{gathered}
$$

We consider the computational domain as $[−1, 1] × [−1, 1]$. We first determine the optimization strategy. There are $200$ points on each boundary, i.e., $N_b = 2 × 200$. In order to calculate the equation loss of NLS-MB, $20,000$ points are randomly selected within the domain.

### 2.2 Optical rogue wave

Optical rogue waves are a phenomenon in optics, similar to rogue waves in the ocean, but in optical systems. They are light waves that appear suddenly and have unusually high amplitudes. Optical rogue waves have some potential applications, especially in the fields of optical communications and laser technology. Some studies suggest that they can be used to enhance the transmission and processing of optical signals, or to generate ultrashort pulse lasers.
We consider the computational domain as $[−0.5, 0.5] × [−2.5, 2.5]$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

This paper uses the classic PINN MLP model for training.

``` py linenums="94"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:94:95
--8<--
```

### 3.2 Equation Construction

Since Optical soliton uses the NLS-MB equation, `NLSMB` built in PaddleScience can be used directly.

``` py linenums="97"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:97:100
--8<--
```

### 3.3 Computational Domain Construction

In this paper, the Optical soliton problem acts on the spatiotemporal region of space (-1.0, 1.0), time (-1.0, 1.0),
so the spatiotemporal geometry `time_interval` built in PaddleScience can be used directly as the computational domain.

``` py linenums="108"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:108:114
--8<--
```

### 3.4 Constraint Construction

Since the dataset is an analytical solution, we first construct the analytical solution function

``` py linenums="26"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:26:44
--8<--
```

#### 3.4.1 Interior Point Constraint

Taking `InteriorConstraint` acting on internal points as an example, the code is as follows:

``` py linenums="150"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:150:169
--8<--
```

The first parameter of `InteriorConstraint` is the equation (system) expression, used to describe how to calculate the constraint target. Here, fill in `equation["NLS-MB"].equations` instantiated in the [3.2 Equation Construction](#32) chapter;

The second parameter is the target value of the constraint variable. In this problem, it is hoped that each equation of NLS-MB is optimized to 0;

The third parameter is the computational domain on which the constraint equation acts. Here, fill in `geom["time_interval"]` instantiated in the [3.3 Computational Domain Construction](#33) chapter;

The fourth parameter is the sampling configuration on the computational domain. Here `batch_size` is set to `20000`.

The fifth parameter is the loss function. Here the commonly used MSE function is selected, and `reduction` is set to `"mean"`, that is, the mean square error of all data points involved in the calculation will be calculated;

The sixth parameter is the name of the constraint condition. Each constraint condition needs to be named for subsequent indexing. Here it is named "EQ".

#### 3.4.2 Boundary Constraint

Since our boundary points and initial value points have analytical solutions, we use supervised constraints

``` py linenums="171"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:171:176
--8<--
```

### 3.5 Hyperparameter Setting

Next, the number of training epochs and learning rate need to be specified. Here, based on experimental experience, 50,000 training epochs and an initial learning rate of 0.001 are used.

``` yaml linenums="41"
--8<--
examples/NLS-MB/conf/NLS-MB_soliton.yaml:41:54
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected.

``` py linenums="184"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:184:185
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.GeometryValidator` is used to construct the validator.

``` py linenums="187"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:187:208
--8<--
```

### 3.8 Visualizer Construction

After the model training is completed, we can take points in the computational domain for prediction, manually calculate the amplitude, and visualize the results.

``` py linenums="255"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:255:269
--8<--
```

### 3.9 Model Training, Evaluation and Visualization

#### 3.9.1 Training with Adam

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training, evaluation, and visualization.

``` py linenums="210"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:210:227
--8<--
```

#### 3.9.2 Fine-tuning with L-BFGS [Optional]

After training with the `Adam` optimizer, we can replace the optimizer with the second-order optimizer `L-BFGS` to continue training for a small number of epochs (here we use 10% of the `Adam` optimization epochs), thereby further improving model accuracy.

``` py linenums="229"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py:229:253
--8<--
```

???+ tip "Tip"

    After training with conventional optimizers, using `L-BFGS` to fine-tune for a small number of epochs can effectively further improve model accuracy in most scenarios.

## 4. Complete Code

``` py linenums="1" title="NLS-MB_optical_soliton.py"
--8<--
examples/NLS-MB/NLS-MB_optical_soliton.py
--8<--
```

## 5. Result Display

### 5.1 optical_soliton

<figure markdown>
  ![optical_soliton](https://paddle-org.bj.bcebos.com/paddlescience/docs/NLS-MB/pred_optical_soliton.png){ loading=lazy}
  <figcaption>Comparison between analytical solution results and PINN prediction results, from top to bottom: slowly varying electric field (E), resonance bias (p) and population inversion degree (eta)</figcaption>
</figure>

### 5.2 optical_rogue_wave

<figure markdown>
  ![optical_rogue_wave](https://paddle-org.bj.bcebos.com/paddlescience/docs/NLS-MB/pred_optical_rogue_wave.png){ loading=lazy}
  <figcaption>Comparison between analytical solution results and PINN prediction results, from top to bottom: slowly varying electric field (E), resonance bias (p) and population inversion degree (eta)</figcaption>
</figure>

It can be seen that the PINN prediction results are basically consistent with the analytical solution results.

## 6. References

1. [S.-Y. Xu, Q. Zhou, and W. Liu, Prediction of Soliton Evolution and Equation Parameters for NLS–MB Equation Based on the phPINN Algorithm, Nonlinear Dyn (2023)](https://doi.org/10.1007/s11071-023-08824-w).
