# NSFNet4

<a href="https://aistudio.baidu.com/projectdetail/7305374" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # VP_NSFNet4
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip -P ./data/
    unzip ./data/NSF4_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip --create-dirs -o ./data/NSF4_data.zip
    # unzip ./data/NSF4_data.zip
    python VP_NSFNet4.py data_dir=./data/

    ```

=== "Model Evaluation Command"

    ``` sh
    # VP_NSFNet4
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip -P ./data/
    unzip ./data/NSF4_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip --create-dirs -o ./data/NSF4_data.zip
    # unzip ./data/NSF4_data.zip
    python VP_NSFNet4.py mode=eval data_dir=./data/ EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet4.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python VP_NSFNet4.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # VP_NSFNet4
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip -P ./data/
    unzip ./data/NSF4_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip --create-dirs -o ./data/NSF4_data.zip
    # unzip ./data/NSF4_data.zip
    python VP_NSFNet4.py mode=infer
    ```

## 1. Background Introduction

In recent years, deep learning has achieved remarkable achievements in many fields, especially in computer vision and natural language processing. Inspired by the rapid development of deep learning and based on the powerful function approximation ability of deep learning, neural networks have also achieved success in the field of scientific computing. Current research is mainly divided into two categories. One is to add physical information and physical constraints to the loss function to train neural networks, represented by PINN and Deep Ritz Net. The other is data-driven deep neural network operators, represented by FNO and DeepONet. These methods have been widely used in scientific practice, such as weather forecasting, quantum chemistry, biological engineering, and computational fluid dynamics. In order to fully explore the ability of PINN to solve fluid equations, the author of this reproduction [paper](https://arxiv.org/abs/2003.06496) designed NSFNets, and successively used two-dimensional and three-dimensional Navier-Stokes equations with analytical or numerical solutions, as well as datasets solved with high precision using the DNS method as references, to perform forward problem solving training. The paper experiments show that PINN has excellent numerical solving capabilities for incompressible Navier-Stokes equations. The main goal of this project is to use PaddleScience to reproduce the code for high-precision solving of Navier-Stokes equations implemented in the paper.

## 2. Problem Definition

The classic PINN model is used for this problem, so I won't go into details.

Mainly introduce several types of Navier-Stokes equations solved:

The incompressible Navier-Stokes equation can be expressed as:

$$\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u} \cdot \nabla) \mathbf{u} =-\nabla p+\frac{1}{Re} \nabla^2 \mathbf{u} \quad \text { in } \Omega, $$

$$\nabla \cdot \mathbf{u} =0 \quad  \text { in } \Omega, $$

$$\mathbf{u} =\mathbf{u}_{\Gamma} \quad \text { on } \Gamma_D, $$

$$\frac{\partial \mathbf{u}}{\partial n} =0 \quad \text { on } \Gamma_N.$$

### 2.1 JHTDB Dataset

The dataset is a high-precision dataset of three-dimensional incompressible forced isotropic turbulence at Re=999.35 solved using DNS. Detailed parameters can be found in [readme](https://turbulence.pha.jhu.edu/Forced_isotropic_turbulence.aspx).

## 3. Problem Solving

### 3.1 Model Construction

This paper uses the classic PINN MLP model for training.

``` py linenums="137"
--8<--
examples/nsfnet/VP_NSFNet4.py:137:137
--8<--
```

### 3.2 Data Generation

Boundary points, initial value points, and internal points for calculating residuals are taken successively (see section 3.3 of [paper](https://arxiv.org/abs/2003.06496) for specific selection method) and test points are generated.

``` py linenums="139"
--8<--
examples/nsfnet/VP_NSFNet4.py:139:167
--8<--
```

### 3.3 Normalization

To change the selected smaller rectangular area into a cubic area, we embed the normalization function before the network training.

``` py linenums="169"
--8<--
examples/nsfnet/VP_NSFNet4.py:169:174
--8<--
```

### 3.4 Constraint Construction

Since our boundary points and initial value points have analytical solutions, we use supervised constraints, where alpha and beta are the weights of the loss function, which are both taken as 100 in this code, consistent with the description in the paper.

``` py linenums="226"
--8<--
examples/nsfnet/VP_NSFNet4.py:226:237
--8<--
```

Use internal points to construct residual constraints of Navier-Stokes equations

``` py linenums="239"
--8<--
examples/nsfnet/VP_NSFNet4.py:239:262
--8<--
```

### 3.5 Validator Construction

Use the test set generated during data generation for model evaluation:

``` py linenums="271"
--8<--
examples/nsfnet/VP_NSFNet4.py:271:276
--8<--
```

### 3.6 Optimizer Construction

Consistent with the description in the paper, we use piecewise learning rate to construct the Adam optimizer, where the number of training epochs can be adjusted by adjusting epoch_list.

``` py linenums="281"
--8<--
examples/nsfnet/VP_NSFNet4.py:281:283
--8<--
```

### 3.7 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`.

``` py linenums="284"
--8<--
examples/nsfnet/VP_NSFNet4.py:284:302
--8<--
```

Finally start training:

``` py linenums="303"
--8<--
examples/nsfnet/VP_NSFNet4.py:303:304
--8<--
```

## 4. Complete Code

``` py linenums="1" title="NSFNet.py"
--8<--
examples/nsfnet/VP_NSFNet4.py
--8<--
```

## 5. Result Display

### NSFNet4

As shown in the figure, the error of NSFNet in time is relatively stable, and the error accumulation problem often found in traditional methods does not appear. Among them, although the velocities in the three directions were not weighted during the training process, it can be seen from the training results that the neural network has the best approximation effect on the first velocity direction u, followed by the third velocity direction w, and the second velocity v has the worst approximation effect and a relatively obvious error accumulation phenomenon appears.
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/error.jpg)

As shown in the figure, in the contour map of the y-z plane at x=12.47, the first one is the contour map of velocity u, the second is the contour map of velocity v, the third is the contour map of velocity w, and the fourth is the contour map of velocity p. It can be seen that the contour map of velocity u is relatively smoother than v, w, p.
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/x%3D0%20plane.png)

As shown in the figure, in the contour map of the x-y plane at z=4.61, the first one is the contour map of velocity u, the second is the contour map of velocity v, the third is the contour map of velocity w, and the fourth is the contour map of velocity p. It can be seen that the contour map of velocity u is relatively smoother than v, w, p.
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/z%3D0%20plane.png)

In summary, although u, v, w three velocity directions all require neural network training, for the JHTDB dataset, u direction data is smoother and easier to be learned by neural networks. Therefore, in subsequent research, we can try to divide and conquer the components in three different directions, increase the training intensity of complex component directions, and reduce the training intensity of simple component directions.

## 6. Results Description

We use PINN to numerically solve the incompressible Navier-Stokes equations. In PINN, randomly selected time and space coordinates are used as input values, corresponding velocity fields and pressure fields are used as output values, and initial values, boundary conditions are used as supervised constraints and the Navier-Stokes equation itself is used as unsupervised constraints added to the loss function for training. We use high-precision JHTDB dataset for training. Through the decrease of the loss function, the convergence of the neural network in solving the Navier-Stokes equation can be proven, indicating that PINN possesses the ability to solve incompressible forced isotropic turbulence. The experimental results show that PINN can well approximate the corresponding high-precision incompressible forced isotropic turbulence dataset, and we found that increasing the weights of boundary constraints and initial value constraints can enable the neural network to have better approximation effects. In contrast, within the allowable error range, using PINN to solve the Navier-Stokes equation is faster than the original DNS method inference speed.

## 7. References

- [NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

- [Github NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
