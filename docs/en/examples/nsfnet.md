# NSFNets

<a href="https://aistudio.baidu.com/projectdetail/7305373" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py

    # VP_NSFNet2
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat --create-dirs -o ./data/cylinder_nektar_wake.mat
    python VP_NSFNet2.py data_dir=./data/cylinder_nektar_wake.mat

    # VP_NSFNet3
    python VP_NSFNet3.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py mode=eval pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet1.pdparams

    # VP_NSFNet2
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat --create-dirs -o ./data/cylinder_nektar_wake.mat

    python VP_NSFNet2.py mode=eval data_dir=./data/cylinder_nektar_wake.mat pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet2.pdparams

    # VP_NSFNet3
    python VP_NSFNet3.py mode=eval pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet3.pdparams
    ```

## 1. Background Introduction
In recent years, deep learning has achieved remarkable achievements in many fields, especially in computer vision and natural language processing. Inspired by the rapid development of deep learning and based on the powerful function approximation ability of deep learning, neural networks have also achieved success in the field of scientific computing. Current research is mainly divided into two categories. One is to add physical information and physical constraints to the loss function to train neural networks, represented by PINN and Deep Ritz Net. The other is data-driven deep neural network operators, represented by FNO and DeepONet. These methods have been widely used in scientific practice, such as weather forecasting, quantum chemistry, biological engineering, and computational fluid dynamics. In order to fully explore the ability of PINN to solve fluid equations, the author of this reproduction [paper](https://arxiv.org/abs/2003.06496) designed NSFNets, and successively used two-dimensional and three-dimensional Navier-Stokes equations with analytical or numerical solutions, as well as datasets solved with high precision using the DNS method as references, to perform forward problem solving training. The paper experiments show that PINN has excellent numerical solving capabilities for incompressible Navier-Stokes equations. The main goal of this project is to use PaddleScience to reproduce the code for high-precision solving of Navier-Stokes equations implemented in the paper.

## 2. Problem Definition
The classic PINN model is used for this problem, so I won't go into details.

Mainly introduce several types of Navier-Stokes equations solved:

The incompressible Navier-Stokes equation can be expressed as:

$$\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u} \cdot \nabla) \mathbf{u} =-\nabla p+\frac{1}{Re} \nabla^2 \mathbf{u} \quad \text { in } \Omega,$$

$$\nabla \cdot \mathbf{u} =0 \quad  \text { in } \Omega,$$

$$\mathbf{u} =\mathbf{u}_{\Gamma} \quad \text { on } \Gamma_D,$$

$$\frac{\partial \mathbf{u}}{\partial n} =0 \quad \text { on } \Gamma_N.$$

### 2.1 Kovasznay flow(NSFNet1)
We use Kovasznay flow as the first test case to demonstrate the performance of NSFnets. This two-dimensional steady Navier-Stokes flow has the following analytical solution:

$$u(x, y)=1-e^{\lambda x} \cos (2 \pi y),$$

$$v(x, y)=\frac{\lambda}{2 \pi} e^{\lambda x} \sin (2 \pi y),$$

$$p(x, y)=\frac{1}{2}\left(1-e^{2 \lambda x}\right),$$

where

$$\lambda=\frac{1}{2 \nu}-\sqrt{\frac{1}{4 \nu^2}+4 \pi^2}, \quad \nu=\frac{1}{Re}=\frac{1}{40} .$$

We consider the computational domain as $[−0.5, 1.0] × [−0.5, 1.5]$. We first determine the optimization strategy. There are $101$ points with fixed spatial coordinates on each boundary, i.e., $Nb = 4 × 101$. To calculate the equation loss of NSFnet, $2,601$ points are randomly selected within the domain. This steady flow has no initial conditions. We use the Adam optimizer to provide a better set of initial neural network learnable variables. Then, L-BFGS-B is used to fine-tune the neural network to obtain higher accuracy. The training process of L-BFGS-B terminates automatically based on the increment tolerance. In this section, we use $3 × 10^4$ Adam iterations with a learning rate of $10^{−3}$ before L-BFGS-B training. The effect of the number of Adam iterations is discussed in Figure A.1 of Appendix A of the paper, and we also studied the performance of NSFnet in terms of the number of sampling points and boundary points.

### 2.2 Cylinder wake (NSFNet2)
Here we use NSFnets to simulate $2D$ vortex shedding behind a cylinder at $Re = 100$. The cylinder is placed at $(x, y) = (0, 0)$ with diameter $D = 1$. High-fidelity DNS data from [$M. Raissi 2019$](https://www.sciencedirect.com/science/article/am/pii/S0021999118307125) is used as a reference and provides boundary and initial data for NSFnet training. We consider the domain defined by $[1, 8] × [−2, 2]$, with a time interval of $[0, 7]$ (over one shedding period) and a time step $Δt = 0.1$. For training data, we place $100$ points along the $x$ direction boundary and $50$ points along the y direction boundary to control boundary conditions, and use $140,000$ spatiotemporal scattered points within the domain to calculate residuals. NSFnet contains $10$ hidden layers with $100$ neurons each. [Cylinder wake AIstudio dataset link](https://aistudio.baidu.com/datasetdetail/236213).

### 2.3 Beltrami flow (NSFNet3)
$$u(x, y, z, t)= -a\left[e^{a x} \sin (a y+d z)+e^{a z} \cos (a x+d y)\right] e^{-d^2 t}, $$

$$v(x, y, z, t)= -a\left[e^{a y} \sin (a z+d x)+e^{a x} \cos (a y+d z)\right] e^{-d^2 t}, $$

$$w(x, y, z, t)= -a\left[e^{a z} \sin (a x+d y)+e^{a y} \cos (a z+d x)\right] e^{-d^2 t}, $$

$$p(x, y, z, t)= -\frac{1}{2} a^2\left[e^{2 a x}+e^{2 a y}+e^{2 a z}+2 \sin (a x+d y) \cos (a z+d x) e^{a(y+z)} +2 \sin (a y+d z) \cos (a x+d y) e^{a(z+x)} +2 \sin (a z+d x) \cos (a y+d z) e^{a(x+y)}\right] e^{-2 d^2 t}.$$

## 3. Problem Solving
### 3.1 Model Construction
This paper uses the classic PINN MLP model for training.
``` py linenums="175"
--8<--
examples/nsfnet/VP_NSFNet3.py:175:175
--8<--
```
### 3.2 Hyperparameter Setting
Specify the number of residual points, boundary points, initial value points, and the weights of boundary loss function and initial value loss function can be specified
``` py linenums="178"
--8<--
examples/nsfnet/VP_NSFNet3.py:178:186
--8<--
```
### 3.3 Data Generation
Since the dataset is an analytical solution, we first construct the analytical solution function
``` py linenums="10"
--8<--
examples/nsfnet/VP_NSFNet3.py:10:51
--8<--
```

Then take boundary points, initial value points, and internal points for calculating residuals (see section 3.3 of [paper](https://arxiv.org/abs/2003.06496) for specific selection method) and generate test points.
``` py linenums="187"
--8<--
examples/nsfnet/VP_NSFNet3.py:187:214
--8<--
```
### 3.4 Constraint Construction
Since our boundary points and initial value points have analytical solutions, we use supervised constraints
``` py linenums="266"
--8<--
examples/nsfnet/VP_NSFNet3.py:266:277
--8<--
```

where alpha and beta are the weights of the loss function, which are both taken as 100 in this code, consistent with the description in the paper.

Use internal points to construct residual constraints of Navier-Stokes equations
``` py linenums="280"
--8<--
examples/nsfnet/VP_NSFNet3.py:280:297
--8<--
```
### 3.5 Validator Construction
Use the test set generated during data generation for model evaluation:
``` py linenums="305"
--8<--
examples/nsfnet/VP_NSFNet3.py:305:319
--8<--
```

### 3.6 Optimizer Construction
Consistent with the description in the paper, we use piecewise learning rate to construct the Adam optimizer, where the number of training epochs can be adjusted by adjusting _epoch_list_.
``` py linenums="321"
--8<--
examples/nsfnet/VP_NSFNet3.py:321:331
--8<--
```

### 3.7 Model Training and Evaluation
After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`.

``` py linenums="333"
--8<--
examples/nsfnet/VP_NSFNet3.py:333:350
--8<--
```

Finally start training:

``` py linenums="351"
--8<--
examples/nsfnet/VP_NSFNet3.py:351:352
--8<--
```


## 4. Complete Code
NSFNet1:
``` py linenums="1" title="NSFNet1.py"
--8<--
examples/nsfnet/VP_NSFNet1.py
--8<--
```
NSFNet2:
``` py linenums="1" title="NSFNet2.py"
--8<--
examples/nsfnet/VP_NSFNet2.py
--8<--
```
NSFNet3:
``` py linenums="1" title="NSFNet3.py"
--8<--
examples/nsfnet/VP_NSFNet3.py
--8<--
```
## 5. Result Display

Mainly refer to paper data and reference code data.

### 5.1 NSFNet1(Kovasznay flow)

| velocity | paper | code | PaddleScience | NN size |
|:--|:--|:--|:--|:--|
| u | 0.072% | 0.080% | 0.056% | 4 × 50 |
| v | 0.058% | 0.539% | 0.399% | 4 × 50 |
| p | 0.027% | 0.722% | 1.123% | 4 × 50 |

As shown in the table, columns 2, 3, and 4 are the $L_{2}$ errors reproduced by the paper, other developers, and PaddleScience respectively. The $L_{2}$ errors of Kovasznay flow velocity $u$, $v$ in $x$, $y$ directions are 0.055% and 0.399%, which are better than the paper (Table 2) and reference code.

### 5.2 NSFNet2(Cylinder wake)
The $L_{2}$ error of Cylinder wake predicted at $t=0$. As shown in the table, the $L_{2}$ errors of Cylinder flow velocity $u$, $v$ in $x$, $y$ directions are 0.138% and 0.488%, which are close to the paper (Figure 9) and code.

| velocity | paper (VP-NSFnet, $\alpha=\beta=1$) | paper (VP-NSFnet, dynamic weights)  | code | PaddleScience  | NN size |
|:--|:--|:--|:--|:--|:--|
| u | 0.09% | 0.01% | 0.403% | 0.138% | 4 × 50 |
| v | 0.25% | 0.05% | 1.5%   | 0.488% | 4 × 50 |
| p | 1.9%  | 0.8%  |  /     | /      | 4 × 50 |

The velocity field of the NSFNet2 (2D Cylinder Flow) case is shown in the figure below. The two pictures in the first row are the cylinder wake region. The picture in the first row shows the numerical distribution of the flow velocity $u$ in the $x$ streamline direction. The left side is the DNS high-fidelity data as a reference, and the right side is the neural network predicted value. Blue is a smaller value and green is a larger value. The distribution area is $x=[1,8]$, $y=[-2, 2]$. The picture in the second row shows the distribution of the flow velocity $v$ in the $y$ spanwise direction. The left side is the DNS high-fidelity data reference value, and the right side is the neural network predicted value. The distribution area is $x=[1,8]$, $y=[-2, 2]$.

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/Cylinder%20wake.gif)

Based on the velocity field, we can calculate the vorticity field. As shown in the figure, it is the contour map of the vorticity field of the NSFNet2 (2D Cylinder Flow) case at time $t=4.0$. We calculate the vorticity map as shown in the figure based on the flow velocities $u$, $v$ in the $x$, $y$ directions through the vorticity calculation formula. The vorticity structure has good continuity and is consistent with the paper. The calculation distribution area is $x=[1, 8]$, $y=[-2, 2]$.

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/NSFNet2_vorticity.png)
### 5.3 NSFNet3(Beltrami flow)
The relative errors of the test dataset (analytical solution) are shown in the table. The $L_{2}$ errors of Beltrami flow velocities $u$, $v$, $w$ in $x$, $y$, $z$ directions are 0.059%, 0.082% and 0.0732%, which are better than the code data.

| velocity |  code(NN size:10×100) | PaddleScience (NN size:10×100)|
|:--|:--|:--|
| u | 0.0766% | 0.059% |
| v | 0.0689% | 0.082% |
| w | 0.1090% | 0.073% |
| p | /       | /      |

The predicted relative errors of Beltrami flow at time $ t=1 $ on the $ z=0 $ plane are shown in the table. The $L_{2}$ errors of Beltrami flow velocities $u, v, w$ in $x, y, z$ directions are 0.115%, 0.199% and 0.217%, and the $L_{2}$ error of pressure $p$ is 0.1.986%, which are all better than the paper data (Table 4. VP).

| velocity | paper(NN size:7×50) | PaddleScience(NN size:10×100) |
|:--|:--|:--|
| u | 0.1634±0.0418% | 0.115% |
| v | 0.2185±0.0530% | 0.199% |
| w | 0.1783±0.0300% | 0.217% |
| p | 8.9335±2.4350% | 1.986% |

Beltrami flow velocity field, as shown in the figure, the left side is the analytical solution reference value, the right side is the neural network predicted value, blue is a smaller value, red is a larger value, the distribution area is $x=[-1,1]$, $y=[-1, 1]$, the first row is the distribution of flow velocity $u$ in the $x$ direction, the second row is the distribution of flow velocity $v$ in the $y$ direction, and the third row is the distribution of flow velocity $w$ in the $z$ direction.

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/Beltrami%20flow.gif)

## 6. Results Description
We use PINN to numerically solve the incompressible Navier-Stokes equations. In PINN, randomly selected time and space coordinates are used as input values, corresponding velocity fields and pressure fields are used as output values, and initial values, boundary conditions are used as supervised constraints and the Navier-Stokes equation itself is used as unsupervised constraints added to the loss function for training. We designed three different fluid cases for three different types of PINN Navier-Stokes equations, namely NSFNet1, NSFNet2, and NSFNet3. Through the decrease of the loss function, the comparison of the network prediction results with high-fidelity DNS data, and the reduction of the $L_{2}$ error of the analytical solution, the convergence of the neural network in solving the Navier-Stokes equation can be proven, indicating that the NSFNets architecture possesses the ability to solve the incompressible Navier-Stokes equation. The experimental results show that the three forward problem cases using NSFNet can well approximate the reference solution, and we found that increasing the weights of boundary constraints and initial value constraints can enable the neural network to have better approximation effects.

## 7. References
[NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

[Github NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
