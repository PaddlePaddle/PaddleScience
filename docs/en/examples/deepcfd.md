# DeepCFD (Deep Computational Fluid Dynamics)

<a href="https://aistudio.baidu.com/projectdetail/8691947" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
    wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
    # windows
    # curl --create-dirs -o ./datasets/dataX.pkl https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
    # curl --create-dirs -o ./datasets/dataX.pkl https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
    python deepcfd.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
    wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
    # windows
    # curl --create-dirs -o ./datasets/dataX.pkl https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
    # curl --create-dirs -o ./datasets/dataX.pkl https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
    python deepcfd.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/deepcfd/deepcfd_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python deepcfd.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
    wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
    # windows
    # curl --create-dirs -o ./datasets/dataX.pkl https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
    # curl --create-dirs -o ./datasets/dataX.pkl https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
    python deepcfd.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [deepcfd_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/deepcfd/deepcfd_pretrained.pdparams) | MSE.Total_MSE(mse_validator): 1.92947<br>MSE.Ux_MSE(mse_validator): 0.70684<br>MSE.Uy_MSE(mse_validator): 0.21337<br>MSE.p_MSE(mse_validator): 1.00926 |

## 1. Background Introduction

Computational fluid dynamics (CFD) simulation can obtain the distribution of various physical quantities of fluids, such as density, pressure and velocity, by solving the Navier-Stokes equations (N-S equations). It is widely used in fields such as microelectromechanical systems, civil engineering and aerospace.

In some complex application scenarios, such as wing optimization and fluid-structure interaction, tens of millions or even hundreds of millions of grids are needed to model the problem (as shown in the figure below, the figure shows the full-machine internal and external flow integrated structured grid model of the F-18 fighter), resulting in huge computational costs for CFD. Therefore, there is an urgent need to develop a method that is more efficient than traditional CFD methods while maintaining computational accuracy.

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/DeepCFD/f-18.jpg){ loading=lazy}
  <figcaption>Full-machine internal and external flow integrated structured grid model of F-18 fighter</figcaption>
</figure>

## 2. Problem Definition

The Navier-Stokes equations are equations used to describe fluid motion. Their two-dimensional form is as follows,

Mass conservation:

$$\nabla \cdot    \bf{u}=0$$

Momentum conservation:

$$\rho(\frac{\partial}{\partial t}  + \bf{u} \cdot  div ) \bf{u} = - \nabla p +  - \nabla \tau + \bf{f}$$

Where $\bf{u}$ is the velocity field (with x and y dimensions), $\rho$ is the density, $p$ is the pressure field, and $\bf{f}$ is the body force (such as gravity).

Assuming non-uniform steady-state fluid conditions are met, the time-dependent term can be removed from the equation, and $\bf{u}$ can be decomposed into velocity components $u_x$ and $u_y$. The momentum equation can be rewritten as:

$$u_x\frac{\partial u_x}{\partial x} + u_y\frac{\partial u_x}{\partial y} = - \frac{1}{\rho}\frac{\partial p}{\partial x} + \nu \nabla^2 u_x + g_x$$

$$u_x\frac{\partial u_y}{\partial x} + u_y\frac{\partial u_y}{\partial y} = - \frac{1}{\rho}\frac{\partial p}{\partial y} + \nu \nabla^2 u_y + g_y$$

Where $g$ represents gravitational acceleration and $\nu$ represents the kinematic viscosity of the fluid.

## 3. Problem Solving

The above problem can usually be solved using OpenFOAM for traditional numerical methods, but the calculation amount is large. Next, we will explain how to solve this problem using deep learning methods based on PaddleScience code.

This case is solved based on the method of the paper [Ribeiro M D, Rehman A, Ahmed S, et al. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks](https://arxiv.org/abs/2004.08826). For the theoretical part of this method, please refer to the [original paper](https://arxiv.org/abs/2004.08826).
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The data in this dataset is obtained using OpenFOAM. The dataset has two files, dataX and dataY. dataX contains the input information of the geometric shapes of 981 channel flow samples, and dataY contains the corresponding OpenFOAM solution results.

Before running the code for this problem, please download [dataX](https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl) and [dataY](https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl) according to the command below:

``` sh
wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
```

dataX and dataY both have the same dimensions (Ns, Nc, Nx, Ny), where the first axis is the number of samples (Ns), the second axis is the number of channels (Nc), and the third and fourth axes are the number of elements in x and y (Nx and Ny) respectively. In the input data dataX, the first channel is the SDF (Signed distance function) of the obstacle in the computational domain, the second channel is the label of the flow region, and the third channel is the SDF of the computational domain boundary. In the output data dataY, the first channel is the horizontal velocity component (Ux), the second channel is the vertical velocity component (Uy), and the third channel is the fluid pressure (p).

The original download address of the dataset is: <https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1>

We divide the dataset into training set and validation set at a ratio of 7:3, the code is as follows:

``` py linenums="202" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:202:216
--8<--
```

### 3.2 Model Construction

In the above problem, we determine that the input is input and the output is output. According to the paper, we use the UNetEx network containing 3 encoders and decoders to create the model.

The input of the model contains the SDF (Signed distance function) of the obstacle, the label of the flow region and the SDF of the computational domain boundary. The output of the model contains the horizontal velocity component (Ux), the vertical velocity component (Uy) and the fluid pressure (p).

<figure markdown>
  ![DeepCFD](https://ai-studio-static-online.cdn.bcebos.com/150bd6248d5f4c0bb6186e3498e87b57fcc5f67ffa5148fd9b139edb61d370a6){ loading=lazy}
  <figcaption>DeepCFD network structure</figcaption>
</figure>

Model creation is expressed in PaddleScience code as follows:

``` py linenums="218" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:218:219
--8<--
```

### 3.3 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints, the code is as follows:

``` py linenums="234" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:234:264
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here fill in the variable names of relevant data.

The second parameter is the definition of the loss function. Here, a custom loss function is used to calculate the mean square error of Ux and Uy, and the standard deviation of p, and then the weighted sum of the three.

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named "sup_constraint".

After the supervised constraint is constructed, encapsulate it into a dictionary with the name we just named as the key for subsequent access.

``` py linenums="266" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:266:267
--8<--
```

### 3.4 Hyperparameter Setting

Next, you need to specify the number of training epochs in the configuration file. Here, based on experimental experience, we use one thousand training epochs.

``` yaml linenums="47" title="examples/deepcfd/conf/deepcfd.yaml"
--8<--
examples/deepcfd/conf/deepcfd.yaml:47:52
--8<--
```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the more commonly used `Adam` optimizer is selected, the learning rate is set to 0.001, and the weight decay is set to 0.005.

``` py linenums="269" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:269:272
--8<--
```

### 3.6 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set at a certain epoch interval. We use `ppsci.validate.SupervisedValidator` to construct the validator.

``` py linenums="274" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:274:314
--8<--
```

Evaluation metric `metric` here defines four indicators Total_MSE, Ux_MSE, Uy_MSE and p_MSE.

Other configurations are similar to the settings of [Constraint Construction](#33-constraint-construction).

### 3.7 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="316" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:316:335
--8<--
```

### 3.8 Result Visualization

Use matplotlib to plot the calculation results of OpenFOAM and DeepCFD with the same input parameters for comparison. Here, the calculation results of the 0th data of the validation set are plotted.

``` py linenums="337" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py:337:342
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/deepcfd/deepcfd.py"
--8<--
examples/deepcfd/deepcfd.py
--8<--
```

## 5. Result Display

<figure markdown>
  ![DeepCFD](https://ai-studio-static-online.cdn.bcebos.com/288c37b569d5400aa7b2265ff13fcf0edad3115e70fe4fafb6736215355771fe){ loading=lazy}
  <figcaption>Comparison of OpenFOAM calculation results and DeepCFD prediction results, from top to bottom: horizontal velocity component (Ux), vertical velocity component (Uy) and fluid pressure (p)</figcaption>
</figure>

It can be seen that the DeepCFD method is basically consistent with the OpenFOAM results.

## 6. References

* [Ribeiro M D, Rehman A, Ahmed S, et al. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks](https://arxiv.org/abs/2004.08826)
* [DeepCFD Reproduction based on PaddlePaddle](https://aistudio.baidu.com/projectdetail/4400677)
