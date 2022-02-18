*Current version of PaddleScience is v0.1.*

# Introduction
PaddleScience extends the PaddlePaddle framework with reusable
software components for developing novel scientific computing applications. Such new
applications include Physics-informed Machine Learning, neural network based PDE solvers,
machine learning for CFD, and so on. PaddleScience is currently under active development.
Its design is evolving and its APIs are subject to change.  

# Core features and organization

PaddleScience currently focuses on the PINNs model. The core components are as follows.

- Geometry, a declarative interface for defining the geometric domain. Automatic
    discretization is supported 

- Neural net, currently supporting fully connected layers with customizable size and depth.

- PDE, delineating partial differential equations in symbolic forms. Specific PDEs derive the
    the base PDE class. Two native PDEs are currently included: Laplace2d and NavierStokes2d. 

- Loss, defining what exact penalties are enforced during the training process. By default,
    the L2 loss is applied. In the current design, the total loss is a weighted sum of
    three parts, the equation loss, the boundary condition loss and the initial condition loss.

- Optimizer, specifying which optimizer to use for training. Adam is the default option. More
    optimizers, such as BFGS, will be available in the future.

- Solver, managing the training process given the training data in a batchly fashion.

- Visualization, an easy access to the graph drawing utilities. 

The component organization is illustrated in the following figure. 

![image](./docs/source/img/pscicode.png)


# Getting started

## Prerequisites: 

Hardware requirements: NVIDIA GPU V100, NVIDIA GPU A100

Package dependencies: paddle, matplotlib, vtk. 

PaddleScience currently relies on new features of the Paddle framework so please be advised to download the latest version of Paddle on GitHub or on Gitee. 

For more details on installation, please refer to the offical [PaddlePaddle repository on GitHub](https://github.com/PaddlePaddle/Paddle) or [PaddlePaddle repository on Gitee](https://gitee.com/paddlepaddle/Paddle).

## Download and environment setup

```
Download from GitHub: git clone git@github.com:PaddlePaddle/PaddleScience.git
Download from Gitee:  git clone git@gitee.com:paddlepaddle/PaddleScience.git

cd PaddleScience
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Run examples

Some simple examples are baked in for quick demonstration. Please find them in the `examples` directory. To run an example, just enter the subdirectory and run the demo code in Python commandline. 

```
cd examples/laplace2d
python3.7 laplace2d.py

cd examples/darcy2d
python3.7 darcy2d.py

cd examples/ldc2d
python3.7 ldc2d.py
```

## Short tutorials on how to construct and solve a PINN model

[A tutorial on Lid Driven Cavity flow](./examples/ldc2d/README.md)

[A tutorial on Darcy flow in porous medium](./examples/darcy2d/README.md)
