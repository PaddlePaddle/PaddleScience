*Current version of PaddleScience is v1.0 Beta.*

# Introduction

PaddleScience extends the PaddlePaddle framework with reusable
software components for developing novel scientific computing applications. Such new
applications include Physics-informed Machine Learning, neural network based PDE solvers,
machine learning for CFD, and so on. PaddleScience is currently under active development.
Its design is evolving and its APIs are subject to change.

# Core features and organization

PaddleScience currently focuses on the PINNs model. The core components are as follows.

- PDE, delineating partial differential equations in symbolic forms. Specific PDEs derive the
    the base PDE class.

- Geometry, a declarative interface for defining the geometric domain. Automatic
    discretization is supported

- Neural net, currently supporting fully connected layers with customizable size and depth.

- Loss, defining what exact penalties are enforced during the training process. By default,
    the L2 loss is applied. In the current design, the total loss is a weighted sum of
    four parts, the equation loss, the boundary condition loss, the initial condition loss and the data loss.

- Optimizer, specifying which optimizer to use for training. Adam is the default option. More
    optimizers, such as BFGS, will be available in the future.

- Solver, managing the training process given the training data in a batchly fashion.

- Visualization, an easy access to the graph drawing utilities.

# Getting started

## Prerequisites

Hardware requirements: NVIDIA GPU V100, NVIDIA GPU A100

Package dependencies: paddle, cuda (11.0 or higher), numpy, scipy, sympy, matplotlib, vtk, pyevtk, wget.

PaddleScience currently relies on new features of the Paddle framework so please be advised to download the latest version of Paddle on GitHub or on Gitee.

For more details on installation, please refer to the offical [PaddlePaddle repository on GitHub](https://github.com/PaddlePaddle/Paddle) or [PaddlePaddle repository on Gitee](https://gitee.com/paddlepaddle/Paddle).

## Download and environment setup

``` shell
Download from GitHub: git clone https://github.com/PaddlePaddle/PaddleScience.git
Download from Gitee:  git clone https://gitee.com/PaddlePaddle/PaddleScience.git

cd PaddleScience
export PYTHONPATH=$PWD:$PYTHONPATH

pip install -r requirements.txt
```

## Run examples

Some examples are baked in for quick demonstration. Please find them in the `examples` directory. To run an example, just enter the subdirectory and run the demo code in Python commandline.

``` shell
# Darcy flow (Poisson equation)
cd examples/darcy
python darcy2d.py

# LDC steady (Steady Navier-Stokes eqution)
cd examples/ldc
python ldc2d_steady_train.py

# Lid Driven Cavity unsteady with continue-time method (Unsteady Navier-Stokes equation)
cd examples/ldc
python ldc2d_unsteady_t.py

# Flow around a circular cylinder with discrete-time method (Unsteady Navier-Stokes equation)
cd examples/cylinder/3d_unsteady_discrete/baseline
python cylinder3d_unsteady.py

```

## Short tutorials on how to construct and solve a PINN model

[A tutorial on Lid Driven Cavity flow](./examples/ldc/README.md)

[A tutorial on Darcy flow in porous medium](./examples/darcy/README.md)
