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
    optimizers, such as BFGS, will be added in the future.

- Solver, managing the training process given the training data in a batchly fashion.

- Visualization, an easy access to the graph drawing utilities. 

The component organization is illustrated in the following figure. 

![image](https://user-images.githubusercontent.com/3903722/142380670-32d49736-aa4a-4e42-ae66-22a8320a235d.png)


# Getting started

## Prerequisites: 

Package dependencies: paddle, matplotlib, vtk. 

To install the PaddlePaddle framework, please refer to the offical [PaddlePaddle repository](https://github.com/PaddlePaddle/Paddle).

## Download and setup environment

```
git clone git@github.com:PaddlePaddle/PaddleScience.git
cd PaddleScience
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Run examples

To find simple examples, please look into the `examples` directory. Feel free to
try out examples and welcome to give us feedbacks. 

```
cd examples/laplace2d
python3.7 laplace2d.py

cd examples/darcy2d
python3.7 darcy2d.py

cd examples/ldc2d
python3.7 ldc2d.py
```
