Demo models
===========

Demo codes in the subdirectories show how to build and solve a PINN model
using currently offerred APIs in this package.


Laplace2d
---------

Solving a steady state 2-dimensional Laplacian equation in a rectangular domain.


Darcy2d
-------

Solving a Darcy flow problem in a 2d setting.


Ldc2d
-----

Solving a Lid Driven Cavity problem in rectangular domain.


2D unsteady circular cylinder
-----------------------------
Solving an unsteady 2-D NS equation in a rectangular domain with cylinder

Preprocessing: set data path and filenames in loading_cfd_data.py

Training:  python examples/cylinder/2d_unsteady_cylinder_train.py

Predict:  python examples/cylinder/2d_unsteady_cylinder_predict.py

# https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/launch.html
Distributed Scripts:  python -m paddle.distributed.launch --gpus="xxx" examples/cylinder/2d_unsteady_cylinder_train.py

Pretrained checkpoint:  ./cylinder/checkpoint/pretrained_net_params
