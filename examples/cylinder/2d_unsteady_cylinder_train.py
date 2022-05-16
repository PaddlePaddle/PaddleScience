# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created in May. 2022
@author: Hui Xiang, Yanbo Zhang, Shengze Cai
"""

import vtk
import copy
import numpy as np

import paddle
import paddlescience as psci
import paddle.distributed as dist
from pyevtk.hl import pointsToVTK

import loading_cfd_data as cfd
import module.cfd.pinn_solver as psolver


def train(net_params=None, distributed_env=False):
    if distributed_env:
        dist.init_parallel_env()

    #bc:10, eq:1, ic:10, suervise:10, nu=5e-3, t_start:0.6, t_end:0..9, t_step:20, init:0.5
    PINN = psolver.PysicsInformedNeuralNetwork(
        layers=6, nu=2e-2, bc_weight=10, eq_weight=1, ic_weight=10, supervised_data_weight=10, 
        outlet_weight=1, training_type='half-supervised', checkpoint_path='./checkpoint/', 
        net_params=net_params, distributed_env=distributed_env)

    # Loading data from openfoam 
    path = './data_0430/'
    dataloader = cfd.DataLoader(path=path, N_f=9000, N_b=1000, time_start=1, time_end=50, time_nsteps=50)
    training_time_list = dataloader.select_discretized_time(num_time=30)

    # Set initial data, | p, u, v, t, x, y
    initial_data = dataloader.loading_initial_data([1])
    PINN.set_initial_data(X=initial_data)

    # Set boundary data, | u, v, t, x, y
    boundary_data = dataloader.loading_boundary_data(training_time_list)
    PINN.set_boundary_data(X=boundary_data)

    # Set supervised data, | p, u, v, t, x, y
    supervised_data = dataloader.loading_supervised_data(training_time_list)
    PINN.set_supervised_data(X=supervised_data)
    # Set outlet data, | p, t, x, y
    outlet_data = dataloader.loading_outlet_data(training_time_list)
    PINN.set_outlet_data(X=outlet_data)

    # Set training data, | t, x, y
    training_data = dataloader.loading_train_inside_domain_data(training_time_list)
    PINN.set_eq_training_data(X=training_data)

    # Training
    adm_opt = paddle.optimizer.Adam(learning_rate=1e-5, parameters=PINN.net.parameters())
    PINN.train(num_epoch=100000, optimizer=adm_opt)

    #bfgs_opt = psci.optimizer.BFGS()


if __name__ == "__main__":
    # Loss | eq 0.0071044, bc 0.0003727, ic 0.06930, data 0.001471
    net_params = './checkpoint/pretrained_net_params'
    train(net_params=net_params)
    # distributed_training
    #train(net_params=net_params, distributed_env=True)




