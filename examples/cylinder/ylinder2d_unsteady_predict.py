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
import time
import numpy as np

import paddle
import paddlescience as psci
import paddle.distributed as dist
from pyevtk.hl import pointsToVTK

import loading_cfd_data as cfd
from module.cfd import pinn_solver as psolver


def predict(net_params=None, vtk_filename = './vtk/uvp_t_'):
    # Loading checkpoint state_dict params
    net_params = net_params

    PINN = psolver.PysicsInformedNeuralNetwork(layers=6, net_params=net_params)

    # Loading data from openfoam
    path = './data_0430/'
    dataloader = cfd.DataLoader(path=path, N_f=100000, time_start=1, time_end=50, time_nsteps=50)
    #training_time_list = dataloader.select_discretized_time(num_time=30)
    training_time_list = dataloader.select_ordered_time(num_time=50)

    # Set outlet data, | p, t, x, y
    outlet_data = dataloader.loading_outlet_data(training_time_list)
    PINN.set_outlet_data(X=outlet_data)

    # Set boundary data, | u, v, t, x, y
    boundary_data = dataloader.loading_boundary_data(training_time_list)
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | t, x, y
    training_data = dataloader.loading_train_inside_domain_data(training_time_list)
    PINN.set_eq_training_data(X=training_data)

    #outlet
    outlet_p, outlet_t, outlet_x, outlet_y = outlet_data
    num_outlet_data = int(outlet_t.shape[0]/len(training_time_list))

    #boundary
    boundary_u, boundary_v, boundary_t, boundary_x, boundary_y = boundary_data
    num_boundary_data = int(boundary_t.shape[0]/len(training_time_list))

    #training
    training_t, training_x, training_y = training_data
    num_train_data = int(training_t.shape[0]/len(training_time_list))
    #start_time = time.perf_counter()
    for i in range(len(training_time_list)):
        t_step = training_time_list[i]

        # training
        t_x = training_x[(i*num_train_data):(i+1)*num_train_data, :].astype('float32')
        t_y = training_y[(i*num_train_data):(i+1)*num_train_data, :].astype('float32')
        t_t = training_t[(i*num_train_data):(i+1)*num_train_data, :].astype('float32')

        # outlet 
        o_x = outlet_x[(i*num_outlet_data):(i+1)*num_outlet_data, :].astype('float32')
        o_y = outlet_y[(i*num_outlet_data):(i+1)*num_outlet_data, :].astype('float32')
        o_t = outlet_t[(i*num_outlet_data):(i+1)*num_outlet_data, :].astype('float32')

        # boundary 
        b_x = boundary_x[(i*num_boundary_data):(i+1)*num_boundary_data, :].astype('float32')
        b_y = boundary_y[(i*num_boundary_data):(i+1)*num_boundary_data, :].astype('float32')
        b_t = boundary_t[(i*num_boundary_data):(i+1)*num_boundary_data, :].astype('float32')

        all_x = np.concatenate((t_x, o_x, b_x ))
        all_y = np.concatenate((t_y, o_y, b_y))
        all_t = np.concatenate((t_t, o_t, b_t))

        tensor_t = paddle.to_tensor(all_t, dtype='float32')
        tensor_x = paddle.to_tensor(all_x, dtype='float32')
        tensor_y = paddle.to_tensor(all_y, dtype='float32')

        #start_time = time.perf_counter()
        u, v, p = PINN.predict(net_params, (tensor_t,tensor_x,tensor_y))
        #stop_time = time.perf_counter()
        #print('Spend %.3f seconds to predict 1 time steps.'%(stop_time - start_time))
        u = u.numpy()
        v = v.numpy()
        p = p.numpy()

        axis_z = np.zeros(num_train_data+num_outlet_data+num_boundary_data, dtype="float32")
        filename = vtk_filename + str(t_step)
        pointsToVTK(filename, all_x.flatten().copy(), all_y.flatten().copy(), axis_z.flatten().copy(),
                    data={"u": u.copy(), "v":v.copy(), "p": p.copy()})        

def predict_once_for_all(net_params=None, vtk_filename = './vtk/uvp_t_'):
    # Loading checkpoint state_dict params
    net_params = net_params

    PINN = psolver.PysicsInformedNeuralNetwork(layers=6, net_params=net_params)

    # Loading data from openfoam
    path = './data_0430/'
    dataloader = cfd.DataLoader(path=path, N_f=100000, time_start=1, time_end=50, time_nsteps=50)
    #training_time_list = dataloader.select_discretized_time(num_time=30)
    training_time_list = dataloader.select_ordered_time(num_time=50)

    # Set outlet data, | p, t, x, y
    outlet_data = dataloader.loading_outlet_data(training_time_list)
    PINN.set_outlet_data(X=outlet_data)

    # Set boundary data, | u, v, t, x, y
    boundary_data = dataloader.loading_boundary_data(training_time_list)
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | t, x, y
    training_data = dataloader.loading_train_inside_domain_data(training_time_list)
    PINN.set_eq_training_data(X=training_data)

    #outlet
    outlet_p, outlet_t, outlet_x, outlet_y = outlet_data
    num_outlet_data = int(outlet_t.shape[0]/len(training_time_list))

    #boundary
    boundary_u, boundary_v, boundary_t, boundary_x, boundary_y = boundary_data
    num_boundary_data = int(boundary_t.shape[0]/len(training_time_list))

    #training
    training_t, training_x, training_y = training_data
    num_train_data = int(training_t.shape[0]/len(training_time_list))
    #start_time = time.perf_counter()
        # training
    t_x = training_x[:, :].astype('float32')
    t_y = training_y[:, :].astype('float32')
    t_t = training_t[:, :].astype('float32')

    # outlet 
    o_x = outlet_x[:, :].astype('float32')
    o_y = outlet_y[:, :].astype('float32')
    o_t = outlet_t[:, :].astype('float32')

    # boundary 
    b_x = boundary_x[:, :].astype('float32')
    b_y = boundary_y[:, :].astype('float32')
    b_t = boundary_t[:, :].astype('float32')

    all_x = np.concatenate((t_x, o_x, b_x ))
    all_y = np.concatenate((t_y, o_y, b_y))
    all_t = np.concatenate((t_t, o_t, b_t))

    tensor_t = paddle.to_tensor(all_t, dtype='float32')
    tensor_x = paddle.to_tensor(all_x, dtype='float32')
    tensor_y = paddle.to_tensor(all_y, dtype='float32')

    # start_time = time.perf_counter()
    u, v, p = PINN.predict(net_params, (tensor_t,tensor_x,tensor_y))
    # stop_time = time.perf_counter()
    # print('Spend %.3f seconds to predict 50 time steps.'%(stop_time - start_time))
    u = u.numpy()
    v = v.numpy()
    p = p.numpy()

    total_len = num_train_data+num_outlet_data+num_boundary_data
    for i in range(len(training_time_list)):
        t_step = training_time_list[i]
        axis_z = np.zeros(num_train_data+num_outlet_data+num_boundary_data, dtype="float32")
        filename = vtk_filename + str(t_step)
        pointsToVTK(filename, all_x[i*total_len:(i+1)*total_len,:].flatten().copy(), 
                    all_y[i*total_len:(i+1)*total_len,:].flatten().copy(), axis_z.flatten().copy(),
                    data={"u": u[i*total_len:(i+1)*total_len].copy(), 
                          "v": v[i*total_len:(i+1)*total_len].copy(), 
                          "p": p[i*total_len:(i+1)*total_len].copy()})        

if __name__ == "__main__":
    net_params = './checkpoint/pretrained_net_params'
    vtk_filename = './vtk/uvp_t_'
    predict_once_for_all(net_params=net_params, vtk_filename=vtk_filename)



