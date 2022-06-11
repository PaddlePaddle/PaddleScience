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
'''
Created in May. 2022
PINNs for Inverse VIV Problem
@Author: Xuhui Meng, Zhicheng Wang, Hui Xiang, Yanbo Zhang
'''

import copy
import time
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import paddle
import paddle.nn as nn
import paddlescience as psci
import paddle.distributed as dist

from dataset import Dataset

import paddlescience.module.fsi.viv_pinn_solver as psolver

np.random.seed(1234)


def predict(net_params=None):

    Neta = 100
    N_train = 150
    t_range = [0.0625, 10]

    data = Dataset(t_range, Neta, N_train)

    #inputdata
    t_eta, eta, t_f, f, tmin, tmax = data.build_data()

    PINN = psolver.PysicsInformedNeuralNetwork(
        layers=6,
        hidden_size=30,
        num_ins=1,
        num_outs=1,
        t_max=tmax,
        t_min=tmin,
        N_f=f.shape[0],
        checkpoint_path='./checkpoint/',
        net_params=net_params,
        mode='predict')

    PINN.set_eta_data(X=(t_eta, eta))
    PINN.set_f_data(X=(t_f, f))
    eta_pred, f_pred = PINN.predict((-4.0, 0.0))

    error_f = np.linalg.norm(
        f.reshape([-1]) - f_pred.numpy().reshape([-1]), 2) / np.linalg.norm(f,
                                                                            2)
    error_eta = np.linalg.norm(
        eta.reshape([-1]) - eta_pred.numpy().reshape([-1]),
        2) / np.linalg.norm(eta, 2)
    print('------------------------')
    print('Error f: %e' % (error_f))
    print('Error eta: %e' % (error_eta))
    print('------------------------')


if __name__ == "__main__":
    net_params = './checkpoint/net_params_100000'
    predict(net_params=net_params)
