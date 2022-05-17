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


def train(net_params=None):

    Neta = 100
    N_train = 100
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
        net_params=net_params)
    PINN.set_eta_data(X=(t_eta, eta))
    PINN.set_f_data(X=(t_f, f))

    # Training
    batchsize = 150
    scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=1e-3, step_size=20000, gamma=0.9)
    adm_opt = paddle.optimizer.Adam(
        scheduler, weight_decay=None, parameters=PINN.net.parameters())
    PINN.train(
        num_epoch=100000,
        batchsize=batchsize,
        optimizer=adm_opt,
        scheduler=scheduler)
    adm_opt = psci.optimizer.Adam(
        learning_rate=1e-5,
        weight_decay=None,
        parameters=PINN.net.parameters())
    PINN.train(num_epoch=100000, batchsize=batchsize, optimizer=adm_opt)


if __name__ == "__main__":
    net_params = None
    train(net_params=net_params)
