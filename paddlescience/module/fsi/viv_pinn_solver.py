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
# -*- coding: utf-8 -*-
'''
Created in May. 2022
PINNs for Inverse VIV Problem
@Author: Xuhui Meng, Zhicheng Wang, Hui Xiang, Yanbo Zhang
'''
import time
import paddle
import numpy as np
import paddle.nn as nn
import paddlescience as psci
import paddle.distributed as dist

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

__all__ = ["PINN_Solver"]


class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 layers=4,
                 hidden_size=40,
                 num_ins=2,
                 num_outs=3,
                 learning_rate=0.001,
                 opt=None,
                 net_params=None,
                 N_f=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/',
                 t_min=None,
                 t_max=None,
                 distributed_env=False,
                 mode='train'):

        self.mode = mode
        self.t_min = t_min
        self.t_max = t_max

        self.eta_weight = 100

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path
        self.distributed_env = distributed_env

        self.N_f = N_f if N_f is not None else 0

        # initialize NN_2, u,t -> eta
        self.net = self.initialize_NN(
            num_ins=num_ins,
            num_outs=num_outs,
            num_layers=layers,
            hidden_size=hidden_size)

        if mode == 'train':
            self.k1 = paddle.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=-4.0))
            self.k2 = paddle.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=0.0))
            self.net.add_parameter('k1', self.k1)
            self.net.add_parameter('k2', self.k2)

        if net_params:
            load_params = paddle.load(net_params)
            self.net.set_state_dict(load_params)

        self.opt = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            parameters=self.net.parameters()) if not opt else opt

        if self.distributed_env:
            self.parallel_net = paddle.DataParallel(self.net)

    def set_eta_data(self, X=None, continuous_time=True):
        stop_gradient = True
        self.t_eta = paddle.to_tensor(
            X[0], dtype='float32', stop_gradient=False)
        self.eta = paddle.to_tensor(X[1], dtype='float32')

    def set_f_data(self, X=None, continuous_time=True):
        stop_gradient = False
        self.t_f = paddle.to_tensor(X[0], dtype='float32', stop_gradient=False)
        self.f = paddle.to_tensor(X[1], dtype='float32', stop_gradient=False)

    def set_optimizers(self, opt):
        self.opt = opt

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        return psci.network.FCNet(
            num_ins=num_ins,
            num_outs=num_outs,
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation='tanh')

    def neural_net_eta(self, t, u=None):
        eta = self.net.nn_func(t)
        return eta

    def neural_net_equations(self, t, u=None):
        eta = self.net.nn_func(t)
        eta_t = self.autograd(eta, t)
        eta_tt = self.autograd(eta_t, t, create_graph=False)

        rho = 2.0
        k1_ = paddle.exp(self.k1)
        k2_ = paddle.exp(self.k2)
        f = rho * eta_tt + k1_ * eta_t + k2_ * eta
        return eta, f

    def autograd(self, U, x, create_graph=True):
        return paddle.autograd.grad(
            U, x, retain_graph=True, create_graph=create_graph)[0]

    def fwd_computing_loss_2d(self, idx=None):
        self.eta_pred, self.f_pred = self.neural_net_equations(self.t_f)
        self.eta_loss = paddle.mean((self.eta - self.eta_pred)**2)
        self.eq_loss = paddle.mean((self.f - self.f_pred)**2)

        self.loss = (self.eta_weight * self.eta_loss + self.eq_loss)
        losses = [self.eta_loss, self.eq_loss]
        return self.loss, losses

    def predict(self, X=None):
        self.k1, self.k2 = X
        self.k1 = paddle.to_tensor(self.k1, dtype='float32')
        self.k2 = paddle.to_tensor(self.k2, dtype='float32')
        eta_pred = self.neural_net_eta(self.t_eta)
        _, f_pred = self.neural_net_equations(self.t_f)
        return eta_pred, f_pred

    def train(self,
              num_epoch=1,
              optimizer=None,
              batchsize=None,
              scheduler=None):
        self.opt = optimizer
        self.scheduler = scheduler
        if isinstance(self.opt, paddle.optimizer.AdamW) or isinstance(
                self.opt, paddle.optimizer.Adam):
            return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch,
                                   batchsize)
        elif self.opt is paddle.incubate.optimizer.functional.minimize_bfgs:
            return self.solve_bfgs(self.fwd_computing_loss_2d, num_epoch,
                                   batchsize)

    def solve_Adam(self, loss_func, num_epoch=1000, batchsize=50000):
        loss = 0
        for epoch_id in range(num_epoch):
            self.epoch = epoch_id
            loss, losses = loss_func()
            self.opt.clear_grad()
            loss.backward()
            self.opt.step()
            if self.scheduler:
                self.scheduler.step()

            print("current lr is {}".format(self.opt.get_lr()))
            k1__ = np.exp(float(self.k1))
            k2__ = np.exp(float(self.k2))
            print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
                  "loss[Adam]: ",
                  float(loss), "k1:", k1__, "k2:", k2__, "eta loss:",
                  float(losses[0]), "eq_loss:", float(losses[1]))
            if (epoch_id + 1) % self.checkpoint_freq == 0:
                paddle.save(
                    self.net.state_dict(),
                    self.checkpoint_path + 'net_params_' + str(epoch_id + 1))

    def solve_bfgs(self, loss_func, num_epoch=1000, batchsize=None):
        batch_id = 0
        step = 0
        loss = None
        losses = []

        def _f(x):
            nonlocal batch_id, loss, losses, step
            self.net.reconstruct(x)
            loss, losses = loss_func()
            return loss

        x0 = self.net.flatten_params()

        for epoch_id in range(num_epoch):
            results = self.opt(_f,
                               x0,
                               initial_inverse_hessian_estimate=None,
                               line_search_fn='strong_wolfe',
                               dtype='float32')
            x0 = results[2]
            print("Step: {step:>6} [LS] loss[BFGS]: ",
                  float(results[3]), "total loss:",
                  float(loss), "eq_loss: ",
                  float(losses[0]), "bc_loss: ",
                  float(losses[1]), "ic_loss: ", float(losses[2]))
            step += 1

        print("======Optimization results======")
        print(x0)
