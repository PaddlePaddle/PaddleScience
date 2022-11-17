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
"""
Created in May. 2022
@author: Hui Xiang, Yanbo Zhang, Shengze Cai
"""

import time
import paddle
import numpy as np
import paddlescience as psci
import paddle.distributed as dist

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

__all__ = ["PINN_Solver"]


class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 layers=10,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 supervised_data_weight=1,
                 opt=None,
                 training_type='unsupervised',
                 nu=1.e-4,
                 net_params=None,
                 distributed_env=False,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path
        self.distributed_env = distributed_env

        self.nu = nu
        self.training_type = training_type
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0

        # initialize NN
        self.net = self.initialize_NN(num_layers=layers)
        if net_params:
            load_params = paddle.load(net_params)
            self.net.set_state_dict(load_params)

        self.opt = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            weight_decay=0.005,
            parameters=self.net.parameters()) if not opt else opt

        if self.distributed_env:
            self.parallel_net = paddle.DataParallel(self.net)

    def set_outlet_data(self, X=None, continuous_time=True):
        # p, t, x, y
        stop_gradient = True
        if continuous_time:
            self.p_o = paddle.to_tensor(
                X[0], dtype='float32', stop_gradient=stop_gradient)
            self.t_o = paddle.to_tensor(
                X[1], dtype='float32', stop_gradient=stop_gradient)
            self.x_o = paddle.to_tensor(
                X[2], dtype='float32', stop_gradient=stop_gradient)
            self.y_o = paddle.to_tensor(
                X[3], dtype='float32', stop_gradient=stop_gradient)

    def set_initial_data(self, X=None, continuous_time=True):
        # initial training data | u, v, x, y
        stop_gradient = True
        self.p_i = paddle.to_tensor(
            X[0], dtype='float32', stop_gradient=stop_gradient)
        self.u_i = paddle.to_tensor(
            X[1], dtype='float32', stop_gradient=stop_gradient)
        self.v_i = paddle.to_tensor(
            X[2], dtype='float32', stop_gradient=stop_gradient)
        self.t_i = paddle.to_tensor(
            X[3], dtype='float32', stop_gradient=stop_gradient)
        self.x_i = paddle.to_tensor(
            X[4], dtype='float32', stop_gradient=stop_gradient)
        self.y_i = paddle.to_tensor(
            X[5], dtype='float32', stop_gradient=stop_gradient)

    def set_boundary_data(self, X=None, continuous_time=True):
        # boundary training data | u, v, t, x, y
        stop_gradient = True
        if continuous_time:
            self.u_b = paddle.to_tensor(
                X[0], dtype='float32', stop_gradient=stop_gradient)
            self.v_b = paddle.to_tensor(
                X[1], dtype='float32', stop_gradient=stop_gradient)
            self.t_b = paddle.to_tensor(
                X[2], dtype='float32', stop_gradient=stop_gradient)
            self.x_b = paddle.to_tensor(
                X[3], dtype='float32', stop_gradient=stop_gradient)
            self.y_b = paddle.to_tensor(
                X[4], dtype='float32', stop_gradient=stop_gradient)

    def set_boundary_conditions(self, X=None, condition='dirichlet'):
        if condition == 'dirichlet':
            pass
        elif condition == 'neumann':
            pass

    def set_supervised_data(self, X=None, continuous_time=True):
        # Training data: BC data || Equation data || Supervised data || IC data
        # p, u, v, t, x, y
        #X = paddle.to_tensor(X, dtype='float32', stop_gradient=False)
        stop_gradient = True
        if continuous_time:
            self.p_s = paddle.to_tensor(
                X[0], dtype='float32', stop_gradient=stop_gradient)
            self.u_s = paddle.to_tensor(
                X[1], dtype='float32', stop_gradient=stop_gradient)
            self.v_s = paddle.to_tensor(
                X[2], dtype='float32', stop_gradient=stop_gradient)
            self.t_s = paddle.to_tensor(
                X[3], dtype='float32', stop_gradient=stop_gradient)
            self.x_s = paddle.to_tensor(
                X[4], dtype='float32', stop_gradient=stop_gradient)
            self.y_s = paddle.to_tensor(
                X[5], dtype='float32', stop_gradient=stop_gradient)

    def set_eq_training_data(self,
                             X=None,
                             continuous_time=True,
                             data_source='half-supervised'):
        # Training data: BC data || Equation data || Supervised data || IC data
        if data_source == 'random':
            pass

        elif data_source == 'orthogonal':
            pass

        elif data_source == 'half-supervised':
            if continuous_time:
                self.t_f = paddle.to_tensor(
                    X[0], dtype='float32', stop_gradient=False)
                self.x_f = paddle.to_tensor(
                    X[1], dtype='float32', stop_gradient=False)
                self.y_f = paddle.to_tensor(
                    X[2], dtype='float32', stop_gradient=False)

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

    def neural_net_u(self, t, x, y):
        X = paddle.concat([t, x, y], axis=1)
        uvp = self.net.nn_func(X)
        u = uvp[:, 0]
        v = uvp[:, 1]
        p = uvp[:, 2]
        return u, v, p

    def neural_net_equations(self, t, x, y):
        X = paddle.concat([t, x, y], axis=1)
        uvp = self.net.nn_func(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        u_t = self.autograd(u, t)
        u_x = self.autograd(u, x)
        u_y = self.autograd(u, y)
        u_xx = self.autograd(u_x, x, create_graph=False)
        u_yy = self.autograd(u_y, y, create_graph=False)

        v_t = self.autograd(
            v,
            t, )
        v_x = self.autograd(
            v,
            x, )
        v_y = self.autograd(
            v,
            y, )
        v_xx = self.autograd(v_x, x, create_graph=False)
        v_yy = self.autograd(v_y, y, create_graph=False)

        p_x = self.autograd(p, x)
        p_y = self.autograd(p, y)

        # NS 
        eq1 = (u * u_x + v * u_y) + p_x - (self.nu) * (u_xx + u_yy) + u_t
        eq2 = (u * v_x + v * v_y) + p_y - (self.nu) * (v_xx + v_yy) + v_t
        # Continuty
        eq3 = u_x + v_y

        residual = (eq1 * u + eq2 * v)
        #return eq1, eq2, eq3, residual
        return eq1, eq2, eq3

    def autograd(self, U, x, create_graph=True):
        return paddle.autograd.grad(
            U, x, retain_graph=True, create_graph=create_graph)[0]

    def set_training_loss(self, loss):
        self.psci_loss = loss

    def predict(self, net_params, X):
        t, x, y = X
        return self.neural_net_u(t, x, y)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.numpy()
        shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return paddle.to_tensor(
            tensor_to_numpy, dtype='float32', stop_gradient=False)

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # physics informed neural networks (inside the domain)
        # initial data
        (self.u_pred_i, self.v_pred_i,
         self.p_pred_i) = self.neural_net_u(self.t_i, self.x_i, self.y_i)

        # IC loss
        if loss_mode == 'L2':
            self.loss_i = paddle.norm((self.u_i.reshape([-1]) - self.u_pred_i.reshape([-1])), p=2) + \
                          paddle.norm((self.v_i.reshape([-1]) - self.v_pred_i.reshape([-1])), p=2) + \
                          paddle.norm((self.p_i.reshape([-1]) - self.p_pred_i.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_i = paddle.mean(paddle.square(self.u_i.reshape([-1]) - self.u_pred_i.reshape([-1]))) + \
                          paddle.mean(paddle.square(self.v_i.reshape([-1]) - self.v_pred_i.reshape([-1]))) + \
                          paddle.mean(paddle.square(self.p_i.reshape([-1]) - self.p_pred_i.reshape([-1])))

# boundary data
        (self.u_pred_b, self.v_pred_b,
         self.p_pred_b) = self.neural_net_u(self.t_b, self.x_b, self.y_b)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b = paddle.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          paddle.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = paddle.mean(paddle.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          paddle.mean(paddle.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

        # outlet data
        (self.u_pred_o, self.v_pred_o,
         self.p_pred_o) = self.neural_net_u(self.t_o, self.x_o, self.y_o)

        # outlet loss
        if loss_mode == 'L2':
            self.loss_o = paddle.norm(
                (self.p_o.reshape([-1]) - self.p_pred_o.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_o = paddle.mean(
                paddle.square(
                    self.p_o.reshape([-1]) - self.p_pred_o.reshape([-1])))

        # supervised interior data
        if self.training_type == 'half-supervised':
            (self.u_pred_s, self.v_pred_s,
             self.p_pred_s) = self.neural_net_u(self.t_s, self.x_s, self.y_s)
            # supervised data loss
            if loss_mode == 'L2':
                self.loss_s = paddle.norm((self.u_s.reshape([-1]) - self.u_pred_s.reshape([-1])), p=2) + \
                              paddle.norm((self.v_s.reshape([-1]) - self.v_pred_s.reshape([-1])), p=2)
            if loss_mode == 'MSE':
                self.loss_s = paddle.mean(paddle.square(self.u_s.reshape([-1]) - self.u_pred_s.reshape([-1]))) + \
                              paddle.mean(paddle.square(self.v_s.reshape([-1]) - self.v_pred_s.reshape([-1])))

        # equation        
        if self.training_type == 'unsupervised' or self.training_type == 'half-supervised':
            (self.eq1_pred, self.eq2_pred,
             self.eq3_pred) = self.neural_net_equations(self.t_f, self.x_f,
                                                        self.y_f)
            # equation residual loss
            if loss_mode == 'L2':
                self.loss_e = paddle.norm(self.eq1_pred.reshape([-1]), p=2) + \
                              paddle.norm(self.eq2_pred.reshape([-1]), p=2) + \
                              paddle.norm(self.eq3_pred.reshape([-1]), p=2)
            if loss_mode == 'MSE':
                self.loss_e = paddle.mean(paddle.square(self.eq1_pred.reshape([-1]))) + \
                              paddle.mean(paddle.square(self.eq2_pred.reshape([-1]))) + \
                              paddle.mean(paddle.square(self.eq3_pred.reshape([-1])))

        self.loss = self.alpha_b * self.loss_b + \
                    self.alpha_s * self.loss_s + \
                    self.alpha_e * self.loss_e + \
                    self.alpha_i * self.loss_i + \
                    self.alpha_o * self.loss_o

        return self.loss, [
            self.loss_e, self.loss_b, self.loss_s, self.loss_o, self.loss_i
        ]

    def train(self,
              num_epoch=1,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        self.opt = optimizer
        if isinstance(self.opt, paddle.optimizer.AdamW) or isinstance(
                self.opt, paddle.optimizer.Adam):
            return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch,
                                   batchsize, scheduler)
        elif self.opt is paddle.incubate.optimizer.functional.minimize_bfgs:
            return self.solve_bfgs(self.fwd_computing_loss_2d, num_epoch,
                                   batchsize)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        if self.distributed_env:
            for epoch_id in range(num_epoch):
                with self.parallel_net.no_sync():
                    loss, losses = loss_func()
                    loss.backward()
                fused_allreduce_gradients(list(self.net.parameters()), None)
                self.opt.step()
                self.opt.clear_grad()
                if scheduler:
                    scheduler.step()
                self.print_log(loss, losses, epoch_id, num_epoch)
        else:
            for epoch_id in range(num_epoch):
                loss, losses = loss_func()
                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
                if scheduler:
                    scheduler.step()
                self.print_log(loss, losses, epoch_id, num_epoch)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        print("current lr is {}".format(self.opt.get_lr()))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = float(losses[0])
        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: ",
              float(loss), "eq_loss: ", eq_loss, "bc_loss: ",
              float(losses[1]), "supervised data_loss: ",
              float(losses[2]), "outlet_loss: ",
              float(losses[3]), "initial_loss: ", float(losses[4]))

        if (epoch_id + 1) % self.checkpoint_freq == 0:
            paddle.save(
                self.net.state_dict(),
                self.checkpoint_path + 'net_params_' + str(epoch_id + 1))

    def solve_bfgs(self, loss_func, num_epoch=1000, batch_size=None):
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
