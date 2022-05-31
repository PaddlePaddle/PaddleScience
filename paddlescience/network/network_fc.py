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

import paddle
import paddle.nn.functional as F
from .network_base import NetworkBase


class FCNet(NetworkBase):
    """
    Full connected network. Each layer consists of a matmul operator, an elementwise_add operator, and an activation function operator expect for the last layer.

    Parameters:
        num_ins (integer): Number of inputs.
        num_outs (integer): Number of outputs.
        num_layers (integer): Number of layers.
        hidden_size (integer): Hiden size in each layer.
        activation (optional, "tanh" / "sigmoid"): Activation function used in each layer. The default value is "tanh".

    Example:
        >>> import paddlescience as psci
        >>> net = psci.network.FCNet(2, 3, 10, 50, activiation='tanh')
    """

    def __init__(self,
                 num_ins,
                 num_outs,
                 num_layers,
                 hidden_size,
                 activation='tanh'):
        super(FCNet, self).__init__()

        self.num_ins = num_ins
        self.num_outs = num_outs
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.weights = []
        self.biases = []
        if activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = paddle.tanh
        else:
            assert 0, "Unsupported activation type."

        # dynamic mode: make network here
        # static  mode: make network in solver
        if paddle.in_dynamic_mode():
            self.make_network_dynamic()

        # self.make_network_static()

        # dynamic mode: net's parameters 
        # static  mode: None
    def parameters(self):
        if paddle.in_dynamic_mode():
            return super(FCNet, self).parameters()
        else:
            return None

    def make_network_dynamic(self):
        for i in range(self.num_layers):
            if i == 0:
                lsize = self.num_ins
                rsize = self.hidden_size
            elif i == (self.num_layers - 1):
                lsize = self.hidden_size
                rsize = self.num_outs
            else:
                lsize = self.hidden_size
                rsize = self.hidden_size

            w = self.create_parameter(
                shape=[lsize, rsize], dtype=self._dtype, is_bias=False)
            b = self.create_parameter(
                shape=[rsize], dtype=self._dtype, is_bias=True)
            self.weights.append(w)
            self.biases.append(b)
            self.add_parameter("w_" + str(i), w)
            self.add_parameter("b_" + str(i), b)

    def make_network_static(self):
        for i in range(self.num_layers):
            if i == 0:
                lsize = self.num_ins
                rsize = self.hidden_size
            elif i == (self.num_layers - 1):
                lsize = self.hidden_size
                rsize = self.num_outs
            else:
                lsize = self.hidden_size
                rsize = self.hidden_size

            w = paddle.static.create_parameter(
                shape=[lsize, rsize], dtype=self._dtype, is_bias=False)
            b = paddle.static.create_parameter(
                shape=[rsize], dtype=self._dtype, is_bias=True)

            self.weights.append(w)
            self.biases.append(b)
            self.add_parameter("w_" + str(i), w)
            self.add_parameter("b_" + str(i), b)

    def nn_func(self, ins):
        u = ins
        for i in range(self.num_layers - 1):
            u = paddle.matmul(u, self.weights[i])
            u = paddle.add(u, self.biases[i])
            u = self.activation(u)
        # if not paddle.in_dynamic_mode():
        #         paddle.static.Print(self.weights[-1])
        u = paddle.matmul(u, self.weights[-1])
        u = paddle.add(u, self.biases[-1])
        return u

    def flatten_params(self):
        flat_vars = list(map(paddle.flatten, self.weights + self.biases))
        return paddle.flatten(paddle.concat(flat_vars))

    def reconstruct(self, param_data):
        params = self.weights + self.biases
        param_sizes = [param.size for param in params]
        flat_params = paddle.split(param_data, param_sizes)
        is_biases = [False for _ in self.weights] + [True for _ in self.biases]

        self.weights = []
        self.biases = []

        for old_param, flat_param, is_bias in zip(params, flat_params,
                                                  is_biases):
            shape = old_param.shape
            value = paddle.reshape(flat_param, shape)
            # new_param = self.create_parameter(shape,
            #                                   dtype=self.dtype,
            #                                   is_bias=is_bias,
            #                                   default_initializer=Assign(value))
            # if is_bias:
            #     self.biases.append(new_param)
            # else:
            #     self.weights.append(new_param)
            # self.add_parameter(old_param.name.split('.')[-1], new_param)
            new_param = value
            if is_bias:
                self.biases.append(new_param)
            else:
                self.weights.append(new_param)
