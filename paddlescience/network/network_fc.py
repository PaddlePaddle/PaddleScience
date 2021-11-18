# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
    def __init__(self,
                 num_ins,
                 num_outs,
                 num_layers,
                 hidden_size,
                 dtype='float32',
                 activation='tanh'):
        super(FCNet, self).__init__()

        self.num_ins = num_ins
        self.num_outs = num_outs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.weights = []
        self.biases = []
        self.activation = F.sigmoid if activation == 'sigmoid' else paddle.tanh
        self.make_network()

    def make_network(self):
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
                shape=[lsize, rsize], dtype=self.dtype, is_bias=False)
            b = self.create_parameter(
                shape=[rsize], dtype=self.dtype, is_bias=True)
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
        u = paddle.matmul(u, self.weights[-1])
        u = paddle.add(u, self.biases[-1])
        return u
