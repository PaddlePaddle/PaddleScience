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


class NetOut:
    def __init__(self, net, input):
        self._net = net
        self._input = input


class FCNet(NetworkBase):
    """
    Full connected network. Each layer consists of a matmul operator, an elementwise_add operator, and an activation function operator expect for the last layer.

    Parameters:
        num_ins (integer): Number of inputs.
        num_outs (integer): Number of outputs.
        num_layers (integer): Number of layers.
        hidden_size (integer): Hiden size in each layer.
        activation (optional, "tanh" / "sigmoid" / PaddlePaddle's operator): Activation function used in each layer. Currently, expected input is string format[sigmoid, tanh] or PaddlePaddle's operator (e.g. paddle.exp). The default value is "tanh".  

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

        self._weights = [None for i in range(num_layers)]
        self._biases = [None for i in range(num_layers)]
        self._weights_attr = [None for i in range(num_layers)]
        self._bias_attr = [None for i in range(num_layers)]

        act_str = {
            'sigmoid': F.sigmoid,
            'tanh': paddle.tanh,
            'exp': paddle.exp,
            'sin': paddle.sin,
            'cos': paddle.cos
        }
        if isinstance(activation, str) and (activation in act_str):
            self.activation = act_str.get(activation)
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                'Expected activation is String format[sigmoid, tanh] or Callable object.'
            )

        # dynamic mode: make network here
        # static  mode: make network in solver
        if paddle.in_dynamic_mode():
            self.make_network()

        # self.make_network_static()
        self.params_path = None

        # dynamic mode: net's parameters 
        # static  mode: None
    def parameters(self):
        if paddle.in_dynamic_mode():
            return super(FCNet, self).parameters()
        else:
            return None

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

            w_attr = self._weights_attr[i]
            b_attr = self._bias_attr[i]

            # create parameter with attr
            self._weights[i] = self.create_parameter(
                shape=[lsize, rsize],
                dtype=self._dtype,
                is_bias=False,
                attr=w_attr)
            self._biases[i] = self.create_parameter(
                shape=[rsize], dtype=self._dtype, is_bias=True, attr=b_attr)

            # add parameter
            self.add_parameter("w_" + str(i), self._weights[i])
            self.add_parameter("b_" + str(i), self._biases[i])

    def __nn_func_paddle(self, ins):
        u = ins
        for i in range(self.num_layers - 1):
            u = paddle.matmul(u, self._weights[i])
            u = paddle.add(u, self._biases[i])
            u = self.activation(u)
        u = paddle.matmul(u, self._weights[-1])
        u = paddle.add(u, self._biases[-1])
        return u

    def nn_func(self, ins, params=None):
        return self.__nn_func_paddle(ins)

    def __call__(self, input):
        return NetOut(self, input)

    def initialize(self,
                   path=None,
                   n=None,
                   weight_init=None,
                   bias_init=None,
                   learaning_rate=1.0):
        """
        Initialize network parameters. There are two methods:

        - initialize with parameters from file. This needs to specify parameter "path".
        - initialize with paddle.nn.initializer. This needs to specify parameter "n", "weight_init/bias_init" and "learning_rate".

        Parameters:
            path (string): parameter file 
            n (integer or list of integer): layers to initialize
            weight_init (paddle.nn.initializer): initializer used for weight
            bias_init (paddle.nn.initializer): initializer used for bias
            learning_rate (float, optional): learning rate 

        Example:
            >>> import paddlescience as psci
            >>> net = psci.network.FCNet(num_ins=2, num_outs=3, num_layers=10, hidden_size=20, activation="tanh")

            >>> # option 1: use file fc.pdparams to initialize
            >>> net.initialize(path="fc.pdparams")

            >>> # option 2: initialize layer 1 and layer 2 with constants
            >>> wcst = paddle.nn.initializer.Constant(2.0)
            >>> bcst = paddle.nn.initializer.Constant(3.0)
            >>> net.initialize(n=[1,2], weight_init=wcst, bias_init=bcst) 
        """

        if type(path) is str:
            self.params_path = path
            # In dynamic graph mode, load the params.
            # In static graph mode, just save the filename 
            # and initialize it in solver program.
            if paddle.in_dynamic_mode():
                layer_state_dict = paddle.load(path)
                self.set_state_dict(layer_state_dict)
        else:
            # convert int to list of int
            if isinstance(n, int):
                n = list(n)

            for i in n:
                # shape of parameter
                if i == 0:
                    lsize = self.num_ins
                    rsize = self.hidden_size
                elif i == (self.num_layers - 1):
                    lsize = self.hidden_size
                    rsize = self.num_outs
                else:
                    lsize = self.hidden_size
                    rsize = self.hidden_size

                # update weight 
                if weight_init is not None:
                    w_attr = paddle.ParamAttr(
                        name="w_" + str(i),
                        initializer=weight_init,
                        learning_rate=learaning_rate)
                    self._weights_attr[i] = w_attr
                    # dynamic mode: create parameter here
                    # static mode: create parameter in make_network
                    if paddle.in_dynamic_mode():
                        self._weights[i] = self.create_parameter(
                            shape=[lsize, rsize],
                            dtype=self._dtype,
                            is_bias=False,
                            attr=w_attr)
                        # add parameter
                        self.add_parameter("w_" + str(i), self._weights[i])

                # update bias
                if bias_init is not None:
                    b_attr = paddle.ParamAttr(
                        name="b_" + str(i),
                        initializer=bias_init,
                        learning_rate=learaning_rate)
                    self._bias_attr[i] = b_attr
                    # dynamic mode: create parameter here
                    # static mode: create parameter in make_network
                    if paddle.in_dynamic_mode():
                        self._biases[i] = self.create_parameter(
                            shape=[rsize],
                            dtype=self._dtype,
                            is_bias=True,
                            attr=b_attr)
                        # add parameter
                        self.add_parameter("b_" + str(i), self._biases[i])

    def flatten_params(self):
        flat_vars = list(map(paddle.flatten, self._weights + self._biases))
        return paddle.flatten(paddle.concat(flat_vars))

    def reconstruct(self, param_data):
        params = self._weights + self._biases
        param_sizes = [param.size for param in params]
        flat_params = paddle.split(param_data, param_sizes)
        is_biases = [False
                     for _ in self._weights] + [True for _ in self._biases]

        self._weights = []
        self._biases = []

        for old_param, flat_param, is_bias in zip(params, flat_params,
                                                  is_biases):
            shape = old_param.shape
            value = paddle.reshape(flat_param, shape)
            new_param = value
            if is_bias:
                self._biases.append(new_param)
            else:
                self._weights.append(new_param)

    def get_shared_layer(self):
        return self._weights[-1]
