from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
from paddle import nn

import ppsci
from ppsci.arch import base


class Model(base.Arch):
    # Initialize the class
    def __init__(self, layer_list: Tuple[List[int], List[int], List[int]]):
        super().__init__()
        # Initialize NNs
        self.weights1, self.biases1, self.amplitudes1 = self.initialize_nn(
            layer_list[0], "layers1"
        )
        self.weights2, self.biases2, self.amplitudes2 = self.initialize_nn(
            layer_list[1], "layers2"
        )
        self.weights3, self.biases3, self.amplitudes3 = self.initialize_nn(
            layer_list[2], "layers3"
        )

    def forward(self, input: Dict[str, paddle.Tensor]):
        u1 = self.net_u1(input["x_f1"], input["y_f1"])
        u2 = self.net_u2(input["x_f2"], input["y_f2"])
        u3 = self.net_u3(input["x_f3"], input["y_f3"])
        u1i1 = self.net_u1(input["xi1"], input["yi1"])
        u2i1 = self.net_u2(input["xi1"], input["yi1"])
        u1i2 = self.net_u1(input["xi2"], input["yi2"])
        u3i2 = self.net_u3(input["xi2"], input["yi2"])
        ub_pred = self.net_u1(input["xb"], input["yb"])

        return {
            "x_f1": input["x_f1"],
            "y_f1": input["y_f1"],
            "x_f2": input["x_f2"],
            "y_f2": input["y_f2"],
            "x_f3": input["x_f3"],
            "y_f3": input["y_f3"],
            "xi1": input["xi1"],
            "yi1": input["yi1"],
            "xi2": input["xi2"],
            "yi2": input["yi2"],
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "u1i1": u1i1,
            "u2i1": u2i1,
            "u1i2": u1i2,
            "u3i2": u3i2,
            "ub_pred": ub_pred,
        }

    def initialize_nn(self, layers: List[int], name_prefix: str):
        # The weight used in neural_net
        weights = []
        # The bias used in neural_net
        biases = []
        # The amplitude used in neural_net
        amplitudes = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            weight = self.create_parameter(
                shape=[layers[l], layers[l + 1]],
                dtype="float64",
                default_initializer=self.w_init((layers[l], layers[l + 1])),
            )
            bias = self.create_parameter(
                shape=[1, layers[l + 1]],
                dtype="float64",
                is_bias=True,
                default_initializer=nn.initializer.Constant(0.0),
            )
            amplitude = self.create_parameter(
                shape=[1],
                dtype="float64",
                is_bias=True,
                default_initializer=nn.initializer.Constant(0.05),
            )

            self.add_parameter(name_prefix + "_w_" + str(l), weight)
            self.add_parameter(name_prefix + "_b_" + str(l), bias)
            self.add_parameter(name_prefix + "_a_" + str(l), amplitude)
            weights.append(weight)
            biases.append(bias)
            amplitudes.append(amplitude)
        return weights, biases, amplitudes

    def w_init(self, size: Tuple[int, int]):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        param = paddle.empty(size, "float64")
        param = ppsci.utils.initializer.trunc_normal_(param, 0.0, xavier_stddev)
        return nn.initializer.Assign(param)

    def neural_net_tanh(
        self,
        x: List[paddle.Tensor],
        weights: List[paddle.Tensor],
        biases: List[paddle.Tensor],
        amplitudes: List[paddle.Tensor],
    ):
        num_layers = len(weights) + 1

        h = x
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            h = paddle.tanh(20 * amplitudes[l] * paddle.add(paddle.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = paddle.add(paddle.matmul(h, w), b)
        return y

    def neural_net_sin(
        self,
        x: List[paddle.Tensor],
        weights: List[paddle.Tensor],
        biases: List[paddle.Tensor],
        amplitudes: List[paddle.Tensor],
    ):
        num_layers = len(weights) + 1

        h = x
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            h = paddle.sin(20 * amplitudes[l] * paddle.add(paddle.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = paddle.add(paddle.matmul(h, w), b)
        return y

    def neural_net_cos(
        self,
        x: List[paddle.Tensor],
        weights: List[paddle.Tensor],
        biases: List[paddle.Tensor],
        amplitudes: List[paddle.Tensor],
    ):
        num_layers = len(weights) + 1

        h = x
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            h = paddle.cos(20 * amplitudes[l] * paddle.add(paddle.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = paddle.add(paddle.matmul(h, w), b)
        return y

    def net_u1(self, x: paddle.Tensor, y: paddle.Tensor):
        return self.neural_net_tanh(
            paddle.concat([x, y], 1), self.weights1, self.biases1, self.amplitudes1
        )

    def net_u2(self, x: paddle.Tensor, y: paddle.Tensor):
        return self.neural_net_sin(
            paddle.concat([x, y], 1), self.weights2, self.biases2, self.amplitudes2
        )

    def net_u3(self, x, y):
        return self.neural_net_cos(
            paddle.concat([x, y], 1), self.weights3, self.biases3, self.amplitudes3
        )
