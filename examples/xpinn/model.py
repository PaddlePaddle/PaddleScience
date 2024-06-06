from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
from paddle import nn

import ppsci
from ppsci.arch import base


class Model(base.Arch):
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
        residual1_u = self.net_subdomain1(input["residual1_x"], input["residual1_y"])
        residual2_u = self.net_subdomain2(input["residual2_x"], input["residual2_y"])
        residual3_u = self.net_subdomain3(input["residual3_x"], input["residual3_y"])
        interface1_u_sub1 = self.net_subdomain1(
            input["interface1_x"], input["interface1_y"]
        )
        interface1_u_sub2 = self.net_subdomain2(
            input["interface1_x"], input["interface1_y"]
        )
        interface2_u_sub1 = self.net_subdomain1(
            input["interface2_x"], input["interface2_y"]
        )
        interface2_u_sub3 = self.net_subdomain3(
            input["interface2_x"], input["interface2_y"]
        )
        boundary_u = self.net_subdomain1(input["boundary_x"], input["boundary_y"])

        return {
            "residual1_x": input["residual1_x"],
            "residual1_y": input["residual1_y"],
            "residual2_x": input["residual2_x"],
            "residual2_y": input["residual2_y"],
            "residual3_x": input["residual3_x"],
            "residual3_y": input["residual3_y"],
            "interface1_x": input["interface1_x"],
            "interface1_y": input["interface1_y"],
            "interface2_x": input["interface2_x"],
            "interface2_y": input["interface2_y"],
            "residual1_u": residual1_u,
            "residual2_u": residual2_u,
            "residual3_u": residual3_u,
            "interface1_u_sub1": interface1_u_sub1,
            "interface1_u_sub2": interface1_u_sub2,
            "interface2_u_sub1": interface2_u_sub1,
            "interface2_u_sub3": interface2_u_sub3,
            "boundary_u": boundary_u,
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
                dtype="float32",
                default_initializer=self.w_init((layers[l], layers[l + 1])),
            )
            bias = self.create_parameter(
                shape=[1, layers[l + 1]],
                dtype="float32",
                is_bias=True,
                default_initializer=nn.initializer.Constant(0.0),
            )
            amplitude = self.create_parameter(
                shape=[1],
                dtype="float32",
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

    def net_subdomain1(self, x: paddle.Tensor, y: paddle.Tensor):
        return self.neural_net_tanh(
            paddle.concat([x, y], 1), self.weights1, self.biases1, self.amplitudes1
        )

    def net_subdomain2(self, x: paddle.Tensor, y: paddle.Tensor):
        return self.neural_net_sin(
            paddle.concat([x, y], 1), self.weights2, self.biases2, self.amplitudes2
        )

    def net_subdomain3(self, x, y):
        return self.neural_net_cos(
            paddle.concat([x, y], 1), self.weights3, self.biases3, self.amplitudes3
        )
