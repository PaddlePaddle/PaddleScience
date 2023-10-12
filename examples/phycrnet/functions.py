# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import Dict

import numpy as np
import paddle
import paddle.nn as nn


def metric_expr(output_dict, *args) -> Dict[str, paddle.Tensor]:
    return {"dummy_loss": paddle.to_tensor(0.0)}


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic"):
        self.dim = dim

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = paddle.concat(
                (
                    paddle.arange(start=0, end=k_max, step=1),
                    paddle.arange(start=-k_max, end=0, step=1),
                ),
                0,
            )

            self.sqrt_eig = (
                size
                * math.sqrt(2.0)
                * sigma
                * ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = paddle.concat(
                (
                    paddle.arange(start=0, end=k_max, step=1),
                    paddle.arange(start=-k_max, end=0, step=1),
                ),
                0,
            ).tile((size, 1))

            perm = list(range(wavenumers.ndim))
            perm[1] = 0
            perm[0] = 1
            k_x = wavenumers.transpose(perm=perm)
            k_y = wavenumers

            self.sqrt_eig = (
                (size**2)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = paddle.concat(
                (
                    paddle.arange(start=0, end=k_max, step=1),
                    paddle.arange(start=-k_max, end=0, step=1),
                ),
                0,
            ).tile((size, size, 1))

            perm = list(range(wavenumers.ndim))
            perm[1] = 2
            perm[2] = 1
            k_x = wavenumers.transpose(perm=perm)
            k_y = wavenumers

            perm = list(range(wavenumers.ndim))
            perm[0] = 2
            perm[2] = 0
            k_z = wavenumers.transpose(perm=perm)

            self.sqrt_eig = (
                (size**3)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2 + k_z**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = paddle.randn((N, *self.size, 2))

        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]

        if self.dim == 2:
            u = paddle.as_real(paddle.fft.ifft2(paddle.as_complex(coeff)))
        else:
            raise f"self.dim not in (2): {self.dim}"

        u = u[..., 0]

        return u


def compute_loss(output, loss_func):
    """calculate the physics loss"""

    # Padding x axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = paddle.concat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), axis=3)

    # Padding y axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = paddle.concat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), axis=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, paddle.zeros_like(f_u).cuda()) + mse_loss(
        f_v, paddle.zeros_like(f_v).cuda()
    )

    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)


def post_process(output, true, num):
    """
    num: Number of time step
    """
    u_star = true[num, 0, 1:-1, 1:-1]
    u_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

    v_star = true[num, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, 1:-1, 1:-1].detach().cpu().numpy()

    return u_star, u_pred, v_star, v_pred


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor**2))


class Dataset:
    def __init__(self, initial_state, input):
        self.initial_state = initial_state
        self.input = input

    def get(self, epochs=1):
        input_dict_train = {
            "initial_state": [],
            "initial_state_shape": [],
            "input": [],
        }
        label_dict_train = {"dummy_loss": []}
        input_dict_val = {
            "initial_state": [],
            "initial_state_shape": [],
            "input": [],
        }
        label_dict_val = {"dummy_loss": []}
        for i in range(epochs):
            # paddle not support rand >=7, so reshape, and then recover in input_transform
            shape = self.initial_state.shape
            input_dict_train["initial_state"].append(self.initial_state.reshape((-1,)))
            input_dict_train["initial_state_shape"].append(paddle.to_tensor(shape))
            input_dict_train["input"].append(self.input)
            label_dict_train["dummy_loss"].append(paddle.to_tensor(0.0))

            if i == epochs - 1:
                shape = self.initial_state.shape
                input_dict_val["initial_state"].append(
                    self.initial_state.reshape((-1,))
                )
                input_dict_val["initial_state_shape"].append(paddle.to_tensor(shape))
                input_dict_val["input"].append(self.input)
                label_dict_val["dummy_loss"].append(paddle.to_tensor(0.0))

        return input_dict_train, label_dict_train, input_dict_val, label_dict_val
