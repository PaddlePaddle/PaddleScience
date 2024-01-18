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

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn

from ppsci.arch import phycrnet

dt = None
dx = None
num_time_batch = None
uv = None
time_steps = None

# transform
def transform_in(input):
    shape = input["initial_state_shape"][0]
    input_transformed = {
        "initial_state": input["initial_state"][0].reshape(shape.tolist()),
        "input": input["input"][0],
    }
    return input_transformed


def transform_out(input, out, model):
    # Stop the transform to avoid circulation
    model.enable_transform = False

    loss_func = phycrnet.loss_generator(dt, dx)
    batch_loss = 0
    state_detached = []
    prev_output = []
    for time_batch_id in range(num_time_batch):
        # update the first input for each time batch
        if time_batch_id == 0:
            hidden_state = input["initial_state"]
            u0 = input["input"]
        else:
            hidden_state = state_detached
            u0 = prev_output[-2:-1].detach()  # second last output
            out = model({"initial_state": hidden_state, "input": u0})

        # output is a list
        output = out["outputs"]
        second_last_state = out["second_last_state"]

        # [t, c, height (Y), width (X)]
        output = paddle.concat(tuple(output), axis=0)

        # concatenate the initial state to the output for central diff
        output = paddle.concat((u0.cuda(), output), axis=0)

        # get loss
        loss = compute_loss(output, loss_func)
        batch_loss += loss

        # update the state and output for next batch
        prev_output = output
        state_detached = []
        for i in range(len(second_last_state)):
            (h, c) = second_last_state[i]
            state_detached.append((h.detach(), c.detach()))  # hidden state

    model.enable_transform = True
    return {"loss": batch_loss}


def tranform_output_val(input, out, name="results.npz"):
    output = out["outputs"]
    input = input["input"]

    # shape: [t, c, h, w]
    output = paddle.concat(tuple(output), axis=0)
    output = paddle.concat((input.cuda(), output), axis=0)

    # Padding x and y axis due to periodic boundary condition
    output = paddle.concat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), axis=3)
    output = paddle.concat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), axis=2)

    # [t, c, h, w]
    truth = uv[0:time_steps, :, :, :]

    # [101, 2, 131, 131]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)
    truth = paddle.to_tensor(truth)
    # post-process
    ten_true = []
    ten_pred = []
    for i in range(0, 1001):
        u_star, u_pred, v_star, v_pred = post_process(
            output,
            truth,
            num=i,
        )

        ten_true.append(paddle.stack([u_star, v_star]))
        ten_pred.append(paddle.stack([u_pred, v_pred]))
    ten_true = paddle.stack(ten_true)
    ten_pred = paddle.stack(ten_pred)
    # compute the error
    # a-RMSE
    error = (
        paddle.sum((ten_pred - ten_true) ** 2, axis=(1, 2, 3))
        / ten_true.shape[2]
        / ten_true.shape[3]
    )
    N = error.shape[0]
    M = 0
    for i in range(N):
        M = M + np.eye(N, k=-i)
    M = M.T / np.arange(N)
    M[:, 0] = 0
    M[0, :] = 0
    M = paddle.to_tensor(M)
    aRMSE = paddle.sqrt(M.T @ error)
    np.savez(
        name,
        error=np.array(error),
        ten_true=ten_true,
        ten_pred=ten_pred,
        aRMSE=np.array(aRMSE),
    )
    error = paddle.linalg.norm(error)
    return {"loss": paddle.to_tensor([error])}


def train_loss_func(result_dict, *args) -> paddle.Tensor:
    """For model calculation of loss.

    Args:
        result_dict (Dict[str, paddle.Tensor]): The result dict.

    Returns:
        paddle.Tensor: Loss value.
    """
    return result_dict["loss"]


def val_loss_func(result_dict, *args) -> paddle.Tensor:
    return result_dict["loss"]


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
    output = paddle.concat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), axis=3)

    # Padding y axis due to periodic boundary condition
    output = paddle.concat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), axis=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, paddle.zeros_like(f_u).cuda()) + mse_loss(
        f_v, paddle.zeros_like(f_v).cuda()
    )

    return loss


def post_process(output, true, num):
    """
    num: Number of time step
    """
    u_star = true[num, 0, 1:-1, 1:-1]
    u_pred = output[num, 0, 1:-1, 1:-1].detach()

    v_star = true[num, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, 1:-1, 1:-1].detach()

    return u_star, u_pred, v_star, v_pred


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


def output_graph(model, input_dataset, fig_save_path, case_name):
    output_dataset = model(input_dataset)
    output = output_dataset["outputs"]
    input = input_dataset["input"][0]
    output = paddle.concat(tuple(output), axis=0)
    output = paddle.concat((input.cuda(), output), axis=0)

    # Padding x and y axis due to periodic boundary condition
    output = paddle.concat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), axis=3)
    output = paddle.concat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), axis=2)
    truth = uv[0:2001, :, :, :]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # post-process
    ten_true = []
    ten_pred = []

    for i in range(0, 100):
        u_star, u_pred, v_star, v_pred = post_process(output, truth, num=20 * i)
        ten_true.append([u_star, v_star])
        ten_pred.append([u_pred, v_pred])

    ten_true = np.stack(ten_true)
    ten_pred = np.stack(ten_pred)

    # compute the error
    # a-RMSE
    error = (
        np.sum((ten_pred - ten_true) ** 2, axis=(1, 2, 3))
        / ten_true.shape[2]
        / ten_true.shape[3]
    )
    N = error.shape[0]
    M = 0
    for i in range(N):
        M = M + np.eye(N, k=-i)
    M = M.T / np.arange(N)
    M[:, 0] = 0
    M[0, :] = 0

    M = paddle.to_tensor(M)
    aRMSE = paddle.sqrt(M.T @ error)
    t = np.linspace(0, 4, N)
    plt.plot(t, aRMSE, color="r")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("a-RMSE")
    plt.ylim((1e-4, 10))
    plt.xlim((0, 4))
    plt.legend(
        [
            "PhyCRNet",
        ],
        loc="upper left",
    )
    plt.title(case_name)
    plt.savefig(fig_save_path + "/error.jpg")

    _, ax = plt.subplots(3, 4, figsize=(18, 12))
    ax[0, 0].contourf(ten_true[25, 0])
    ax[0, 0].set_title("t=1")
    ax[0, 0].set_ylabel("truth")
    ax[1, 0].contourf(ten_pred[25, 0])
    ax[1, 0].set_ylabel("pred")
    ax[2, 0].contourf(ten_true[25, 0] - ten_pred[25, 0])
    ax[2, 0].set_ylabel("error")
    ax[0, 1].contourf(ten_true[50, 0])
    ax[0, 1].set_title("t=2")
    ax[1, 1].contourf(ten_pred[50, 0])
    ax[2, 1].contourf(ten_true[50, 0] - ten_pred[50, 0])
    ax[0, 2].contourf(ten_true[75, 0])
    ax[0, 2].set_title("t=3")
    ax[1, 2].contourf(ten_pred[75, 0])
    ax[2, 2].contourf(ten_true[75, 0] - ten_pred[75, 0])
    ax[0, 3].contourf(ten_true[99, 0])
    ax[0, 3].set_title("t=4")
    ax[1, 3].contourf(ten_pred[99, 0])
    ax[2, 3].contourf(ten_true[99, 0] - ten_pred[99, 0])
    plt.title(case_name)
    plt.savefig(fig_save_path + "/Burgers.jpg")
    plt.close()
