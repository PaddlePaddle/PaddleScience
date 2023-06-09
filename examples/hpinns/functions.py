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

"""
This module is heavily adapted from https://github.com/lululxvi/hpinn
"""

from typing import Dict
from typing import List

import numpy as np
import paddle
import paddle.nn.functional as F

"""All functions used in hpinns example, including functions of transform and loss."""

# define constants
BOX = np.array([[-2, -2], [2, 3]])
DPML = 1
OMEGA = 2 * np.pi
SIGMA0 = -np.log(1e-20) / (4 * DPML**3 / 3)
l_BOX = BOX + np.array([[-DPML, -DPML], [DPML, DPML]])
beta = 2.0
mu = 2

# define variables which will be updated during training
lambda_re: np.ndarray = None
lambda_im: np.ndarray = None
loss_weight: List[float] = None
input_dict: Dict[str, paddle.Tensor] = None
train_mode: str = None

# define log variables for plotting
loss_log = []  # record all losses, [pde, lag, obj]
loss_obj = 0.0  # record last objective loss of each k
lambda_log = []  # record all lambdas


# transform
def transform_in(_in):
    global input_dict
    input_dict = _in
    # Periodic BC in x
    P = BOX[1][0] - BOX[0][0] + 2 * DPML
    w = 2 * np.pi / P
    x, y = input_dict["x"], input_dict["y"]
    input_transformed = {}
    for t in range(1, 7):
        input_transformed[f"x_cos_{t}"] = paddle.cos(t * w * x)
        input_transformed[f"x_sin_{t}"] = paddle.sin(t * w * x)
    input_transformed["y"] = y
    input_transformed["y_cos_1"] = paddle.cos(OMEGA * y)
    input_transformed["y_sin_1"] = paddle.sin(OMEGA * y)

    return input_transformed


def transform_out_all(var):
    x, y = input_dict["x"], input_dict["y"]
    # Zero Dirichlet BC
    a, b = BOX[0][1] - DPML, BOX[1][1] + DPML
    t = (1 - paddle.exp(a - y)) * (1 - paddle.exp(y - b))
    return t * var


def transform_out_real_part(out):
    re = out["e_re"]
    trans_out = transform_out_all(re)
    return {"e_real": trans_out}


def transform_out_imaginary_part(out):
    im = out["e_im"]
    trans_out = transform_out_all(im)
    return {"e_imaginary": trans_out}


def transform_out_epsilon(out):
    eps = out["eps"]
    # 1 <= eps <= 12
    eps = F.sigmoid(eps) * 11 + 1
    return {"epsilon": eps}


# loss
def init_lambda(output_dict: Dict[str, paddle.Tensor], bound: int):
    """Init lambdas of Lagrangian and weights of losses.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.
        bound (int): The bound of the data range that should be used.
    """
    global lambda_re, lambda_im, loss_weight
    x, y = output_dict["x"], output_dict["y"]
    lambda_re = np.zeros((len(x[bound:]), 1))
    lambda_im = np.zeros((len(y[bound:]), 1))
    # loss_weight: [PDE loss 1, PDE loss 2, Lagrangian loss 1, Lagrangian loss 2, objective loss]
    if train_mode == "aug_lag":
        loss_weight = [0.5 * mu] * 2 + [1.0, 1.0] + [1.0]
    else:
        loss_weight = [0.5 * mu] * 2 + [0.0, 0.0] + [1.0]


def update_lambda(output_dict: Dict[str, paddle.Tensor], bound: int):
    """Update lambdas of Lagrangian.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.
        bound (int): The bound of the data range that should be used.
    """
    global lambda_re, lambda_im, lambda_log
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    loss_re = loss_re[bound:]
    loss_im = loss_im[bound:]
    lambda_re += mu * loss_re.numpy()
    lambda_im += mu * loss_im.numpy()
    lambda_log.append([lambda_re.copy().squeeze(), lambda_im.copy().squeeze()])


def update_mu():
    """Update mu."""
    global mu, loss_weight
    mu *= beta
    loss_weight[:2] = [0.5 * mu] * 2


def _sigma_1(d):
    return SIGMA0 * d**2 * np.heaviside(d, 0)


def _sigma_2(d):
    return 2 * SIGMA0 * d * np.heaviside(d, 0)


def sigma(x, a, b):
    """sigma(x) = 0 if a < x < b, else grows cubically from zero."""
    return _sigma_1(a - x) + _sigma_1(x - b)


def dsigma(x, a, b):
    return -_sigma_2(a - x) + _sigma_2(x - b)


def perfectly_matched_layers(x: paddle.Tensor, y: paddle.Tensor):
    """Apply the technique of perfectly matched layers(PMLs) proposed by paper arXiv:2108.05348.

    Args:
        x (paddle.Tensor): one of input contains tensor.
        y (paddle.Tensor): one of input contains tensor.

    Returns:
        np.ndarray: Parameters of pde formula.
    """
    x = x.numpy()
    y = y.numpy()

    sigma_x = sigma(x, BOX[0][0], BOX[1][0])
    AB1 = 1 / (1 + 1j / OMEGA * sigma_x) ** 2
    A1, B1 = AB1.real, AB1.imag

    dsigma_x = dsigma(x, BOX[0][0], BOX[1][0])
    AB2 = -1j / OMEGA * dsigma_x * AB1 / (1 + 1j / OMEGA * sigma_x)
    A2, B2 = AB2.real, AB2.imag

    sigma_y = sigma(y, BOX[0][1], BOX[1][1])
    AB3 = 1 / (1 + 1j / OMEGA * sigma_y) ** 2
    A3, B3 = AB3.real, AB3.imag

    dsigma_y = dsigma(y, BOX[0][1], BOX[1][1])
    AB4 = -1j / OMEGA * dsigma_y * AB3 / (1 + 1j / OMEGA * sigma_y)
    A4, B4 = AB4.real, AB4.imag
    return A1, B1, A2, B2, A3, B3, A4, B4


def obj_func_J(y):
    # Approximate the objective function
    y = y.numpy() + 1.5
    h = 0.2
    return 1 / (h * np.pi**0.5) * np.exp(-((y / h) ** 2)) * (np.abs(y) < 0.5)


def compute_real_and_imaginary_loss(
    output_dict: Dict[str, paddle.Tensor]
) -> paddle.Tensor:
    """Compute real and imaginary_loss.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.

    Returns:
        paddle.Tensor: Real and imaginary_loss.
    """
    x, y = output_dict["x"], output_dict["y"]
    e_re = output_dict["e_real"]
    e_im = output_dict["e_imaginary"]
    eps = output_dict["epsilon"]

    condition = np.logical_and(y.numpy() < 0, y.numpy() > -1).astype(
        paddle.get_default_dtype()
    )

    eps = eps * condition + 1 - condition

    de_re_x = output_dict["de_re_x"]
    de_re_y = output_dict["de_re_y"]
    de_re_xx = output_dict["de_re_xx"]
    de_re_yy = output_dict["de_re_yy"]
    de_im_x = output_dict["de_im_x"]
    de_im_y = output_dict["de_im_y"]
    de_im_xx = output_dict["de_im_xx"]
    de_im_yy = output_dict["de_im_yy"]

    a1, b1, a2, b2, a3, b3, a4, b4 = perfectly_matched_layers(x, y)

    loss_re = (
        (a1 * de_re_xx + a2 * de_re_x + a3 * de_re_yy + a4 * de_re_y) / OMEGA
        - (b1 * de_im_xx + b2 * de_im_x + b3 * de_im_yy + b4 * de_im_y) / OMEGA
        + eps * OMEGA * e_re
    )
    loss_im = (
        (a1 * de_im_xx + a2 * de_im_x + a3 * de_im_yy + a4 * de_im_y) / OMEGA
        + (b1 * de_re_xx + b2 * de_re_x + b3 * de_re_yy + b4 * de_re_y) / OMEGA
        + eps * OMEGA * e_im
        + obj_func_J(y)
    )
    return loss_re, loss_im


def pde_loss_fun(output_dict: Dict[str, paddle.Tensor]) -> paddle.Tensor:
    """Compute pde loss and lagrangian loss.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.

    Returns:
        paddle.Tensor: PDE loss (and lagrangian loss if using Augmented Lagrangian method).
    """
    global loss_log
    bound = output_dict["bound"]
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    loss_re = loss_re[bound:]
    loss_im = loss_im[bound:]

    loss_eqs1 = paddle.mean(loss_re**2)
    loss_eqs2 = paddle.mean(loss_im**2)
    # augmented_Lagrangian
    if lambda_im is None:
        init_lambda(output_dict, bound)
    loss_lag1 = paddle.mean(loss_re * lambda_re)
    loss_lag2 = paddle.mean(loss_im * lambda_im)

    losses = (
        loss_weight[0] * loss_eqs1
        + loss_weight[1] * loss_eqs2
        + loss_weight[2] * loss_lag1
        + loss_weight[3] * loss_lag2
    )
    loss_log.append(float(loss_eqs1 + loss_eqs2))  # for plotting
    loss_log.append(float(loss_lag1 + loss_lag2))  # for plotting
    return losses


def obj_loss_fun(output_dict: Dict[str, paddle.Tensor]) -> paddle.Tensor:
    """Compute objective loss.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.

    Returns:
        paddle.Tensor: Objective loss.
    """
    global loss_log, loss_obj
    x, y = output_dict["x"], output_dict["y"]
    bound = output_dict["bound"]
    e_re = output_dict["e_real"]
    e_im = output_dict["e_imaginary"]

    f1 = paddle.heaviside((x + 0.5) * (0.5 - x), paddle.to_tensor(0.5))
    f2 = paddle.heaviside((y - 1) * (2 - y), paddle.to_tensor(0.5))
    j = e_re[:bound] ** 2 + e_im[:bound] ** 2 - f1[:bound] * f2[:bound]
    loss_opt_area = paddle.mean(j**2)

    if lambda_im is None:
        init_lambda(output_dict, bound)
    losses = loss_weight[4] * loss_opt_area
    loss_log.append(float(loss_opt_area))  # for plotting
    loss_obj = float(loss_opt_area)  # for plotting
    return losses


def eval_loss_fun(output_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Compute objective loss for evalution.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.

    Returns:
        paddle.Tensor: Objective loss.
    """
    x, y = output_dict["x"], output_dict["y"]
    e_re = output_dict["e_real"]
    e_im = output_dict["e_imaginary"]

    f1 = paddle.heaviside((x + 0.5) * (0.5 - x), paddle.to_tensor(0.5))
    f2 = paddle.heaviside((y - 1) * (2 - y), paddle.to_tensor(0.5))
    j = e_re**2 + e_im**2 - f1 * f2
    losses = paddle.mean(j**2)

    return losses


def eval_metric_fun(output_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Compute metric for evalution.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Dict of outputs contains tensor.

    Returns:
        Dict[str, paddle.Tensor]: MSE metric.
    """
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    eps_opt = paddle.concat([loss_re, loss_im], axis=-1)
    loss = paddle.mean(eps_opt**2)

    metric_dict = {"eval_loss": loss}
    return metric_dict
