# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Chengxi Zang
from __future__ import annotations

import math
from typing import Dict
from typing import Tuple

import paddle

from ppsci.arch import base
from ppsci.arch.moflow_glow import Glow
from ppsci.arch.moflow_glow import GlowOnGraph


def gaussian_nll(x, mean, ln_var, reduce="sum"):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing mean of a Gaussian distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing logarithm of variance of a Gaussian distribution,
            :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ("sum", "no"):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is given" % reduce
        )
    x_prec = paddle.exp(x=-ln_var)
    x_diff = x - mean
    x_power = x_diff * x_diff * x_prec * -0.5
    loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
    if reduce == "sum":
        return loss.sum()
    else:
        return loss


def rescale_adj(adj, type="all"):
    if type == "view":
        out_degree = adj.sum(axis=-1)
        out_degree_sqrt_inv = out_degree.pow(y=-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float("inf")] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(axis=-1) * adj
    else:
        num_neighbors = adj.sum(axis=(1, 2)).astype(dtype="float32")
        num_neighbors_inv = num_neighbors.pow(y=-1)
        num_neighbors_inv[num_neighbors_inv == float("inf")] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj
    return adj_prime


class MoFlowNet(base.Arch):
    """
    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("nodes","edges",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output","sum_log_det").
        hyper_params (object): More parameters derived from hyper_params for easy use.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        hyper_params: None,
    ):
        super(MoFlowNet, self).__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.hyper_params = hyper_params
        self.b_n_type = hyper_params.b_n_type
        self.a_n_node = hyper_params.a_n_node
        self.a_n_type = hyper_params.a_n_type
        self.b_size = self.a_n_node * self.a_n_node * self.b_n_type
        self.a_size = self.a_n_node * self.a_n_type
        self.noise_scale = hyper_params.noise_scale
        if hyper_params.learn_dist:
            self.ln_var = paddle.create_parameter(
                paddle.zeros(shape=[1]).shape,
                paddle.zeros(shape=[1]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.zeros(shape=[1])
                ),
            )

        else:
            self.register_buffer(name="ln_var", tensor=paddle.zeros(shape=[1]))
        self.bond_model = Glow(
            in_channel=hyper_params.b_n_type,
            n_flow=hyper_params.b_n_flow,
            n_block=hyper_params.b_n_block,
            squeeze_fold=hyper_params.b_n_squeeze,
            hidden_channel=hyper_params.b_hidden_ch,
            affine=hyper_params.b_affine,
            conv_lu=hyper_params.b_conv_lu,
        )
        self.atom_model = GlowOnGraph(
            n_node=hyper_params.a_n_node,
            in_dim=hyper_params.a_n_type,
            hidden_dim_dict={
                "gnn": hyper_params.a_hidden_gnn,
                "linear": hyper_params.a_hidden_lin,
            },
            n_flow=hyper_params.a_n_flow,
            n_block=hyper_params.a_n_block,
            mask_row_size_list=hyper_params.mask_row_size_list,
            mask_row_stride_list=hyper_params.mask_row_stride_list,
            affine=hyper_params.a_affine,
        )

    def forward(self, x):
        h = x[self.input_keys[0]]
        adj = x[self.input_keys[1]]
        adj_normalized = rescale_adj(adj).to(adj)

        if self.training:
            if self.noise_scale == 0:
                h = h / 2.0 - 0.5 + paddle.rand(shape=h.shape, dtype=h.dtype) * 0.4
            else:
                h = h + paddle.rand(shape=h.shape, dtype=h.dtype) * self.noise_scale
        h, sum_log_det_jacs_x = self.atom_model(adj_normalized, h)
        if self.training:
            if self.noise_scale == 0:
                adj = (
                    adj / 2.0
                    - 0.5
                    + paddle.rand(shape=adj.shape, dtype=adj.dtype) * 0.4
                )
            else:
                adj = (
                    adj
                    + paddle.rand(shape=adj.shape, dtype=adj.dtype) * self.noise_scale
                )
        adj_h, sum_log_det_jacs_adj = self.bond_model(adj)
        out = [h, adj_h]
        result_dict = {
            self.output_keys[0]: out,
            self.output_keys[1]: [sum_log_det_jacs_x, sum_log_det_jacs_adj],
        }

        return result_dict

    def reverse(self, z, true_adj=None):
        """
        Returns a molecule, given its latent vector.

        Args:
            z: latent vector. Shape: [B, N*N*M + N*T]    (100,369) 369=9*9 * 4 + 9*5
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
            true_adj: used for testing. An adjacency matrix of a real molecule

        return:
            adjacency matrix and feature matrix of a molecule
        """
        batch_size = tuple(z.shape)[0]
        with paddle.no_grad():
            z_x = z[:, : self.a_size]
            z_adj = z[:, self.a_size :]
            if true_adj is None:
                h_adj = z_adj.reshape(
                    [batch_size, self.b_n_type, self.a_n_node, self.a_n_node]
                )
                h_adj = self.bond_model.reverse(h_adj)
                if self.noise_scale == 0:
                    h_adj = (h_adj + 0.5) * 2
                adj = h_adj
                adj = adj + adj.transpose(perm=[0, 1, 3, 2])
                adj = adj / 2
                adj = paddle.nn.functional.softmax(adj, axis=1)
                max_bond = adj.max(axis=1).reshape(
                    [batch_size, -1, self.a_n_node, self.a_n_node]
                )
                adj = paddle.floor(x=adj / max_bond)
            else:
                adj = true_adj
            h_x = z_x.reshape([batch_size, self.a_n_node, self.a_n_type])
            adj_normalized = rescale_adj(adj).to(h_x)
            h_x = self.atom_model.reverse(adj_normalized, h_x)
            if self.noise_scale == 0:
                h_x = (h_x + 0.5) * 2
        return adj, h_x

    def log_prob_loss(self, output_dict: Dict, *args):
        losses = 0
        z = output_dict[self.output_keys[0]]
        logdet = output_dict[self.output_keys[1]]
        z[0] = z[0].reshape([tuple(z[0].shape)[0], -1])
        z[1] = z[1].reshape([tuple(z[1].shape)[0], -1])
        logdet[0] = logdet[0] - self.a_size * math.log(2.0)
        logdet[1] = logdet[1] - self.b_size * math.log(2.0)
        if len(self.ln_var) == 1:
            ln_var_adj = self.ln_var * paddle.ones(shape=[self.b_size]).to(z[0])
            ln_var_x = self.ln_var * paddle.ones(shape=[self.a_size]).to(z[0])
        else:
            ln_var_adj = self.ln_var[0] * paddle.ones(shape=[self.b_size]).to(z[0])
            ln_var_x = self.ln_var[1] * paddle.ones(shape=[self.a_size]).to(z[0])
        nll_adj = paddle.mean(
            paddle.sum(
                gaussian_nll(
                    z[1],
                    paddle.zeros(shape=self.b_size).to(z[0]),
                    ln_var_adj,
                    reduce="no",
                ),
                axis=1,
            )
            - logdet[1]
        )
        nll_adj = nll_adj / (self.b_size * math.log(2.0))
        nll_x = paddle.mean(
            paddle.sum(
                gaussian_nll(
                    z[0],
                    paddle.zeros(shape=self.a_size).to(z[0]),
                    ln_var_x,
                    reduce="no",
                ),
                axis=1,
            )
            - logdet[0]
        )
        nll_x = nll_x / (self.a_size * math.log(2.0))
        if nll_x.item() < 0:
            print("nll_x:{}".format(nll_x.item()))
        losses = nll_x + nll_adj
        return {"total_loss": losses}

    def save_hyperparams(self, path):
        self.hyper_params.save(path)


class MoFlowProp(base.Arch):
    """
    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("nodes","edges",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output","sum_log_det").
        model (MoFlowNet): pre-trained model.
        hidden_size (int): Hidden dimension list for output regression.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        model: MoFlowNet,
        hidden_size,
    ):
        super(MoFlowProp, self).__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.model = model
        self.latent_size = model.b_size + model.a_size
        self.hidden_size = hidden_size
        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh) - 1):
            modules.append(paddle.nn.Linear(in_features=vh[i], out_features=vh[i + 1]))
            if i < len(vh) - 2:
                modules.append(paddle.nn.Tanh())
        self.propNN = paddle.nn.Sequential(*modules)

    def encode(self, x):
        with paddle.no_grad():
            self.model.eval()
            output_dict = self.model(x)
            z = output_dict["output"]
            sum_log_det_jacs = output_dict["sum_log_det"]
            h = paddle.concat(
                [
                    z[0].reshape([tuple(z[0].shape)[0], -1]),
                    z[1].reshape([tuple(z[1].shape)[0], -1]),
                ],
                axis=1,
            )
        return h, sum_log_det_jacs

    def reverse(self, z):
        with paddle.no_grad():
            self.model.eval()
            adj, x = self.model.reverse(z, true_adj=None)
        return adj, x

    def forward(self, x):
        h, sum_log_det_jacs = self.encode(x)
        output = self.propNN(h)
        result_dict = {
            self.output_keys[0]: [h, output],
            self.output_keys[1]: sum_log_det_jacs,
        }

        return result_dict
