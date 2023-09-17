# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from ppsci.arch import base


class DeepPhyLSTM2(base.Arch):
    """DeepPhyLSTM2 init function.

    Args:
        eta_shape_3 (int): The shape of the eta third dimension
    """

    def __init__(self, eta_shape_3):
        super().__init__()
        self.eta_shape_3 = eta_shape_3
        self.dataset = {}

        self.lstm_model = nn.Sequential(
            nn.LSTM(1, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Linear(100, 3 * eta_shape_3),
        )

        self.lstm_model_f = nn.Sequential(
            nn.LSTM(3 * eta_shape_3, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Linear(100, eta_shape_3),
        )

    def forward(self, dataset):
        # Reducing the dimension when constructing a data set
        dataset["ag"] = dataset["ag"][0]
        dataset["eta"] = dataset["eta"][0]
        dataset["eta_t"] = dataset["eta_t"][0]
        dataset["g"] = dataset["g"][0]
        dataset["lift"] = dataset["lift"][0]
        dataset["ag_c"] = dataset["ag_c"][0]
        dataset["phi"] = dataset["phi"][0]
        self.dataset.update(dataset)

        # physics informed neural networks
        (
            self.eta_pred,
            self.eta_t_pred,
            self.eta_tt_pred,
            self.eta_dot_pred,
            self.g_pred,
        ) = self.net_structure(self.dataset["ag"])

        self.eta_t_pred_c, self.eta_dot_pred_c, self.lift_c_pred = self.net_f(
            self.dataset["ag_c"]
        )

        # loss
        # for measurements
        self.loss_u = paddle.mean(paddle.square(self.dataset["eta"] - self.eta_pred))
        self.loss_udot = paddle.mean(
            paddle.square(self.dataset["eta_t"] - self.eta_dot_pred)
        )
        self.loss_g = paddle.mean(paddle.square(self.dataset["g"] - self.g_pred))
        # for collocations
        self.loss_ut_c = paddle.mean(
            paddle.square(self.eta_t_pred_c - self.eta_dot_pred_c)
        )
        self.loss_e = paddle.mean(
            paddle.square(
                paddle.matmul(
                    self.dataset["lift"],
                    paddle.ones(
                        [self.dataset["lift"].shape[0], 1, self.eta_shape_3],
                        dtype=paddle.get_default_dtype(),
                    ),
                )
                - self.lift_c_pred
            )
        )

        # total loss
        self.loss = self.loss_u + self.loss_udot + self.loss_ut_c + self.loss_e
        return {
            "loss": self.loss.reshape([1]),
            "loss_detach": self.loss.reshape([1]).detach().clone(),
        }

    def net_structure(self, ag):
        output = self.lstm_model(ag)
        eta = output[:, :, 0 : self.eta_shape_3]
        eta_dot = output[:, :, self.eta_shape_3 : 2 * self.eta_shape_3]
        g = output[:, :, 2 * self.eta_shape_3 :]

        # Phi and eta have different shape[0], using partial data
        phi = self.dataset["phi"]
        if eta.shape[0] < phi.shape[0]:
            phi_calc = phi[0 : eta.shape[0], :, :]
        else:
            phi_calc = phi

        eta_t = paddle.matmul(phi_calc, eta)
        eta_tt = paddle.matmul(phi_calc, eta_dot)
        return eta, eta_t, eta_tt, eta_dot, g

    def net_f(self, ag):
        eta, eta_t, eta_tt, eta_dot, g = self.net_structure(ag)
        eta_dot1 = eta_dot[:, :, 0:1]
        tmp = paddle.concat([eta, eta_dot1, g], 2)
        f = self.lstm_model_f(tmp)
        lift = eta_tt + f
        return eta_t, eta_dot, lift


class DeepPhyLSTM3(nn.Layer):
    """DeepPhyLSTM3 init function.

    Args:
        eta_shape_3 (int): The shape of the eta third dimension
    """

    def __init__(self, eta_shape_3):
        super().__init__()
        self.eta_shape_3 = eta_shape_3
        self.dataset = {}

        self.lstm_model = nn.Sequential(
            nn.LSTM(1, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3 * eta_shape_3),
        )

        self.lstm_model_f = nn.Sequential(
            nn.LSTM(3 * eta_shape_3, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, eta_shape_3),
        )

        self.lstm_model_g = nn.Sequential(
            nn.LSTM(2 * eta_shape_3, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, eta_shape_3),
        )

    def forward(self, dataset):
        dataset["ag"] = dataset["ag"][0]
        dataset["eta"] = dataset["eta"][0]
        dataset["eta_t"] = dataset["eta_t"][0]
        dataset["g"] = dataset["g"][0]
        dataset["lift"] = dataset["lift"][0]
        dataset["ag_c"] = dataset["ag_c"][0]
        dataset["phi"] = dataset["phi"][0]
        self.dataset.update(dataset)
        # physics informed neural networks
        (
            self.eta_pred,
            self.eta_t_pred,
            self.eta_tt_pred,
            self.eta_dot_pred,
            self.g_pred,
            self.g_t_pred,
        ) = self.net_structure(self.dataset["ag"])
        (
            self.eta_t_pred_c,
            self.eta_dot_pred_c,
            self.g_t_pred_c,
            self.g_dot_pred_c,
            self.lift_c_pred,
        ) = self.net_f(self.dataset["ag_c"])

        # loss
        # for measurements
        self.loss_u = paddle.mean(paddle.square(self.dataset["eta"] - self.eta_pred))
        self.loss_udot = paddle.mean(
            paddle.square(self.dataset["eta_t"] - self.eta_dot_pred)
        )
        self.loss_g = paddle.mean(paddle.square(self.dataset["g"] - self.g_pred))
        # for collocations
        self.loss_ut_c = paddle.mean(
            paddle.square(self.eta_t_pred_c - self.eta_dot_pred_c)
        )
        self.loss_gt_c = paddle.mean(paddle.square(self.g_t_pred_c - self.g_dot_pred_c))
        self.loss_e = paddle.mean(
            paddle.square(
                paddle.matmul(
                    self.dataset["lift"],
                    paddle.ones(
                        [self.dataset["lift"].shape[0], 1, self.eta_shape_3],
                        dtype=paddle.get_default_dtype(),
                    ),
                )
                - self.lift_c_pred
            )
        )

        self.loss = (
            self.loss_u + self.loss_udot + self.loss_ut_c + self.loss_gt_c + self.loss_e
        )
        return {
            "loss": self.loss.reshape([1]),
            "loss_detach": self.loss.reshape([1]).detach().clone(),
        }

    def net_structure(self, ag):
        output = self.lstm_model(ag)
        eta = output[:, :, 0 : self.eta_shape_3]
        eta_dot = output[:, :, self.eta_shape_3 : 2 * self.eta_shape_3]
        g = output[:, :, 2 * self.eta_shape_3 :]

        # Phi and eta have different shape[0], using partial data
        phi = self.dataset["phi"]
        if eta.shape[0] < phi.shape[0]:
            phi_calc = phi[0 : eta.shape[0], :, :]
        else:
            phi_calc = phi

        eta_t = paddle.matmul(phi_calc, eta)
        eta_tt = paddle.matmul(phi_calc, eta_dot)
        g_t = paddle.matmul(phi_calc, g)

        return eta, eta_t, eta_tt, eta_dot, g, g_t

    def net_f(self, ag):
        eta, eta_t, eta_tt, eta_dot, g, g_t = self.net_structure(ag)
        f = self.lstm_model_f(paddle.concat([eta, eta_dot, g], 2))
        lift = eta_tt + f

        eta_dot1 = eta_dot[:, :, 0:1]
        g_dot = self.lstm_model_g(paddle.concat([eta_dot1, g], 2))
        return eta_t, eta_dot, g_t, g_dot, lift
