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

import random

import paddle
import paddle.nn.functional as F


def metric_expr(output_dict, *args):
    return {"loss": paddle.to_tensor(0.0)}


# transform
def transform_in(input):
    input_transformed = {}
    input_transformed["ag"] = input["ag"][0]
    input_transformed["eta"] = input["eta"][0]
    input_transformed["eta_t"] = input["eta_t"][0]
    input_transformed["g"] = input["g"][0]
    input_transformed["lift"] = input["lift"][0]
    input_transformed["ag_c"] = input["ag_c"][0]
    input_transformed["phi"] = input["phi"][0]
    return input_transformed


def transform_out(input, out):
    # For calculate loss
    out.update(input)
    return out


def pde_loss_func(result_dict, *args):
    # loss
    # for measurements
    loss_u = F.mse_loss(result_dict["eta"], result_dict["eta_pred"])
    loss_udot = F.mse_loss(result_dict["eta_t"], result_dict["eta_dot_pred"])

    # for collocations
    loss_ut_c = F.mse_loss(result_dict["eta_t_pred_c"], result_dict["eta_dot_pred_c"])
    loss_e = F.mse_loss(
        paddle.matmul(
            result_dict["lift"],
            paddle.ones(
                [result_dict["lift"].shape[0], 1, result_dict["eta"].shape[2]],
                dtype=paddle.get_default_dtype(),
            ),
        ),
        result_dict["lift_pred_c"],
    )

    # total loss
    loss = loss_u + loss_udot + loss_ut_c + loss_e
    loss = paddle.square(loss)
    result_dict = {"loss": loss}
    return result_dict


def pde_loss_train_func(result_dict, *args):
    return pde_loss_func(result_dict, *args)["loss"]


def pde_loss_val_func(result_dict, *args):
    return pde_loss_func(result_dict, *args)["loss"].reshape([1])


def pde_loss_func3(result_dict, *args):
    # loss
    # for measurements
    loss_u = F.mse_loss(result_dict["eta"], result_dict["eta_pred"])
    loss_udot = F.mse_loss(result_dict["eta_t"], result_dict["eta_dot_pred"])

    # for collocations
    loss_ut_c = F.mse_loss(result_dict["eta_t_pred_c"], result_dict["eta_dot_pred_c"])
    loss_gt_c = F.mse_loss(result_dict["g_t_pred_c"], result_dict["g_dot_pred_c"])

    loss_e = F.mse_loss(
        paddle.matmul(
            result_dict["lift"],
            paddle.ones(
                [result_dict["lift"].shape[0], 1, result_dict["eta"].shape[2]],
                dtype=paddle.get_default_dtype(),
            ),
        ),
        result_dict["lift_pred_c"],
    )

    loss = loss_u + loss_udot + loss_ut_c + loss_gt_c + loss_e
    loss = paddle.square(loss)
    return {"loss": loss}


def pde_loss_train_func3(result_dict, *args):
    return pde_loss_func3(result_dict, *args)["loss"]


def pde_loss_val_func3(result_dict, *args):
    return pde_loss_func3(result_dict, *args)["loss"].reshape([1])


class Dataset:
    def __init__(self, eta, eta_t, g, ag, ag_c, lift, phi_t, ratio_split=0.8):
        self.eta = paddle.to_tensor(eta, dtype=paddle.get_default_dtype())
        self.eta_t = paddle.to_tensor(eta_t, dtype=paddle.get_default_dtype())
        self.g = paddle.to_tensor(g, dtype=paddle.get_default_dtype())
        self.ag = paddle.to_tensor(ag, dtype=paddle.get_default_dtype())
        self.lift = paddle.to_tensor(lift, dtype=paddle.get_default_dtype())
        self.ag_c = paddle.to_tensor(ag_c, dtype=paddle.get_default_dtype())
        self.phi_t = paddle.to_tensor(phi_t, dtype=paddle.get_default_dtype())
        self.ratio_split = ratio_split

    def get(self, epochs=1):
        input_dict_train = {
            "ag": [],
            "eta": [],
            "eta_t": [],
            "g": [],
            "lift": [],
            "ag_c": [],
            "phi": [],
        }
        label_dict_train = {"loss": []}
        input_dict_val = {
            "ag": [],
            "eta": [],
            "eta_t": [],
            "g": [],
            "lift": [],
            "ag_c": [],
            "phi": [],
        }
        label_dict_val = {"loss": []}
        for i in range(epochs):
            ind = list(range(self.ag.shape[0]))
            random.shuffle(ind)
            ratio_split = self.ratio_split
            ind_tr = ind[0 : round(ratio_split * self.ag.shape[0])]
            ind_val = ind[round(ratio_split * self.ag.shape[0]) :]

            self.ag_tr = self.ag[ind_tr]
            self.eta_tr = self.eta[ind_tr]
            self.eta_t_tr = self.eta_t[ind_tr]
            self.g_tr = self.g[ind_tr]

            self.ag_val = self.ag[ind_val]
            self.eta_val = self.eta[ind_val]
            self.eta_t_val = self.eta_t[ind_val]
            self.g_val = self.g[ind_val]

            input_dict_train["ag"].append(self.ag_tr)
            input_dict_train["eta"].append(self.eta_tr)
            input_dict_train["eta_t"].append(self.eta_t_tr)
            input_dict_train["g"].append(self.g_tr)
            input_dict_train["lift"].append(self.lift)
            input_dict_train["ag_c"].append(self.ag_c)
            input_dict_train["phi"].append(self.phi_t)
            label_dict_train["loss"].append(paddle.to_tensor(0.0))

            if i == epochs - 1:
                input_dict_val["ag"].append(self.ag_val)
                input_dict_val["eta"].append(self.eta_val)
                input_dict_val["eta_t"].append(self.eta_t_val)
                input_dict_val["g"].append(self.g_val)
                input_dict_val["lift"].append(self.lift)
                input_dict_val["ag_c"].append(self.ag_c)
                input_dict_val["phi"].append(self.phi_t)
                label_dict_val["loss"].append(paddle.to_tensor(0.0))

        return input_dict_train, label_dict_train, input_dict_val, label_dict_val
