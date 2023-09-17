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


def pde_loss_func(output_dict, *args):
    return output_dict["loss"]


def pde_loss_val_func(output_dict, *args):
    return output_dict["loss"].detach().clone()


class Dataset:
    def __init__(self, eta, eta_t, g, ag, ag_c, lift, phi_t):
        self.eta = paddle.to_tensor(eta, dtype=paddle.get_default_dtype())
        self.eta_t = paddle.to_tensor(eta_t, dtype=paddle.get_default_dtype())
        self.g = paddle.to_tensor(g, dtype=paddle.get_default_dtype())
        self.ag = paddle.to_tensor(ag, dtype=paddle.get_default_dtype())
        self.lift = paddle.to_tensor(lift, dtype=paddle.get_default_dtype())
        self.ag_c = paddle.to_tensor(ag_c, dtype=paddle.get_default_dtype())
        self.phi_t = paddle.to_tensor(phi_t, dtype=paddle.get_default_dtype())

    def get(self, epochs=1):
        tf_dict = {
            "ag": [],
            "eta": [],
            "eta_t": [],
            "g": [],
            "lift": [],
            "ag_c": [],
            "phi": [],
        }
        label_dict = {"loss": []}
        tf_dict_val = {
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
            ratio_split = 0.8
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

            tf_dict["ag"].append(self.ag_tr)
            tf_dict["eta"].append(self.eta_tr)
            tf_dict["eta_t"].append(self.eta_t_tr)
            tf_dict["g"].append(self.g_tr)
            tf_dict["lift"].append(self.lift)
            tf_dict["ag_c"].append(self.ag_c)
            tf_dict["phi"].append(self.phi_t)
            label_dict["loss"].append(paddle.to_tensor(0.0))

            if i == epochs - 1:
                tf_dict_val["ag"].append(self.ag_val)
                tf_dict_val["eta"].append(self.eta_val)
                tf_dict_val["eta_t"].append(self.eta_t_val)
                tf_dict_val["g"].append(self.g_val)
                tf_dict_val["lift"].append(self.lift)
                tf_dict_val["ag_c"].append(self.ag_c)
                tf_dict_val["phi"].append(self.phi_t)
                label_dict_val["loss"].append(paddle.to_tensor(0.0))

        return tf_dict, label_dict, tf_dict_val, label_dict_val
