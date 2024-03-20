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

""" Elasto-Plastic Neural Network (EPNN)

DEVELOPED AT:
                    COMPUTATIONAL GEOMECHANICS LABORATORY
                    DEPARTMENT OF CIVIL ENGINEERING
                    UNIVERSITY OF CALGARY, AB, CANADA
                    DIRECTOR: Prof. Richard Wan

DEVELOPED BY:
                    MAHDAD EGHBALIAN

MIT License

Copyright (c) 2022 Mahdad Eghbalian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Dict

import numpy as np
import paddle

import ppsci
from ppsci.utils import logger

# log for loss(total, state_elasto, state_plastic, stress), eval error(total, state_elasto, state_plastic, stress)
loss_log = {}  # for plotting
eval_log = {}
plot_keys = {"total", "state_elasto", "state_plastic", "stress"}
for key in plot_keys:
    loss_log[key] = []
    eval_log[key] = []


# transform
def transform_in(input):
    input_transformed = {}
    for key in input:
        input_transformed[key] = paddle.squeeze(input[key], axis=0)
    return input_transformed


def transform_out(input, out):
    # Add transformed input for computing loss
    out.update(input)
    return out


def transform_in_stress(input, model, out_key):
    input_elasto = model(input)[out_key]
    input_elasto = input_elasto.detach().clone()
    input_transformed = {}
    for key in input:
        input_transformed[key] = paddle.squeeze(input[key], axis=0)
    input_state_m = paddle.concat(
        x=(
            input_elasto,
            paddle.index_select(
                input_transformed["state_x"],
                paddle.to_tensor([0, 1, 2, 3, 7, 8, 9, 10, 11, 12]),
                axis=1,
            ),
        ),
        axis=1,
    )
    input_transformed["state_x_f"] = input_state_m
    return input_transformed


common_param = []
gkratio = paddle.to_tensor(
    data=[[0.45]], dtype=paddle.get_default_dtype(), stop_gradient=False
)


def val_loss_criterion(x, y):
    return 100.0 * (
        paddle.linalg.norm(x=x["input"] - y["input"]) / paddle.linalg.norm(x=y["input"])
    )


def train_loss_func(output_dict, *args) -> paddle.Tensor:
    """For model calculation of loss in model.train().

    Args:
        output_dict (Dict[str, paddle.Tensor]): The output dict.

    Returns:
        paddle.Tensor: Loss value.
    """
    # Use ppsci.loss.MAELoss to replace paddle.nn.L1Loss
    loss, loss_elasto, loss_plastic, loss_stress = loss_func(
        output_dict, ppsci.loss.MAELoss()
    )
    loss_log["total"].append(float(loss))
    loss_log["state_elasto"].append(float(loss_elasto))
    loss_log["state_plastic"].append(float(loss_plastic))
    loss_log["stress"].append(float(loss_stress))
    return loss


def eval_loss_func(output_dict, *args) -> paddle.Tensor:
    """For model calculation of loss in model.eval().

    Args:
        output_dict (Dict[str, paddle.Tensor]): The output dict.

    Returns:
        paddle.Tensor: Loss value.
    """
    error, error_elasto, error_plastic, error_stress = loss_func(
        output_dict, val_loss_criterion
    )
    eval_log["total"].append(float(error))
    eval_log["state_elasto"].append(float(error_elasto))
    eval_log["state_plastic"].append(float(error_plastic))
    eval_log["stress"].append(float(error_stress))
    logger.message(
        f"error(total): {float(error)}, error(error_elasto): {float(error_elasto)}, error(error_plastic): {float(error_plastic)}, error(error_stress): {float(error_stress)}"
    )
    return error


def metric_expr(output_dict, *args) -> Dict[str, paddle.Tensor]:
    return {"dummy_loss": paddle.to_tensor(0.0)}


def loss_func(output_dict, criterion) -> paddle.Tensor:
    (
        min_elasto,
        min_plastic,
        range_elasto,
        range_plastic,
        min_stress,
        range_stress,
    ) = common_param

    coeff1 = 2.0
    coeff2 = 1.0
    input_elasto = output_dict["out_state_elasto"]
    input_plastic = output_dict["out_state_plastic"]
    input_stress = output_dict["out_stress"]
    target_elasto = output_dict["state_y"][:, 0:1]
    target_plastic = output_dict["state_y"][:, 1:4]
    loss_elasto = criterion({"input": input_elasto}, {"input": target_elasto})
    loss_plastic = criterion({"input": input_plastic}, {"input": target_plastic})
    oneten_state = paddle.ones(shape=[3, 1], dtype=paddle.get_default_dtype())
    oneten_stress = paddle.ones(
        shape=[output_dict["stress_y"].shape[0], output_dict["stress_y"].shape[1]],
        dtype=paddle.get_default_dtype(),
    )
    dstrain = output_dict["state_x"][:, 10:]
    dstrain_real = (
        paddle.multiply(x=dstrain + coeff2, y=paddle.to_tensor(range_stress)) / coeff1
        + min_stress
    )
    # predict label
    dstrainpl = target_plastic
    dstrainpl_real = (
        paddle.multiply(x=dstrainpl + coeff2, y=paddle.to_tensor(range_elasto[1:4]))
        / coeff1
        + min_elasto[1:4]
    )
    # evaluate label
    dstrainel = dstrain_real - dstrainpl_real
    mu = paddle.multiply(x=gkratio, y=paddle.to_tensor(input_stress[:, 0:1]))
    mu_dstrainel = 2.0 * paddle.multiply(x=mu, y=paddle.to_tensor(dstrainel))
    stress_dstrainel = paddle.multiply(
        x=input_stress[:, 0:1] - 2.0 / 3.0 * mu,
        y=paddle.to_tensor(
            paddle.multiply(
                x=paddle.matmul(x=dstrainel, y=oneten_state),
                y=paddle.to_tensor(oneten_stress),
            )
        ),
    )
    input_stress = (
        coeff1
        * paddle.divide(
            x=mu_dstrainel + stress_dstrainel - min_plastic,
            y=paddle.to_tensor(range_plastic),
        )
        - coeff2
    )
    target_stress = output_dict["stress_y"]
    loss_stress = criterion({"input": input_stress}, {"input": target_stress})
    loss = loss_elasto + loss_plastic + loss_stress
    return loss, loss_elasto, loss_plastic, loss_stress


class Dataset:
    def __init__(self, data_state, data_stress, itrain):
        self.data_state = data_state
        self.data_stress = data_stress
        self.itrain = itrain

    def _cvt_to_ndarray(self, list_dict):
        for key in list_dict:
            list_dict[key] = np.asarray(list_dict[key])
        return list_dict

    def get(self, epochs=1):
        # Slow if using BatchSampler to obtain data
        input_dict_train = {
            "state_x": [],
            "state_y": [],
            "stress_x": [],
            "stress_y": [],
        }
        input_dict_val = {
            "state_x": [],
            "state_y": [],
            "stress_x": [],
            "stress_y": [],
        }
        label_dict_train = {"dummy_loss": []}
        label_dict_val = {"dummy_loss": []}
        for i in range(epochs):
            shuffled_indices = paddle.randperm(
                n=self.data_state.x_train.shape[0]
            ).numpy()
            input_dict_train["state_x"].append(
                self.data_state.x_train[shuffled_indices[0 : self.itrain]]
            )
            input_dict_train["state_y"].append(
                self.data_state.y_train[shuffled_indices[0 : self.itrain]]
            )
            input_dict_train["stress_x"].append(
                self.data_stress.x_train[shuffled_indices[0 : self.itrain]]
            )
            input_dict_train["stress_y"].append(
                self.data_stress.y_train[shuffled_indices[0 : self.itrain]]
            )
            label_dict_train["dummy_loss"].append(0.0)

        shuffled_indices = paddle.randperm(n=self.data_state.x_valid.shape[0]).numpy()
        input_dict_val["state_x"].append(
            self.data_state.x_valid[shuffled_indices[0 : self.itrain]]
        )
        input_dict_val["state_y"].append(
            self.data_state.y_valid[shuffled_indices[0 : self.itrain]]
        )
        input_dict_val["stress_x"].append(
            self.data_stress.x_valid[shuffled_indices[0 : self.itrain]]
        )
        input_dict_val["stress_y"].append(
            self.data_stress.y_valid[shuffled_indices[0 : self.itrain]]
        )
        label_dict_val["dummy_loss"].append(0.0)
        input_dict_train = self._cvt_to_ndarray(input_dict_train)
        label_dict_train = self._cvt_to_ndarray(label_dict_train)
        input_dict_val = self._cvt_to_ndarray(input_dict_val)
        label_dict_val = self._cvt_to_ndarray(label_dict_val)
        return input_dict_train, label_dict_train, input_dict_val, label_dict_val


class Data:
    def __init__(self, dataset_path, train_p=0.6, cross_valid_p=0.2, test_p=0.2):
        data = ppsci.utils.reader.load_dat_file(dataset_path)
        self.x = data["X"]
        self.y = data["y"]
        self.train_p = train_p
        self.cross_valid_p = cross_valid_p
        self.test_p = test_p

    def get_shuffled_data(self):
        # Need to set the seed, otherwise the loss will not match the precision
        ppsci.utils.misc.set_random_seed(seed=10)
        shuffled_indices = paddle.randperm(n=self.x.shape[0]).numpy()
        n_train = math.floor(self.train_p * self.x.shape[0])
        n_cross_valid = math.floor(self.cross_valid_p * self.x.shape[0])
        n_test = math.floor(self.test_p * self.x.shape[0])
        self.x_train = self.x[shuffled_indices[0:n_train]]
        self.y_train = self.y[shuffled_indices[0:n_train]]
        self.x_valid = self.x[shuffled_indices[n_train : n_train + n_cross_valid]]
        self.y_valid = self.y[shuffled_indices[n_train : n_train + n_cross_valid]]
        self.x_test = self.x[
            shuffled_indices[n_train + n_cross_valid : n_train + n_cross_valid + n_test]
        ]
        self.y_test = self.y[
            shuffled_indices[n_train + n_cross_valid : n_train + n_cross_valid + n_test]
        ]


def get_data(dataset_state, dataset_stress, ntrain_size):
    set_common_param(dataset_state, dataset_stress)

    data_state = Data(dataset_state)
    data_stress = Data(dataset_stress)
    data_state.get_shuffled_data()
    data_stress.get_shuffled_data()

    train_size_log10 = np.linspace(
        1, np.log10(data_state.x_train.shape[0]), num=ntrain_size
    )
    train_size_float = 10**train_size_log10
    train_size = train_size_float.astype(int)
    itrain = train_size[ntrain_size - 1]

    return Dataset(data_state, data_stress, itrain).get(10)


def set_common_param(dataset_state, dataset_stress):
    get_data = ppsci.utils.reader.load_dat_file(dataset_state)
    min_state = paddle.to_tensor(data=get_data["miny"])
    range_state = paddle.to_tensor(data=get_data["rangey"])
    min_dstrain = paddle.to_tensor(data=get_data["minx"][10:])
    range_dstrain = paddle.to_tensor(data=get_data["rangex"][10:])
    get_data = ppsci.utils.reader.load_dat_file(dataset_stress)
    min_stress = paddle.to_tensor(data=get_data["miny"])
    range_stress = paddle.to_tensor(data=get_data["rangey"])
    common_param.extend(
        [
            min_state,
            min_stress,
            range_state,
            range_stress,
            min_dstrain,
            range_dstrain,
        ]
    )


def get_model_list(
    nhlayers, nneurons, state_x_output_size, state_y_output_size, stress_x_output_size
):
    NHLAYERS_PLASTIC = 4
    NNEURONS_PLASTIC = 75
    hl_nodes_elasto = [nneurons] * nhlayers
    hl_nodes_plastic = [NNEURONS_PLASTIC] * NHLAYERS_PLASTIC
    node_sizes_state_elasto = [state_x_output_size]
    node_sizes_state_plastic = [state_x_output_size]
    node_sizes_stress = [stress_x_output_size + state_y_output_size - 6]
    node_sizes_state_elasto.extend(hl_nodes_elasto)
    node_sizes_state_plastic.extend(hl_nodes_plastic)
    node_sizes_stress.extend(hl_nodes_elasto)
    node_sizes_state_elasto.extend([state_y_output_size - 3])
    node_sizes_state_plastic.extend([state_y_output_size - 1])
    node_sizes_stress.extend([1])

    activation_elasto = "leaky_relu"
    activation_plastic = "leaky_relu"
    activations_elasto = [activation_elasto]
    activations_plastic = [activation_plastic]
    activations_elasto.extend([activation_elasto for ii in range(nhlayers)])
    activations_plastic.extend([activation_plastic for ii in range(NHLAYERS_PLASTIC)])
    activations_elasto.extend([activation_elasto])
    activations_plastic.extend([activation_plastic])
    drop_p = 0.0
    n_state_elasto = ppsci.arch.Epnn(
        ("state_x",),
        ("out_state_elasto",),
        tuple(node_sizes_state_elasto),
        tuple(activations_elasto),
        drop_p,
    )
    n_state_plastic = ppsci.arch.Epnn(
        ("state_x",),
        ("out_state_plastic",),
        tuple(node_sizes_state_plastic),
        tuple(activations_plastic),
        drop_p,
    )
    n_stress = ppsci.arch.Epnn(
        ("state_x_f",),
        ("out_stress",),
        tuple(node_sizes_stress),
        tuple(activations_elasto),
        drop_p,
    )
    return (n_state_elasto, n_state_plastic, n_stress)


def get_optimizer_list(model_list, cfg):
    optimizer_list = []
    lr_list = [0.001, 0.001, 0.01]
    for i, model in enumerate(model_list):
        scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
            **cfg.TRAIN.lr_scheduler, learning_rate=lr_list[i]
        )()
        optimizer_list.append(
            ppsci.optimizer.Adam(learning_rate=scheduler, weight_decay=0.0)(model)
        )

    scheduler_ratio = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler, learning_rate=0.001
    )()
    optimizer_list.append(
        paddle.optimizer.Adam(
            parameters=[gkratio], learning_rate=scheduler_ratio, weight_decay=0.0
        )
    )
    return ppsci.optimizer.OptimizerList(optimizer_list)


def plotting(output_dir):
    ppsci.utils.misc.plot_curve(
        data=eval_log,
        xlabel="Epoch",
        ylabel="Training Eval",
        output_dir=output_dir,
        smooth_step=1,
        use_semilogy=True,
    )
