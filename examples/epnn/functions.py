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
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import paddle

import ppsci

loss_log = []  # for plotting
# log for loss(total, state1, state2, stress), error(total, state1, state2, stress), eval loss(total, state1, state2, stress)
for i in range(12):
    loss_log.append([])
OUTPUT_DIR = ""

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
    loss, loss11, loss12, loss2 = loss_func(output_dict, ppsci.loss.MAELoss())
    loss_log[0].append(loss.detach().clone())
    loss_log[1].append(loss11.detach().clone())
    loss_log[2].append(loss12.detach().clone())
    loss_log[3].append(loss2.detach().clone())
    error_total, error11, error12, error2 = loss_func(output_dict, val_loss_criterion)
    loss_log[4].append(error_total.detach().clone())
    loss_log[5].append(error11.detach().clone())
    loss_log[6].append(error12.detach().clone())
    loss_log[7].append(error2.detach().clone())
    return loss


def eval_loss_func(output_dict, *args) -> paddle.Tensor:
    error_total, error11, error12, error2 = loss_func(output_dict, val_loss_criterion)
    loss_log[8].append(error_total.detach().clone())
    loss_log[9].append(error11.detach().clone())
    loss_log[10].append(error12.detach().clone())
    loss_log[11].append(error2.detach().clone())

    loss_len = len(loss_log[0])
    if loss_len > 0 and loss_len % 1000 == 0:
        plot_loss()
    return error_total


def val_metric_func(output_dict, *args) -> paddle.Tensor:
    """For model calculation of loss in metric.

    Args:
        output_dict (Dict[str, paddle.Tensor]): The output dict.

    Returns:
        paddle.Tensor: Loss value.
    """
    loss, _, _, _ = loss_func(output_dict, val_loss_criterion)
    return loss


def loss_func(output_dict, criterion) -> paddle.Tensor:
    min1, min2, range1, range2, min3, range3 = common_param

    coeff1 = 2.0
    coeff2 = 1.0
    input11 = output_dict["out_state1"]
    input12 = output_dict["out_state2"]
    input21 = output_dict["out_stress"]
    target11 = output_dict["state_y"][:, 0:1]
    target12 = output_dict["state_y"][:, 1:4]
    loss11 = criterion({"input": input11}, {"input": target11})
    loss12 = criterion({"input": input12}, {"input": target12})
    oneten1 = paddle.ones(shape=[3, 1], dtype=paddle.get_default_dtype())
    oneten2 = paddle.ones(
        shape=[output_dict["stress_y"].shape[0], output_dict["stress_y"].shape[1]],
        dtype=paddle.get_default_dtype(),
    )
    dstrain = output_dict["state_x"][:, 10:]
    dstrain_real = (
        paddle.multiply(x=dstrain + coeff2, y=paddle.to_tensor(range3)) / coeff1 + min3
    )
    # predict label
    dstrainpl = target12
    dstrainpl_real = (
        paddle.multiply(x=dstrainpl + coeff2, y=paddle.to_tensor(range1[1:4])) / coeff1
        + min1[1:4]
    )
    # evaluate label
    dstrainel = dstrain_real - dstrainpl_real
    mu = paddle.multiply(x=gkratio, y=paddle.to_tensor(input21[:, 0:1]))
    input22 = 2.0 * paddle.multiply(x=mu, y=paddle.to_tensor(dstrainel))
    input23 = paddle.multiply(
        x=input21[:, 0:1] - 2.0 / 3.0 * mu,
        y=paddle.to_tensor(
            paddle.multiply(
                x=paddle.matmul(x=dstrainel, y=oneten1), y=paddle.to_tensor(oneten2)
            )
        ),
    )
    input2 = (
        coeff1 * paddle.divide(x=input22 + input23 - min2, y=paddle.to_tensor(range2))
        - coeff2
    )
    target2 = output_dict["stress_y"]
    loss2 = criterion({"input": input2}, {"input": target2})
    loss = loss11 + loss12 + loss2
    return loss, loss11, loss12, loss2


def metric_expr(output_dict, *args) -> Dict[str, paddle.Tensor]:
    return {"dummy_loss": val_metric_func(output_dict)}


class Dataset:
    def __init__(self, data_state, data_stress, itrain):
        self.data_state = data_state
        self.data_stress = data_stress
        self.itrain = itrain

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
            shuffled_indices = paddle.randperm(n=self.data_state.x_train.shape[0])
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
            label_dict_train["dummy_loss"].append(paddle.to_tensor(0.0))

        shuffled_indices = paddle.randperm(n=self.data_state.x_valid.shape[0])
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
        label_dict_val["dummy_loss"].append(paddle.to_tensor(0.0))
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
        shuffled_indices = paddle.randperm(n=self.x.shape[0])
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
    nhlayers1, nneurons, state_x_shape_1, state_y_shape_1, stress_x_shape_1
):
    NHLAYERS2 = 4
    NNEURONS2 = 75
    hl_nodes_1 = [nneurons] * nhlayers1
    hl_nodes_2 = [NNEURONS2] * NHLAYERS2
    node_sizes_state1 = [state_x_shape_1]
    node_sizes_state2 = [state_x_shape_1]
    node_sizes_stress = [stress_x_shape_1 + state_y_shape_1 - 6]
    node_sizes_state1.extend(hl_nodes_1)
    node_sizes_state2.extend(hl_nodes_2)
    node_sizes_stress.extend(hl_nodes_1)
    node_sizes_state1.extend([state_y_shape_1 - 3])
    node_sizes_state2.extend([state_y_shape_1 - 1])
    node_sizes_stress.extend([1])

    activation1 = "leaky_relu"
    activation2 = "leaky_relu"
    activations1 = [activation1]
    activations2 = [activation2]
    activations1.extend([activation1 for ii in range(nhlayers1)])
    activations2.extend([activation2 for ii in range(NHLAYERS2)])
    activations1.extend([activation1])
    activations2.extend([activation2])
    drop_p = 0.0
    n1_state1 = ppsci.arch.Epnn(
        ("state_x",),
        ("u",),
        tuple(node_sizes_state1),
        tuple(activations1),
        drop_p,
    )
    n1_state2 = ppsci.arch.Epnn(
        ("state_x",),
        ("v",),
        tuple(node_sizes_state2),
        tuple(activations2),
        drop_p,
    )
    n1_stress = ppsci.arch.Epnn(
        ("state_x_f",),
        ("w",),
        tuple(node_sizes_stress),
        tuple(activations1),
        drop_p,
    )
    return (n1_state1, n1_state2, n1_stress)


def get_optimizer_list(model_list, epochs, iters_per_epoch):
    scheduler_state1 = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        learning_rate=0.001,
        gamma=0.97,
        decay_steps=1,
    )()
    scheduler_state2 = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        learning_rate=0.001,
        gamma=0.97,
        decay_steps=1,
    )()
    scheduler_stress = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        learning_rate=0.01,
        gamma=0.97,
        decay_steps=1,
    )()
    scheduler_ratio = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        learning_rate=0.001,
        gamma=0.97,
        decay_steps=1,
    )()

    optimizer_state1 = ppsci.optimizer.Adam(
        learning_rate=scheduler_state1,
        weight_decay=0.0,
    )(model_list[0])
    optimizer_state2 = ppsci.optimizer.Adam(
        learning_rate=scheduler_state2,
        weight_decay=0.0,
    )(model_list[1])
    optimizer_stress = ppsci.optimizer.Adam(
        learning_rate=scheduler_stress,
        weight_decay=0.0,
    )(model_list[2])
    optimizer_ratio = paddle.optimizer.Adam(
        parameters=[gkratio], learning_rate=scheduler_ratio, weight_decay=0.0
    )
    optimizer_list = ppsci.optimizer.OptimizerList(
        (optimizer_state1, optimizer_state2, optimizer_stress, optimizer_ratio)
    )
    return optimizer_list


def plot_loss():
    font = {"weight": "normal", "size": 10}
    loss_log_np = [np.array(item) for item in loss_log]
    loss_len = len(loss_log_np[0])
    x = range(len(loss_log_np[0]))

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, loss_log_np[0])
    plt.plot(x, loss_log_np[1])
    plt.plot(x, loss_log_np[2])
    plt.plot(x, loss_log_np[3])
    plt.legend(["Loss Total", "Loss State1", "Loss State2", "Loss Stress"])
    plt.xlabel("Iteration ", fontdict=font)
    plt.ylabel("Loss ", fontdict=font)

    plt.subplot(1, 3, 2)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, loss_log_np[4])
    plt.plot(x, loss_log_np[5])
    plt.plot(x, loss_log_np[6])
    plt.plot(x, loss_log_np[7])
    plt.legend(["Error Total", "Error State1", "Error State2", "Error Stress"])
    plt.xlabel("Iteration ", fontdict=font)
    plt.ylabel("Error ", fontdict=font)

    plt.subplot(1, 3, 3)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, loss_log_np[8])
    plt.plot(x, loss_log_np[9])
    plt.plot(x, loss_log_np[10])
    plt.plot(x, loss_log_np[11])
    plt.legend(["Error Total", "Error State1", "Error State2", "Error Stress"])
    plt.xlabel("Iteration ", fontdict=font)
    plt.ylabel("Eval Error ", fontdict=font)
    plt.savefig(os.path.join(OUTPUT_DIR, f"loss_{loss_len}.png"))
