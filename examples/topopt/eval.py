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

import os

import numpy as np
import paddle
from data_utils import augmentation
from matplotlib import pyplot as plt
from paddle import nn
from prepare_datasets import generate_train_test
from sampler_utils import generate_sampler
from TopOptModel import TopOptNN

import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def evaluation_and_plot():
    args = config.parse_args()
    OUTPUT_DIR = "Output_TopOpt" if args.output_dir is None else args.output_dir
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
    DATA_PATH = "./Dataset/PreparedData/top_dataset.h5"
    # initialize logger
    logger.init_logger("ppsci", f"{RESULT_DIR}/results.log", "info")

    # specify evaluation parameters
    BATCH_SIZE = 32
    VOL_COEFF = 1  # coefficient for volume fraction constraint in the loss - beta in equation (3) in paper
    NUM_VAL_STEP = 10  # the number of iteration for each evaluation case
    MODEL_LIST = ["Poisson5", "Poisson10", "Poisson30", "Uniform"]

    # define loss
    def loss_expr(output_dict, label_dict, weight_dict=None):
        y = label_dict["output"].reshape((-1, 1))
        y_pred = output_dict["output"].reshape((-1, 1))
        conf_loss = paddle.mean(
            nn.functional.log_loss(y_pred, y, epsilon=1e-7)
        )  # epsilon = 1e-07 is the default in tf
        vol_loss = paddle.square(paddle.mean(y - y_pred))
        return conf_loss + VOL_COEFF * vol_loss

    # define metric
    def val_metric(output_dict, label_dict, weight_dict=None):
        output = output_dict["output"]
        y = label_dict["output"]
        accurates = paddle.equal(paddle.round(y), paddle.round(output))
        acc = paddle.mean(paddle.cast(accurates, dtype="float32"))
        w00 = paddle.sum(
            paddle.multiply(
                paddle.equal(paddle.round(output), 0.0),
                paddle.equal(paddle.round(y), 0.0),
            ),
            dtype=paddle.get_default_dtype(),
        )
        w11 = paddle.sum(
            paddle.multiply(
                paddle.equal(paddle.round(output), 1.0),
                paddle.equal(paddle.round(y), 1.0),
            ),
            dtype=paddle.get_default_dtype(),
        )
        w01 = paddle.sum(
            paddle.multiply(
                paddle.equal(paddle.round(output), 1.0),
                paddle.equal(paddle.round(y), 0.0),
            ),
            dtype=paddle.get_default_dtype(),
        )
        w10 = paddle.sum(
            paddle.multiply(
                paddle.equal(paddle.round(output), 0.0),
                paddle.equal(paddle.round(y), 1.0),
            ),
            dtype=paddle.get_default_dtype(),
        )
        n0 = paddle.add(w01, w00)
        n1 = paddle.add(w11, w10)
        iou = 0.5 * paddle.add(
            paddle.divide(w00, paddle.add(n0, w10)),
            paddle.divide(w11, paddle.add(n1, w01)),
        )
        current_acc_results.append(np.array(acc))
        current_iou_results.append(np.array(iou))
        return {"Binary_Acc": acc, "IoU": iou}

    # fixed iteration stop times for evaluation
    iterations_stop_times = range(5, 85, 5)
    model = TopOptNN()

    # evaluation for 4 cases
    acc_results_summary = {}
    iou_results_summary = {}
    for model_name in MODEL_LIST:

        # load model parameters
        model_path = (
            f"{OUTPUT_DIR}/{model_name}_vol_coeff{VOL_COEFF}/checkpoints/latest"
        )
        solver = ppsci.solver.Solver(model, pretrained_model_path=model_path)
        solver.epochs = 1
        solver.iters_per_epoch = NUM_VAL_STEP
        acc_results = []
        iou_results = []

        # evaluation for different fixed iteration stop times
        for stop_iter in iterations_stop_times:
            # only evaluate for NUM_VAL_STEP times of iteration
            X_data, Y_data = generate_train_test(
                DATA_PATH, 1, BATCH_SIZE * NUM_VAL_STEP
            )
            sup_validator = ppsci.validate.SupervisedValidator(
                {
                    "dataset": {
                        "name": "NamedArrayDataset",
                        "input": {"input": X_data},
                        "label": {"output": Y_data},
                    },
                    "batch_size": BATCH_SIZE,
                    "sampler": {
                        "name": "BatchSampler",
                        "drop_last": False,
                        "shuffle": True,
                    },
                    "transforms": (
                        {
                            "FunctionalTransform": {
                                "transform_func": augmentation,
                            },
                        },
                    ),
                },
                ppsci.loss.FunctionalLoss(loss_expr),
                {"output": lambda out: out["output"]},
                {"metric": ppsci.metric.FunctionalMetric(val_metric)},
                name="sup_validator",
            )
            validator = {sup_validator.name: sup_validator}
            solver.validator = validator

            # modify the channel_sampler in model
            SIMP_stop_point_sampler = generate_sampler("Fixed", stop_iter)
            solver.model.channel_sampler = SIMP_stop_point_sampler

            current_acc_results = []
            current_iou_results = []
            solver.eval()

            acc_results.append(np.mean(current_acc_results))
            iou_results.append(np.mean(current_iou_results))

        acc_results_summary[model_name] = acc_results
        iou_results_summary[model_name] = iou_results

    # calculate thresholding results
    th_acc_results = []
    th_iou_results = []
    for stop_iter in iterations_stop_times:
        SIMP_stop_point_sampler = generate_sampler("Fixed", stop_iter)

        current_acc_results = []
        current_iou_results = []

        # only calculate for NUM_VAL_STEP times of iteration
        for i in range(10):
            x, y = generate_train_test(DATA_PATH, 1, BATCH_SIZE)
            # thresholding
            k = SIMP_stop_point_sampler()
            x1 = paddle.to_tensor(x)[:, k, :, :]
            x2 = paddle.to_tensor(x)[:, k - 1, :, :]
            x = paddle.stack((x1, x1 - x2), axis=1)
            out = paddle.cast(paddle.to_tensor(x)[:, 0:1, :, :] > 0.5, dtype="float32")
            val_metric({"output": out}, {"output": paddle.to_tensor(y)})

        th_acc_results.append(np.mean(current_acc_results))
        th_iou_results.append(np.mean(current_iou_results))

    acc_results_summary["thresholding"] = th_acc_results
    iou_results_summary["thresholding"] = th_iou_results

    # plot and save figures
    plt.figure(figsize=(12, 6))
    for k, v in acc_results_summary.items():
        plt.plot(iterations_stop_times, v, label=k, lw=4)
    plt.title("Binary accuracy", fontsize=16)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("accuracy", fontsize=14)
    plt.legend(loc="best", fontsize=13)
    plt.grid()
    plt.savefig(os.path.join(RESULT_DIR, "Binary_Accuracy.png"))
    plt.show()

    plt.figure(figsize=(12, 6))
    for k, v in iou_results_summary.items():
        plt.plot(iterations_stop_times, v, label=k, lw=4)
    plt.title("IoU", fontsize=16)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("accuracy", fontsize=14)
    plt.legend(loc="best", fontsize=13)
    plt.grid()
    plt.savefig(os.path.join(RESULT_DIR, "IoU.png"))
    plt.show()
