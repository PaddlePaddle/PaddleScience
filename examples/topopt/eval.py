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
from matplotlib import pyplot as plt
from sampler_utils import generate_sampler
from TopOptModel import TopOptNN

import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def evaluation_and_plot(dataloader):
    args = config.parse_args()
    OUTPUT_DIR = "Output_TopOpt" if args.output_dir is None else args.output_dir
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "results")
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/results.log", "info")

    # specify evaluation parameters
    BATCH_SIZE = 32
    VOL_COEFF = 1  # coefficient for volume fraction constraint in the loss - beta in equation (3) in paper
    NUM_VAL_STEP = 20  # the number of iteration for each evaluation case
    MODEL_LIST = ["Poisson5", "Poisson10", "Poisson30", "Uniform"]

    # define metric
    def metric_expr(output_dict, label_dict, weight_dict=None):
        output = output_dict["output"]
        y = label_dict["output"]
        accurates = paddle.equal(paddle.round(y), paddle.round(output))
        acc = paddle.mean(paddle.cast(accurates, dtype="float32"))
        w00 = paddle.cast(
            paddle.sum(
                paddle.multiply(
                    paddle.equal(paddle.round(output), 0.0),
                    paddle.equal(paddle.round(y), 0.0),
                )
            ),
            dtype="float32",
        )
        w11 = paddle.cast(
            paddle.sum(
                paddle.multiply(
                    paddle.equal(paddle.round(output), 1.0),
                    paddle.equal(paddle.round(y), 1.0),
                )
            ),
            dtype="float32",
        )
        w01 = paddle.cast(
            paddle.sum(
                paddle.multiply(
                    paddle.equal(paddle.round(output), 1.0),
                    paddle.equal(paddle.round(y), 0.0),
                )
            ),
            dtype="float32",
        )
        w10 = paddle.cast(
            paddle.sum(
                paddle.multiply(
                    paddle.equal(paddle.round(output), 0.0),
                    paddle.equal(paddle.round(y), 1.0),
                )
            ),
            dtype="float32",
        )
        n0 = paddle.add(w01, w00)
        n1 = paddle.add(w11, w10)
        iou = 0.5 * paddle.add(
            paddle.divide(w00, paddle.add(n0, w10)),
            paddle.divide(w11, paddle.add(n1, w01)),
        )
        return {"Binary_Acc": acc, "IoU": iou}

    # fixed iteration stop times for evaluation
    iterations_stop_times = range(5, 85, 5)
    model = TopOptNN()

    # evaluation for 4 cases
    acc_results_summary = {}
    iou_results_summary = {}
    for model_name in MODEL_LIST:

        # load model parameters
        model_path = "Output_TopOpt" if args.output_dir is None else args.output_dir
        model_path = os.path.join(
            model_path,
            "".join([model_name, "_vol_coeff", str(VOL_COEFF)]),
            "checkpoints",
            "latest",
        )
        solver = ppsci.solver.Solver(model, pretrained_model_path=model_path)
        acc_results = []
        iou_results = []

        # evaluation for different fixed iteration stop times
        for stop_iter in iterations_stop_times:

            # modify the channel_sampler in model
            SIMP_stop_point_sampler = generate_sampler("Fixed", stop_iter)
            solver.model.channel_sampler = SIMP_stop_point_sampler

            total_val_steps = 0
            current_acc_results = []
            current_iou_results = []

            # only evaluate for NUM_VAL_STEP times of iteration
            for x, y, _ in iter(dataloader):
                if total_val_steps >= NUM_VAL_STEP:
                    break
                out = solver.predict(x, batch_size=BATCH_SIZE)
                metric = metric_expr(out, y)
                current_acc_results.append(np.array(metric["Binary_Acc"]))
                current_iou_results.append(np.array(metric["IoU"]))
                total_val_steps += 1

            acc_results.append(np.mean(current_acc_results))
            iou_results.append(np.mean(current_iou_results))

        acc_results_summary[model_name] = acc_results
        iou_results_summary[model_name] = iou_results

    # calculate thresholding results
    th_acc_results = []
    th_iou_results = []
    for stop_iter in iterations_stop_times:
        SIMP_stop_point_sampler = generate_sampler("Fixed", stop_iter)

        total_val_steps = 0
        current_acc_results = []
        current_iou_results = []

        # only calculate for NUM_VAL_STEP times of iteration
        for x, y, _ in iter(dataloader):
            if total_val_steps >= NUM_VAL_STEP:
                break
            # thresholding
            k = SIMP_stop_point_sampler()
            x1 = x["input"][:, k, :, :]
            x2 = x["input"][:, k - 1, :, :]
            x = paddle.stack((x1, x1 - x2), axis=1)
            out = paddle.cast(x[:, 0:1, :, :] > 0.5, dtype="float32")
            metric = metric_expr({"output": out}, y)
            current_acc_results.append(np.array(metric["Binary_Acc"]))
            current_iou_results.append(np.array(metric["IoU"]))
            total_val_steps += 1

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
    plt.savefig(os.path.join(OUTPUT_DIR, "Binary_Accuracy.png"))
    plt.show()

    plt.figure(figsize=(12, 6))
    for k, v in iou_results_summary.items():
        plt.plot(iterations_stop_times, v, label=k, lw=4)
    plt.title("IoU", fontsize=16)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("accuracy", fontsize=14)
    plt.legend(loc="best", fontsize=13)
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "IoU.png"))
    plt.show()
