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

import hydra
import numpy as np
import paddle
from functions import augmentation
from functions import generate_sampler
from functions import generate_train_test
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from paddle import nn
from TopOptModel import TopOptNN

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # 4 training cases parameters
    LEARNING_RATE = 0.001 / (1 + cfg.TRAIN.num_epochs // 15)
    ITERS_PER_EPOCH = int(
        cfg.TRAIN.n_samples * cfg.TRAIN.train_test_ratio / cfg.TRAIN.batch_size
    )

    # generate training dataset
    X_train, Y_train = generate_train_test(
        cfg.DATA_PATH, cfg.TRAIN.train_test_ratio, cfg.TRAIN.n_samples
    )

    # define loss
    def loss_expr(output_dict, label_dict, weight_dict=None):
        y = label_dict["output"].reshape((-1, 1))
        y_pred = output_dict["output"].reshape((-1, 1))
        conf_loss = paddle.mean(
            nn.functional.log_loss(y_pred, y, epsilon=cfg.TRAIN.epsilon.log_loss)
        )
        vol_loss = paddle.square(paddle.mean(y - y_pred))
        return conf_loss + cfg.TRAIN.vol_coeff * vol_loss

    # set constraints
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": X_train},
                "label": {"output": Y_train},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": cfg.TRAIN.drop_last,
                "shuffle": cfg.TRAIN.shuffle,
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
        name="sup_constraint",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # train models for 4 cases
    for sampler_key, num in cfg.CASE_PARAM:

        # initialize SIMP iteration stop time sampler
        SIMP_stop_point_sampler = generate_sampler(sampler_key, num)

        # initialize logger for training
        OUTPUT_DIR = cfg.output_dir
        OUTPUT_DIR = (
            f"{OUTPUT_DIR}/{sampler_key}{num}_vol_coeff{cfg.TRAIN.vol_coeff}"
            if num is not None
            else f"{OUTPUT_DIR}/{sampler_key}_vol_coeff{cfg.TRAIN.vol_coeff}"
        )
        logger.init_logger("ppsci", os.path.join(OUTPUT_DIR, "train.log"), "info")

        # set model
        model = TopOptNN(
            in_channel=cfg.MODEL.in_channel,
            out_channel=cfg.MODEL.out_channel,
            kernel_size=cfg.MODEL.kernel_size,
            filters=cfg.MODEL.filters,
            layers=cfg.MODEL.layers,
            channel_sampler=SIMP_stop_point_sampler,
        )
        assert model.num_params == cfg.TRAIN.num_params

        # set optimizer
        optimizer = ppsci.optimizer.Adam(
            learning_rate=LEARNING_RATE, epsilon=cfg.TRAIN.epsilon.optimizer
        )(model)

        # initialize solver
        solver = ppsci.solver.Solver(
            model,
            constraint,
            OUTPUT_DIR,
            optimizer,
            epochs=cfg.TRAIN.num_epochs,
            iters_per_epoch=ITERS_PER_EPOCH,
        )

        # train model
        solver.train()


def evaluate_and_plot(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # evaluate 4 models
    RESULT_DIR = os.path.join(cfg.output_dir, "results")

    # initialize logger for evaluation
    logger.init_logger("ppsci", os.path.join(RESULT_DIR, "results.log"), "info")

    # fixed iteration stop times for evaluation
    iterations_stop_times = range(5, 85, 5)
    model = TopOptNN()

    # define loss
    def loss_expr(output_dict, label_dict, weight_dict=None):
        y = label_dict["output"].reshape((-1, 1))
        y_pred = output_dict["output"].reshape((-1, 1))
        conf_loss = paddle.mean(
            nn.functional.log_loss(y_pred, y, epsilon=cfg.TRAIN.epsilon.log_loss)
        )
        vol_loss = paddle.square(paddle.mean(y - y_pred))
        return conf_loss + cfg.TRAIN.vol_coeff * vol_loss

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

    # evaluation for 4 cases
    acc_results_summary = {}
    iou_results_summary = {}
    for model_name in cfg.EVAL.model_list:

        # load model parameters
        model_path = f"{cfg.output_dir}/{model_name}_vol_coeff{cfg.TRAIN.vol_coeff}/checkpoints/latest"
        solver = ppsci.solver.Solver(model, pretrained_model_path=model_path)
        solver.epochs = 1
        solver.iters_per_epoch = cfg.EVAL.num_val_step
        acc_results = []
        iou_results = []

        # evaluation for different fixed iteration stop times
        for stop_iter in iterations_stop_times:
            # only evaluate for NUM_VAL_STEP times of iteration
            X_data, Y_data = generate_train_test(
                cfg.DATA_PATH, 1, cfg.EVAL.batch_size * cfg.EVAL.num_val_step
            )
            sup_validator = ppsci.validate.SupervisedValidator(
                {
                    "dataset": {
                        "name": "NamedArrayDataset",
                        "input": {"input": X_data},
                        "label": {"output": Y_data},
                    },
                    "batch_size": cfg.EVAL.batch_size,
                    "sampler": {
                        "name": "BatchSampler",
                        "drop_last": cfg.EVAL.drop_last,
                        "shuffle": cfg.EVAL.shuffle,
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
            x, y = generate_train_test(cfg.DATA_PATH, 1, cfg.EVAL.batch_size)
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


@hydra.main(version_base=None, config_path="./conf", config_name="topopt.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate_and_plot(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
