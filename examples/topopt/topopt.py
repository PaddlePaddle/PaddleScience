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

from os import path as osp

import functions as func_module
import h5py
import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from paddle import nn
from topoptmodel import TopOptNN

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # 4 training cases parameters
    LEARNING_RATE = cfg.TRAIN.learning_rate / (1 + cfg.TRAIN.epochs // 15)
    ITERS_PER_EPOCH = int(cfg.n_samples * cfg.train_test_ratio / cfg.TRAIN.batch_size)

    # read h5 data
    h5data = h5py.File(cfg.DATA_PATH, "r")
    data_iters = np.array(h5data["iters"])
    data_targets = np.array(h5data["targets"])

    # generate training dataset
    inputs_train, labels_train = func_module.generate_train_test(
        data_iters, data_targets, cfg.train_test_ratio, cfg.n_samples
    )

    # set constraints
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": inputs_train},
                "label": {"output": labels_train},
                "transforms": (
                    {
                        "FunctionalTransform": {
                            "transform_func": func_module.augmentation,
                        },
                    },
                ),
            },
            "auto_collation": True,
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.FunctionalLoss(loss_wrapper(cfg)),
        name="sup_constraint",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # train models for 4 cases
    for sampler_key, num in cfg.CASE_PARAM:
        # initialize SIMP iteration stop time sampler
        SIMP_stop_point_sampler = func_module.generate_sampler(sampler_key, num)

        # initialize logger for training
        sampler_name = sampler_key + str(num) if num else sampler_key
        OUTPUT_DIR = osp.join(
            cfg.output_dir, f"{sampler_name}_vol_coeff{cfg.vol_coeff}"
        )
        logger.init_logger("ppsci", osp.join(OUTPUT_DIR, "train.log"), "info")

        # set model
        model = TopOptNN(**cfg.MODEL, channel_sampler=SIMP_stop_point_sampler)

        # set optimizer
        optimizer = ppsci.optimizer.Adam(learning_rate=LEARNING_RATE, epsilon=1.0e-7)(
            model
        )

        # initialize solver
        solver = ppsci.solver.Solver(
            model,
            constraint,
            OUTPUT_DIR,
            optimizer,
            epochs=cfg.TRAIN.epochs,
            iters_per_epoch=ITERS_PER_EPOCH,
            eval_during_train=cfg.TRAIN.eval_during_train,
            seed=cfg.seed,
        )

        # train model
        solver.train()


# evaluate 4 models
def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # fixed iteration stop times for evaluation
    iterations_stop_times = range(5, 85, 5)
    model = TopOptNN()

    # evaluation for 4 cases
    acc_results_summary = {}
    iou_results_summary = {}

    # read h5 data
    h5data = h5py.File(cfg.DATA_PATH, "r")
    data_iters = np.array(h5data["iters"])
    data_targets = np.array(h5data["targets"])

    for case_name, model_path in cfg.EVAL.pretrained_model_path_dict.items():
        acc_results, iou_results = evaluate_model(
            cfg, model, model_path, data_iters, data_targets, iterations_stop_times
        )

        acc_results_summary[case_name] = acc_results
        iou_results_summary[case_name] = iou_results

    # calculate thresholding results
    th_acc_results = []
    th_iou_results = []
    for stop_iter in iterations_stop_times:
        SIMP_stop_point_sampler = func_module.generate_sampler("Fixed", stop_iter)

        current_acc_results = []
        current_iou_results = []

        # only calculate for NUM_VAL_STEP times of iteration
        for _ in range(cfg.EVAL.num_val_step):
            input_full_channel, label = func_module.generate_train_test(
                data_iters, data_targets, 1.0, cfg.EVAL.batch_size
            )
            # thresholding
            SIMP_initial_iter_time = SIMP_stop_point_sampler()  # channel k
            input_channel_k = paddle.to_tensor(
                input_full_channel, dtype=paddle.get_default_dtype()
            )[:, SIMP_initial_iter_time, :, :]
            input_channel_k_minus_1 = paddle.to_tensor(
                input_full_channel, dtype=paddle.get_default_dtype()
            )[:, SIMP_initial_iter_time - 1, :, :]
            input = paddle.stack(
                (input_channel_k, input_channel_k - input_channel_k_minus_1), axis=1
            )
            out = paddle.cast(
                paddle.to_tensor(input)[:, 0:1, :, :] > 0.5,
                dtype=paddle.get_default_dtype(),
            )
            th_result = val_metric(
                {"output": out},
                {"output": paddle.to_tensor(label, dtype=paddle.get_default_dtype())},
            )
            acc_results, iou_results = th_result["Binary_Acc"], th_result["IoU"]
            current_acc_results.append(acc_results)
            current_iou_results.append(iou_results)

        th_acc_results.append(np.mean(current_acc_results))
        th_iou_results.append(np.mean(current_iou_results))

    acc_results_summary["thresholding"] = th_acc_results
    iou_results_summary["thresholding"] = th_iou_results

    ppsci.utils.misc.plot_curve(
        acc_results_summary,
        xlabel="iteration",
        ylabel="accuracy",
        output_dir=cfg.output_dir,
    )
    ppsci.utils.misc.plot_curve(
        iou_results_summary, xlabel="iteration", ylabel="iou", output_dir=cfg.output_dir
    )


def evaluate_model(
    cfg, model, pretrained_model_path, data_iters, data_targets, iterations_stop_times
):
    # load model parameters
    solver = ppsci.solver.Solver(
        model,
        epochs=1,
        iters_per_epoch=cfg.EVAL.num_val_step,
        eval_with_no_grad=True,
        pretrained_model_path=pretrained_model_path,
    )

    acc_results = []
    iou_results = []

    # evaluation for different fixed iteration stop times
    for stop_iter in iterations_stop_times:
        # only evaluate for NUM_VAL_STEP times of iteration
        inputs_eval, labels_eval = func_module.generate_train_test(
            data_iters, data_targets, 1.0, cfg.EVAL.batch_size * cfg.EVAL.num_val_step
        )

        sup_validator = ppsci.validate.SupervisedValidator(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {"input": inputs_eval},
                    "label": {"output": labels_eval},
                    "transforms": (
                        {
                            "FunctionalTransform": {
                                "transform_func": func_module.augmentation,
                            },
                        },
                    ),
                },
                "auto_collation": True,
                "batch_size": cfg.EVAL.batch_size,
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": True,
                },
                "num_workers": 0,
            },
            ppsci.loss.FunctionalLoss(loss_wrapper(cfg)),
            {"output": lambda out: out["output"]},
            {"metric": ppsci.metric.FunctionalMetric(val_metric)},
            name="sup_validator",
        )
        validator = {sup_validator.name: sup_validator}
        solver.validator = validator

        # modify the channel_sampler in model
        SIMP_stop_point_sampler = func_module.generate_sampler("Fixed", stop_iter)
        solver.model.channel_sampler = SIMP_stop_point_sampler

        _, eval_result = solver.eval()

        current_acc_results = eval_result["metric"]["Binary_Acc"]
        current_iou_results = eval_result["metric"]["IoU"]

        acc_results.append(current_acc_results)
        iou_results.append(current_iou_results)

    return acc_results, iou_results


# define loss wrapper
def loss_wrapper(cfg: DictConfig):
    def loss_expr(output_dict, label_dict, weight_dict=None):
        label_true = label_dict["output"].reshape((-1, 1))
        label_pred = output_dict["output"].reshape((-1, 1))
        conf_loss = paddle.mean(
            nn.functional.log_loss(label_pred, label_true, epsilon=1.0e-7)
        )
        vol_loss = paddle.square(paddle.mean(label_true - label_pred))
        return conf_loss + cfg.vol_coeff * vol_loss

    return loss_expr


# define metric
def val_metric(output_dict, label_dict, weight_dict=None):
    label_pred = output_dict["output"]
    label_true = label_dict["output"]
    accurates = paddle.equal(paddle.round(label_true), paddle.round(label_pred))
    acc = paddle.mean(paddle.cast(accurates, dtype=paddle.get_default_dtype()))
    true_negative = paddle.sum(
        paddle.multiply(
            paddle.equal(paddle.round(label_pred), 0.0),
            paddle.equal(paddle.round(label_true), 0.0),
        ),
        dtype=paddle.get_default_dtype(),
    )
    true_positive = paddle.sum(
        paddle.multiply(
            paddle.equal(paddle.round(label_pred), 1.0),
            paddle.equal(paddle.round(label_true), 1.0),
        ),
        dtype=paddle.get_default_dtype(),
    )
    false_negative = paddle.sum(
        paddle.multiply(
            paddle.equal(paddle.round(label_pred), 1.0),
            paddle.equal(paddle.round(label_true), 0.0),
        ),
        dtype=paddle.get_default_dtype(),
    )
    false_positive = paddle.sum(
        paddle.multiply(
            paddle.equal(paddle.round(label_pred), 0.0),
            paddle.equal(paddle.round(label_true), 1.0),
        ),
        dtype=paddle.get_default_dtype(),
    )
    n_negative = paddle.add(false_negative, true_negative)
    n_positive = paddle.add(true_positive, false_positive)
    iou = 0.5 * paddle.add(
        paddle.divide(true_negative, paddle.add(n_negative, false_positive)),
        paddle.divide(true_positive, paddle.add(n_positive, false_negative)),
    )
    return {"Binary_Acc": acc, "IoU": iou}


@hydra.main(version_base=None, config_path="./conf", config_name="topopt.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
