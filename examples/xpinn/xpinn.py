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
from typing import Dict

import hydra
import model
import numpy as np
import paddle
import plotting
from omegaconf import DictConfig

import ppsci

# For the use of the second derivative: paddle.cos, paddle.exp
paddle.framework.core.set_prim_eager_enabled(True)


def get_grad(outputs: paddle.Tensor, inputs: paddle.Tensor):
    grad = paddle.grad(outputs, inputs, retain_graph=True, create_graph=True)
    return grad[0]


def xpinn_2d(
    x1: paddle.Tensor,
    y1: paddle.Tensor,
    u1: paddle.Tensor,
    x2: paddle.Tensor,
    y2: paddle.Tensor,
    u2: paddle.Tensor,
    x3: paddle.Tensor,
    y3: paddle.Tensor,
    u3: paddle.Tensor,
    xi1: paddle.Tensor,
    yi1: paddle.Tensor,
    u1i1: paddle.Tensor,
    u2i1: paddle.Tensor,
    xi2: paddle.Tensor,
    yi2: paddle.Tensor,
    u1i2: paddle.Tensor,
    u3i2: paddle.Tensor,
    ub: paddle.Tensor,
    ub_pred: paddle.Tensor,
):
    u1_x = get_grad(u1, x1)
    u1_y = get_grad(u1, y1)
    u1_xx = get_grad(u1_x, x1)
    u1_yy = get_grad(u1_y, y1)

    u2_x = get_grad(u2, x2)
    u2_y = get_grad(u2, y2)
    u2_xx = get_grad(u2_x, x2)
    u2_yy = get_grad(u2_y, y2)

    u3_x = get_grad(u3, x3)
    u3_y = get_grad(u3, y3)
    u3_xx = get_grad(u3_x, x3)
    u3_yy = get_grad(u3_y, y3)

    u1i1_x = get_grad(u1i1, xi1)
    u1i1_y = get_grad(u1i1, yi1)
    u1i1_xx = get_grad(u1i1_x, xi1)
    u1i1_yy = get_grad(u1i1_y, yi1)

    u2i1_x = get_grad(u2i1, xi1)
    u2i1_y = get_grad(u2i1, yi1)
    u2i1_xx = get_grad(u2i1_x, xi1)
    u2i1_yy = get_grad(u2i1_y, yi1)

    u1i2_x = get_grad(u1i2, xi2)
    u1i2_y = get_grad(u1i2, yi2)
    u1i2_xx = get_grad(u1i2_x, xi2)
    u1i2_yy = get_grad(u1i2_y, yi2)

    u3i2_x = get_grad(u3i2, xi2)
    u3i2_y = get_grad(u3i2, yi2)
    u3i2_xx = get_grad(u3i2_x, xi2)
    u3i2_yy = get_grad(u3i2_y, yi2)

    uavgi1 = (u1i1 + u2i1) / 2
    uavgi2 = (u1i2 + u3i2) / 2

    # Residuals
    f1 = u1_xx + u1_yy - (paddle.exp(x1) + paddle.exp(y1))
    f2 = u2_xx + u2_yy - (paddle.exp(x2) + paddle.exp(y2))
    f3 = u3_xx + u3_yy - (paddle.exp(x3) + paddle.exp(y3))

    # Residual continuity conditions on the interfaces
    fi1 = (u1i1_xx + u1i1_yy - (paddle.exp(xi1) + paddle.exp(yi1))) - (
        u2i1_xx + u2i1_yy - (paddle.exp(xi1) + paddle.exp(yi1))
    )
    fi2 = (u1i2_xx + u1i2_yy - (paddle.exp(xi2) + paddle.exp(yi2))) - (
        u3i2_xx + u3i2_yy - (paddle.exp(xi2) + paddle.exp(yi2))
    )

    loss1 = (
        20 * paddle.mean(paddle.square(ub - ub_pred))
        + paddle.mean(paddle.square(f1))
        + 1 * paddle.mean(paddle.square(fi1))
        + 1 * paddle.mean(paddle.square(fi2))
        + 20 * paddle.mean(paddle.square(u1i1 - uavgi1))
        + 20 * paddle.mean(paddle.square(u1i2 - uavgi2))
    )

    loss2 = (
        paddle.mean(paddle.square(f2))
        + 1 * paddle.mean(paddle.square(fi1))
        + 20 * paddle.mean(paddle.square(u2i1 - uavgi1))
    )

    loss3 = (
        paddle.mean(paddle.square(f3))
        + 1 * paddle.mean(paddle.square(fi2))
        + 20 * paddle.mean(paddle.square(u3i2 - uavgi2))
    )
    return loss1, loss2, loss3


def loss_fun(
    output_dict: Dict[str, paddle.Tensor],
    label_dict: Dict[str, paddle.Tensor],
    *args,
):
    loss1, loss2, loss3 = xpinn_2d(
        output_dict["x_f1"],
        output_dict["y_f1"],
        output_dict["u1"],
        output_dict["x_f2"],
        output_dict["y_f2"],
        output_dict["u2"],
        output_dict["x_f3"],
        output_dict["y_f3"],
        output_dict["u3"],
        output_dict["xi1"],
        output_dict["yi1"],
        output_dict["u1i1"],
        output_dict["u2i1"],
        output_dict["xi2"],
        output_dict["yi2"],
        output_dict["u1i2"],
        output_dict["u3i2"],
        label_dict["ub"],
        output_dict["ub_pred"],
    )

    return loss1 + loss2 + loss3


def eval_rmse_func(
    output_dict: Dict[str, paddle.Tensor],
    label_dict: Dict[str, paddle.Tensor],
    *args,
):
    u_pred = paddle.concat([output_dict["u1"], output_dict["u2"], output_dict["u3"]])

    # the shape of label_dict["u_exact"] is [22387, 1], and be cut into [18211, 1] `_eval_by_dataset`(ppsci/solver/eval.py).
    u_exact = paddle.concat(
        [label_dict["u_exact"], label_dict["u_exact2"], label_dict["u_exact3"]]
    )

    error_u_total = paddle.linalg.norm(
        paddle.squeeze(u_exact) - u_pred.flatten(), 2
    ) / paddle.linalg.norm(paddle.squeeze(u_exact), 2)
    return {"total": error_u_total}


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # set training dataset transformation
    def train_dataset_transform_func(
        in_: Dict[str, np.ndarray],
        _label: Dict[str, np.ndarray],
        _weight: Dict[str, np.ndarray],
    ):
        for key in in_:
            in_[key] = paddle.cast(in_[key], paddle.float64)

        # Randomly select the residual points from sub-domains
        id_x1 = np.random.choice(in_["x_f1"].shape[0], cfg.MODEL.num_f1, replace=False)
        in_["x_f1"] = in_["x_f1"][id_x1, :]
        in_["y_f1"] = in_["y_f1"][id_x1, :]

        id_x2 = np.random.choice(in_["x_f2"].shape[0], cfg.MODEL.num_f2, replace=False)
        in_["x_f2"] = in_["x_f2"][id_x2, :]
        in_["y_f2"] = in_["y_f2"][id_x2, :]

        id_x3 = np.random.choice(in_["x_f3"].shape[0], cfg.MODEL.num_f3, replace=False)
        in_["x_f3"] = in_["x_f3"][id_x3, :]
        in_["y_f3"] = in_["y_f3"][id_x3, :]

        # Randomly select boundary points
        id_x4 = np.random.choice(in_["xb"].shape[0], cfg.MODEL.num_ub, replace=False)
        in_["xb"] = in_["xb"][id_x4, :]
        in_["yb"] = in_["yb"][id_x4, :]
        _label["ub"] = _label["ub"][id_x4, :]

        # Randomly select the interface points along two interfaces
        id_xi1 = np.random.choice(in_["xi1"].shape[0], cfg.MODEL.num_i1, replace=False)
        in_["xi1"] = in_["xi1"][id_xi1, :]
        in_["yi1"] = in_["yi1"][id_xi1, :]

        id_xi2 = np.random.choice(in_["xi2"].shape[0], cfg.MODEL.num_i2, replace=False)
        in_["xi2"] = in_["xi2"][id_xi2, :]
        in_["yi2"] = in_["yi2"][id_xi2, :]

        return in_, _label, _weight

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.TRAIN_DATA_DIR,
            "input_keys": (
                "x_f1",
                "y_f1",
                "x_f2",
                "y_f2",
                "x_f3",
                "y_f3",
                "xi1",
                "yi1",
                "xi2",
                "yi2",
                "xb",
                "yb",
            ),
            "label_keys": ("ub",),
            "transforms": (
                {
                    "FunctionalTransform": {
                        "transform_func": train_dataset_transform_func,
                    },
                },
            ),
        }
    }

    layer_list = (
        cfg.MODEL.layers1,
        cfg.MODEL.layers2,
        cfg.MODEL.layers3,
    )

    # set model
    custom_model = model.Model(layer_list)

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.FunctionalLoss(loss_fun),
        {"u1": lambda out: out["u1"]},
        "sup_constraint",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set evaling dataset transformation
    def eval_dataset_transform_func(
        in_: Dict[str, np.ndarray],
        _label: Dict[str, np.ndarray],
        _weight: Dict[str, np.ndarray],
    ):
        for key in in_:
            in_[key] = paddle.cast(in_[key], paddle.float64)
        return in_, _label, _weight

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.TRAIN_DATA_DIR,
            "input_keys": (
                "x_f1",
                "y_f1",
                "x_f2",
                "y_f2",
                "x_f3",
                "y_f3",
                "xi1",
                "yi1",
                "xi2",
                "yi2",
                "xb",
                "yb",
            ),
            "label_keys": ("ub", "u_exact", "u_exact2", "u_exact3"),
            "transforms": (
                {
                    "FunctionalTransform": {
                        "transform_func": eval_dataset_transform_func,
                    },
                },
            ),
        }
    }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(loss_fun),
        output_expr={
            "u1": lambda out: out["u1"],
            "u2": lambda out: out["u2"],
            "u3": lambda out: out["u3"],
        },
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="sup_validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(custom_model)

    # initialize solver
    solver = ppsci.solver.Solver(
        custom_model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    solver.train()
    solver.eval()

    # visualize prediction
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(sup_validator.data_loader):
            u_exact = label["u_exact"]
            output_ = custom_model(input_)
            u_pred = paddle.concat([output_["u1"], output_["u2"], output_["u3"]])

            plotting.log_image(
                x1=input_["x_f1"],
                y1=input_["y_f1"],
                x2=input_["x_f2"],
                y2=input_["y_f2"],
                x3=input_["x_f3"],
                y3=input_["y_f3"],
                xi1=input_["xi1"],
                yi1=input_["yi1"],
                xi2=input_["xi2"],
                yi2=input_["yi2"],
                xb=input_["xb"],
                yb=input_["yb"],
                u_pred=u_pred,
                u_exact=u_exact,
            )


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    layer_list = (
        cfg.MODEL.layers1,
        cfg.MODEL.layers2,
        cfg.MODEL.layers3,
    )

    custom_model = model.Model(layer_list)

    # set evaling dataset transformation
    def eval_dataset_transform_func(
        in_: Dict[str, np.ndarray],
        _label: Dict[str, np.ndarray],
        _weight: Dict[str, np.ndarray],
    ):
        for key in in_:
            in_[key] = paddle.cast(in_[key], paddle.float64)
        return in_, _label, _weight

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.TRAIN_DATA_DIR,
            "input_keys": (
                "x_f1",
                "y_f1",
                "x_f2",
                "y_f2",
                "x_f3",
                "y_f3",
                "xi1",
                "yi1",
                "xi2",
                "yi2",
                "xb",
                "yb",
            ),
            "label_keys": ("ub", "u_exact", "u_exact2", "u_exact3"),
            "transforms": (
                {
                    "FunctionalTransform": {
                        "transform_func": eval_dataset_transform_func,
                    },
                },
            ),
        }
    }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(loss_fun),
        output_expr={
            "u1": lambda out: out["u1"],
            "u2": lambda out: out["u2"],
            "u3": lambda out: out["u3"],
        },
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="sup_validator",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        custom_model,
        output_dir=cfg.output_dir,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    solver.eval()

    # visualize prediction
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(sup_validator.data_loader):
            u_exact = label["u_exact"]
            output_ = custom_model(input_)
            u_pred = paddle.concat([output_["u1"], output_["u2"], output_["u3"]])

            plotting.log_image(
                x1=input_["x_f1"],
                y1=input_["y_f1"],
                x2=input_["x_f2"],
                y2=input_["y_f2"],
                x3=input_["x_f3"],
                y3=input_["y_f3"],
                xi1=input_["xi1"],
                yi1=input_["yi1"],
                xi2=input_["xi2"],
                yi2=input_["yi2"],
                xb=input_["xb"],
                yb=input_["yb"],
                u_pred=u_pred,
                u_exact=u_exact,
            )


@hydra.main(version_base=None, config_path="./conf", config_name="xpinn.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
