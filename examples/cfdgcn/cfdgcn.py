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
from typing import Dict
from typing import List

import hydra
import paddle
import pgl
import su2paddle
from omegaconf import DictConfig
from paddle.nn import functional as F

import ppsci
from ppsci.utils import logger


def train_mse_func(
    output_dict: Dict[str, "paddle.Tensor"],
    label_dict: Dict[str, "pgl.Graph"],
    *args,
) -> paddle.Tensor:
    y = paddle.stack([g.y for g in label_dict["label"]])
    return F.mse_loss(output_dict["pred"], y)


def eval_rmse_func(
    output_dict: Dict[str, List["paddle.Tensor"]],
    label_dict: Dict[str, List["pgl.Graph"]],
    *args,
) -> Dict[str, paddle.Tensor]:
    mse_losses = [
        F.mse_loss(pred, paddle.stack([g.y for g in label]))
        for (pred, label) in zip(output_dict["pred"], label_dict["label"])
    ]
    return {"RMSE": (sum(mse_losses) / len(mse_losses)) ** 0.5}


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", os.path.join(cfg.output_dir, "train.log"), "info")

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.TRAIN_DATA_DIR,
            "mesh_graph_path": cfg.TRAIN_MESH_GRAPH_PATH,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    process_sim = sup_constraint.dataset._preprocess
    fine_marker_dict = sup_constraint.dataset.marker_dict
    # out_channels=sup_constraint.dataset[0][0][cfg.MODEL.input_keys[0]].y.shape[-1]

    # set airfoil model
    model = ppsci.arch.CFDGCN(
        **cfg.MODEL,
        process_sim=process_sim,
        fine_marker_dict=fine_marker_dict,
        su2_module=su2paddle.SU2Module,
    )

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.EVAL_DATA_DIR,
            "mesh_graph_path": cfg.EVAL_MESH_GRAPH_PATH,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"pred": lambda out: out["pred"].unsqueeze(0)},
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="RMSE_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
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
    )

    # train model
    solver.train()


# def evaluate(cfg: DictConfig):


@hydra.main(version_base=None, config_path="./conf", config_name="cfdgcn.yaml")
def main(cfg: DictConfig):
    su2paddle.activate_su2_mpi(remove_temp_files=True)
    if cfg.mode == "train":
        train(cfg)
    # elif cfg.mode == "eval":
    #     evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
