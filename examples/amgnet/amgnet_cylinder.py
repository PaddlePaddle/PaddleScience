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

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import utils
from paddle.nn import functional as F

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if TYPE_CHECKING:
    import paddle
    import pgl


def train_mse_func(
    output_dict: Dict[str, "paddle.Tensor"], label_dict: Dict[str, "pgl.Graph"], *args
) -> paddle.Tensor:
    return F.mse_loss(output_dict["pred"], label_dict["label"].y)


def eval_rmse_func(
    output_dict: Dict[str, List["paddle.Tensor"]],
    label_dict: Dict[str, List["pgl.Graph"]],
    *args,
) -> Dict[str, paddle.Tensor]:
    mse_losses = [
        F.mse_loss(pred, label.y)
        for (pred, label) in zip(output_dict["pred"], label_dict["label"])
    ]
    return {"RMSE": (sum(mse_losses) / len(mse_losses)) ** 0.5}


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_AMGNet_cylinder" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set cylinder model
    model = ppsci.arch.AMGNet(
        input_keys=("input",),
        output_keys=("pred",),
        input_dim=4,
        output_dim=3,
        latent_dim=128,
        num_layers=2,
        message_passing_aggregator="sum",
        message_passing_steps=6,
        speed="norm",
    )

    # set dataloader config
    ITERS_PER_EPOCH = 42
    train_dataloader_cfg = {
        "dataset": {
            "name": "MeshCylinderDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": "./data/cylinderdata/train",
            "mesh_graph_path": "./data/cylinderdata/cylinder.su2",
        },
        "batch_size": 4,
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

    # set training hyper-parameters
    EPOCHS = 500 if not args.epochs else args.epochs

    # set optimizer
    optimizer = ppsci.optimizer.Adam(5e-4)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MeshCylinderDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": "./data/cylinderdata/test",
            "mesh_graph_path": "./data/cylinderdata/cylinder.su2",
        },
        "batch_size": 1,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"pred": lambda out: out["pred"]},
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="RMSE_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=50,
        eval_during_train=True,
        eval_freq=50,
        validator=validator,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()

    # visualize prediction
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(rmse_validator.data_loader):
            truefield = label["input"].y
            prefield = model(input_)
            utils.log_images(
                input_["input"].pos,
                prefield["pred"],
                truefield,
                rmse_validator.data_loader.dataset.elems_list,
                index,
                "cylinder",
            )
