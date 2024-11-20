# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import equation as eq_func
import hydra
from omegaconf import DictConfig

import ppsci
from ppsci.utils import reader


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set equation
    equation = {"Hooke": eq_func.Hooke(E=cfg.E, nu=cfg.nu, P=cfg.P, dim=3)}

    # set geometry
    heart = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    base = ppsci.geometry.Mesh(cfg.BASE_PATH)
    endo = ppsci.geometry.Mesh(cfg.ENDO_PATH)
    epi = ppsci.geometry.Mesh(cfg.EPI_PATH)
    geom = {"geo": heart, "base": base, "endo": endo, "epi": epi}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = heart.bounds

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    bc_base = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["base"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_base},
        ppsci.loss.MSELoss("mean"),
        weight_dict=cfg.TRAIN.weight.bc_base,
        name="BC_BASE",
    )
    bc_endo = ppsci.constraint.BoundaryConstraint(
        equation["Hooke"].equations,
        {"traction": -cfg.P},
        geom["endo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_endo},
        ppsci.loss.MSELoss("mean"),
        weight_dict=cfg.TRAIN.weight.bc_endo,
        name="BC_ENDO",
    )
    bc_epi = ppsci.constraint.BoundaryConstraint(
        equation["Hooke"].equations,
        {"traction": 0},
        geom["epi"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_epi},
        ppsci.loss.MSELoss("mean"),
        weight_dict=cfg.TRAIN.weight.bc_epi,
        name="BC_EPI",
    )
    interior = ppsci.constraint.InteriorConstraint(
        equation["Hooke"].equations,
        {"hooke_x": 0, "hooke_y": 0, "hooke_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.interior},
        ppsci.loss.MSELoss("mean"),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        weight_dict=cfg.TRAIN.weight.interior,
        name="INTERIOR",
    )
    data = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": cfg.DATA_CSV_PATH,
                "input_keys": ("x", "y", "z"),
                "label_keys": ("u", "v", "w"),
            },
        },
        ppsci.loss.MSELoss("sum"),
        name="DATA",
    )

    # wrap constraints together
    constraint = {
        bc_base.name: bc_base,
        bc_endo.name: bc_endo,
        bc_epi.name: bc_epi,
        interior.name: interior,
        data.name: data,
    }

    # set validator
    eval_data_dict = reader.load_csv_file(
        cfg.EVAL_CSV_PATH,
        ("x", "y", "z", "u", "v", "w"),
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "u": "u",
            "v": "v",
            "w": "w",
        },
    )

    input_dict = {
        "x": eval_data_dict["x"],
        "y": eval_data_dict["y"],
        "z": eval_data_dict["z"],
    }

    label_dict = {
        "u": eval_data_dict["u"],
        "v": eval_data_dict["v"],
        "w": eval_data_dict["w"],
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
        },
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size},
        ppsci.loss.MSELoss("mean"),
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="ref_u_v_w",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visualize_u_v_w": ppsci.visualize.VisualizerVtu(
            input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            batch_size=cfg.EVAL.batch_size,
            prefix="result_u_v_w",
        ),
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
    )

    # train
    solver.train()
    # eval
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()
    # plot loss
    solver.plot_loss_history(by_epoch=True)


def evaluate(cfg: DictConfig):
    # set models
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set validator
    eval_data_dict = reader.load_csv_file(
        cfg.EVAL_CSV_PATH,
        ("x", "y", "z", "u", "v", "w"),
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "u": "u",
            "v": "v",
            "w": "w",
        },
    )

    input_dict = {
        "x": eval_data_dict["x"],
        "y": eval_data_dict["y"],
        "z": eval_data_dict["z"],
    }

    label_dict = {
        "u": eval_data_dict["u"],
        "v": eval_data_dict["v"],
        "w": eval_data_dict["w"],
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
        },
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size},
        ppsci.loss.MSELoss("mean"),
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="ref_u_v_w",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer
    visualizer = {
        "visualize_u_v_w": ppsci.visualize.VisualizerVtu(
            input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            batch_size=cfg.EVAL.batch_size,
            prefix="result_u_v_w",
        ),
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
    )

    # evaluate
    solver.eval()
    # visualize prediction
    solver.visualize()


@hydra.main(version_base=None, config_path="./conf", config_name="forward.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
