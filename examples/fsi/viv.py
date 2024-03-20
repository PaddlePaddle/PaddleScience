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

import hydra
from omegaconf import DictConfig

import ppsci


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"VIV": ppsci.equation.Vibration(2, -4, 0)}

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.VIV_DATA_PATH,
            "input_keys": ("t_f",),
            "label_keys": ("eta", "f"),
            "weight_dict": {"eta": 100},
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        # "num_workers": 0,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,) + tuple(equation.values()))

    # set validator
    valid_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.VIV_DATA_PATH,
            "input_keys": ("t_f",),
            "label_keys": ("eta", "f"),
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    eta_mse_validator = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        metric={"MSE": ppsci.metric.MSE()},
        name="eta_mse",
    )
    validator = {eta_mse_validator.name: eta_mse_validator}

    # set visualizer(optional)
    visu_mat = ppsci.utils.reader.load_mat_file(
        cfg.VIV_DATA_PATH,
        ("t_f", "eta_gt", "f_gt"),
        alias_dict={"eta_gt": "eta", "f_gt": "f"},
    )
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerScatter1D(
            visu_mat,
            ("t_f",),
            {
                r"$\eta$": lambda d: d["eta"],  # plot with latex title
                r"$\eta_{gt}$": lambda d: d["eta_gt"],  # plot with latex title
                r"$f$": equation["VIV"].equations["f"],  # plot with latex title
                r"$f_{gt}$": lambda d: d["f_gt"],  # plot with latex title
            },
            num_timestamps=1,
            prefix="viv_pred",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        seed=cfg.seed,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"VIV": ppsci.equation.Vibration(2, -4, 0)}

    # set validator
    valid_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.VIV_DATA_PATH,
            "input_keys": ("t_f",),
            "label_keys": ("eta", "f"),
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    eta_mse_validator = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        metric={"MSE": ppsci.metric.MSE()},
        name="eta_mse",
    )
    validator = {eta_mse_validator.name: eta_mse_validator}

    # set visualizer(optional)
    visu_mat = ppsci.utils.reader.load_mat_file(
        cfg.VIV_DATA_PATH,
        ("t_f", "eta_gt", "f_gt"),
        alias_dict={"eta_gt": "eta", "f_gt": "f"},
    )

    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerScatter1D(
            visu_mat,
            ("t_f",),
            {
                r"$\eta$": lambda d: d["eta"],  # plot with latex title
                r"$\eta_{gt}$": lambda d: d["eta_gt"],  # plot with latex title
                r"$f$": equation["VIV"].equations["f"],  # plot with latex title
                r"$f_{gt}$": lambda d: d["f_gt"],  # plot with latex title
            },
            num_timestamps=1,
            prefix="viv_pred",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )

    # evaluate
    solver.eval()
    # visualize prediction
    solver.visualize()


@hydra.main(version_base=None, config_path="./conf", config_name="viv.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
