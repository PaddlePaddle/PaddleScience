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

from typing import List

import hydra
from omegaconf import DictConfig
from paddle import nn

import ppsci
from ppsci.arch import base


class EqnTranArch(base.Arch):
    def __init__(self, funcs, input_keys: List, output_keys: List):
        super().__init__()
        if not isinstance(funcs, list):
            funcs = [funcs]
        self.modellist = nn.LayerList(funcs)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, x):
        output_dict = {}
        for i, model in enumerate(self.modellist):
            output_dict[self.output_keys[i]] = model(x)
        return output_dict


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
        use_tbd=cfg.use_tbd,
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


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize equation
    equation = {"VIV": ppsci.equation.Vibration(2, -4, 0)}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        equation=equation,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # Convert equation to Arch
    funcs = ppsci.lambdify(
        solver.equation["VIV"].equations["f"],
        solver.model,
        list(solver.equation["VIV"].learnable_parameters),
    )
    eqn = EqnTranArch(funcs, cfg.INFER.input_keys, ["f"])

    # Combine the two instances
    models = ppsci.arch.ModelList((solver.model, eqn))
    # export models
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]

    from paddle import jit

    jit.enable_to_static(True)

    static_model = jit.to_static(
        models,
        input_spec=input_spec,
        full_graph=True,
    )

    jit.save(static_model, cfg.INFER.export_path, skip_prune_program=True)

    jit.enable_to_static(False)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    # set model predictor
    predictor = pinn_predictor.PINNPredictor(cfg)

    infer_mat = ppsci.utils.reader.load_mat_file(
        cfg.VIV_DATA_PATH,
        ("t_f", "eta_gt", "f_gt"),
        alias_dict={"eta_gt": "eta", "f_gt": "f"},
    )

    input_dict = {key: infer_mat[key] for key in cfg.INFER.input_keys}

    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.INFER.output_keys, output_dict.keys())
    }
    infer_mat.update(output_dict)

    ppsci.visualize.plot.save_plot_from_1d_dict(
        "./viv_pred", infer_mat, ("t_f",), ("eta", "eta_gt", "f", "f_gt")
    )


@hydra.main(version_base=None, config_path="./conf", config_name="viv.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
