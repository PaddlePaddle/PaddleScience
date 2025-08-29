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

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
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
        },
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
    eta_l2_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "MatDataset",
                "file_path": cfg.VIV_DATA_PATH,
                "input_keys": ("t_f",),
                "label_keys": ("eta", "f"),
            },
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        metric={"MSE": ppsci.metric.L2Rel()},
        name="eta_l2",
    )
    validator = {eta_l2_validator.name: eta_l2_validator}

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
        optimizer=optimizer,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
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
    eta_l2_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "MatDataset",
                "file_path": cfg.VIV_DATA_PATH,
                "input_keys": ("t_f",),
                "label_keys": ("eta", "f"),
            },
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        metric={"MSE": ppsci.metric.L2Rel()},
        name="eta_l2",
    )
    validator = {eta_l2_validator.name: eta_l2_validator}

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
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
    )

    # evaluate
    solver.eval()
    # visualize prediction
    solver.visualize()


def export(cfg: DictConfig):
    from paddle import nn
    from paddle.static import InputSpec

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)
    # initialize equation
    equation = {"VIV": ppsci.equation.Vibration(2, -4, 0)}
    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        equation=equation,
        cfg=cfg,
    )
    # Convert equation to func
    f_func = ppsci.lambdify(
        solver.equation["VIV"].equations["f"],
        solver.model,
        list(solver.equation["VIV"].learnable_parameters),
    )

    class Wrapped_Model(nn.Layer):
        def __init__(self, model, func):
            super().__init__()
            self.model = model
            self.func = func

        def forward(self, x):
            x = {**x}
            model_out = self.model(x)
            func_out = self.func(x)
            return {**model_out, "f": func_out}

    solver.model = Wrapped_Model(model, f_func)
    # export models
    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path, skip_prune_program=True)


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
