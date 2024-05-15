# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from os import path as osp

import hydra
from omegaconf import DictConfig

import ppsci


def train(cfg: DictConfig):
    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    inverse_lambda_net = ppsci.arch.MLP(**cfg.MODEL.inverse_lambda_net)
    inverse_mu_net = ppsci.arch.MLP(**cfg.MODEL.inverse_mu_net)
    # freeze models
    disp_net.freeze()
    stress_net.freeze()
    # wrap to a model_list
    model = ppsci.arch.ModelList(
        (disp_net, stress_net, inverse_lambda_net, inverse_mu_net)
    )

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((inverse_lambda_net, inverse_mu_net))

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_="lambda_", mu="mu", dim=3
        )
    }

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set dataloader config
    interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
        },
        geom["geo"],
        {
            "dataset": "NamedArrayDataset",
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
            "batch_size": cfg.TRAIN.batch_size.arm_interior,
        },
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        name="INTERIOR",
    )
    constraint = {interior_constraint.name: interior_constraint}

    # set validator
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))  # 0.5769
    MU = cfg.E / (2 * (1 + cfg.NU))  # 0.3846
    geom_validator = ppsci.validate.GeometryValidator(
        {
            "lambda_": lambda out: out["lambda_"],
            "mu": lambda out: out["mu"],
        },
        {
            "lambda_": LAMBDA_,
            "mu": MU,
        },
        geom["geo"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": cfg.EVAL.total_size.validator,
            "batch_size": cfg.EVAL.batch_size.validator,
        },
        ppsci.loss.MSELoss("sum"),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="geo_eval",
    )
    validator = {geom_validator.name: geom_validator}

    # set visualizer(optional)
    # add inferencer data
    samples = geom["geo"].sample_interior(
        cfg.TRAIN.batch_size.visualizer_vtu,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict = {
        k: v for k, v in samples.items() if k in cfg.MODEL.disp_net.input_keys
    }
    visualizer = {
        "visulzie_lambda_mu": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "lambda": lambda out: out["lambda_"],
                "mu": lambda out: out["mu"],
            },
            prefix="vis",
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

    # plot losses
    solver.plot_loss_history(by_epoch=False, smooth_step=1, use_semilogy=True)


def evaluate(cfg: DictConfig):
    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    inverse_lambda_net = ppsci.arch.MLP(**cfg.MODEL.inverse_lambda_net)
    inverse_mu_net = ppsci.arch.MLP(**cfg.MODEL.inverse_mu_net)
    # wrap to a model_list
    model = ppsci.arch.ModelList(
        (disp_net, stress_net, inverse_lambda_net, inverse_mu_net)
    )

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set validator
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))  # 0.57692
    MU = cfg.E / (2 * (1 + cfg.NU))  # 0.38462
    geom_validator = ppsci.validate.GeometryValidator(
        {
            "lambda_": lambda out: out["lambda_"],
            "mu": lambda out: out["mu"],
        },
        {
            "lambda_": LAMBDA_,
            "mu": MU,
        },
        geom["geo"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": cfg.EVAL.total_size.validator,
            "batch_size": cfg.EVAL.batch_size.validator,
        },
        ppsci.loss.MSELoss("sum"),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="geo_eval",
    )
    validator = {geom_validator.name: geom_validator}

    # set visualizer(optional)
    # add inferencer data
    samples = geom["geo"].sample_interior(
        cfg.EVAL.batch_size.visualizer_vtu,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict = {
        k: v for k, v in samples.items() if k in cfg.MODEL.disp_net.input_keys
    }
    visualizer = {
        "visulzie_lambda_mu": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "lambda": lambda out: out["lambda_"],
                "mu": lambda out: out["mu"],
            },
            prefix="vis",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
    )
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def export(cfg: DictConfig):
    from paddle.static import InputSpec

    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    inverse_lambda_net = ppsci.arch.MLP(**cfg.MODEL.inverse_lambda_net)
    inverse_mu_net = ppsci.arch.MLP(**cfg.MODEL.inverse_mu_net)
    # wrap to a model_list
    model = ppsci.arch.ModelList(
        (disp_net, stress_net, inverse_lambda_net, inverse_mu_net)
    )

    # load pretrained model
    solver = ppsci.solver.Solver(
        model=model, pretrained_model_path=cfg.INFER.pretrained_model_path
    )

    # export models
    input_spec = [
        {
            key: InputSpec([None, 1], "float32", name=key)
            for key in cfg.MODEL.disp_net.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor
    from ppsci.visualize import vtu

    # set model predictor
    predictor = pinn_predictor.PINNPredictor(cfg)

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds
    samples = geom["geo"].sample_interior(
        cfg.EVAL.batch_size.visualizer_vtu,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict = {
        k: v for k, v in samples.items() if k in cfg.MODEL.disp_net.input_keys
    }

    output_dict = predictor.predict(pred_input_dict, cfg.INFER.batch_size)

    # mapping data to output_keys
    output_keys = (
        cfg.MODEL.disp_net.output_keys
        + cfg.MODEL.stress_net.output_keys
        + cfg.MODEL.inverse_lambda_net.output_keys
        + cfg.MODEL.inverse_mu_net.output_keys
    )
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(output_keys, output_dict.keys())
    }
    output_dict.update(pred_input_dict)
    vtu.save_vtu_from_dict(
        osp.join(cfg.output_dir, "vis"),
        output_dict,
        cfg.MODEL.disp_net.input_keys,
        output_keys,
        1,
    )


@hydra.main(
    version_base=None, config_path="./conf", config_name="inverse_parameter.yaml"
)
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
