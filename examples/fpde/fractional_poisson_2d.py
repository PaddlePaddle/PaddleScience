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

# Reference: https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_2d.py

import math
from typing import Dict
from typing import Tuple
from typing import Union

import hydra
import numpy as np
import paddle
from matplotlib import cm
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci


def plot(x, y, input_data, output_data, label_data):
    fig = plt.figure()
    # plot prediction
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(
        x, y, output_data["u"], cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax1.set_zlim(0, 1.2)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_zlabel(r"$z$")
    ax1.set_title(r"$u(x,y), label$")
    fig.colorbar(surf1, ax=ax1, aspect=5, orientation="horizontal")

    # plot label
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(
        x, y, label_data, cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax2.set_zlim(0, 1.2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title(r"$u(x,y), prediction$")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf2, ax=ax2, aspect=5, orientation="horizontal")
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig("fractional_poisson_2d_result.png", dpi=400)


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    def output_transform(in_, out):
        return {"u": (1 - (in_["x"] ** 2 + in_["y"] ** 2)) * out["u"]}

    model.register_output_transform(output_transform)

    # set geometry
    geom = {"disk": ppsci.geometry.Disk((0, 0), 1)}

    # set equation
    equation = {
        "fpde": ppsci.equation.FractionalPoisson(cfg.ALPHA, geom["disk"], [8, 100])
    }

    # set constraint
    def u_solution_func(
        out: Dict[str, Union[paddle.Tensor, np.ndarray]]
    ) -> Union[paddle.Tensor, np.ndarray]:
        if isinstance(out["x"], paddle.Tensor):
            return paddle.abs(1 - (out["x"] ** 2 + out["y"] ** 2)) ** (
                1 + cfg.ALPHA / 2
            )
        return np.abs(1 - (out["x"] ** 2 + out["y"] ** 2)) ** (1 + cfg.ALPHA / 2)

    # set transform for input data
    def input_data_fpde_transform(
        input: Dict[str, np.ndarray],
        weight: Dict[str, np.ndarray],
        label: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, paddle.Tensor], Dict[str, paddle.Tensor], Dict[str, paddle.Tensor]
    ]:
        """Get sampling points for integral.

        Args:
            input (Dict[str, paddle.Tensor]): Raw input dict.
            weight (Dict[str, paddle.Tensor]): Raw weight dict.
            label (Dict[str, paddle.Tensor]): Raw label dict.

        Returns:
            Tuple[ Dict[str, paddle.Tensor], Dict[str, paddle.Tensor], Dict[str, paddle.Tensor] ]:
                Input dict contained sampling points, weight dict and label dict.
        """
        points = np.concatenate((input["x"].numpy(), input["y"].numpy()), axis=1)
        x = equation["fpde"].get_x(points)
        return (
            {
                **input,
                **{k: paddle.to_tensor(v) for k, v in x.items()},
            },
            weight,
            label,
        )

    fpde_constraint = ppsci.constraint.InteriorConstraint(
        equation["fpde"].equations,
        {"fpde": 0},
        geom["disk"],
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "transforms": (
                    {
                        "FunctionalTransform": {
                            "transform_func": input_data_fpde_transform,
                        },
                    },
                ),
            },
            "batch_size": cfg.NPOINT_INTERIOR,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        random="Hammersley",
        criteria=lambda x, y: ~geom["disk"].on_boundary(np.hstack((x, y))),
        name="FPDE",
    )
    bc = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["disk"],
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": cfg.NPOINT_BC,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        criteria=lambda x, y: np.isclose(x, -1),
        name="BC",
    )
    # wrap constraints together
    constraint = {
        fpde_constraint.name: fpde_constraint,
        bc.name: bc,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    l2rel_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["disk"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": cfg.NPOINT_EVAL,
        },
        ppsci.loss.MSELoss(),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Metric",
    )
    validator = {l2rel_metric.name: l2rel_metric}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        equation=equation,
        validator=validator,
        cfg=cfg,
    )
    # train model
    solver.train()

    # visualize prediction after finished training
    theta = np.arange(0, 2 * math.pi, 0.04, dtype=paddle.get_default_dtype())
    rho = np.arange(0, 1, 0.005, dtype=paddle.get_default_dtype())
    mt, mr = np.meshgrid(theta, rho)
    x = mr * np.cos(mt)
    y = mr * np.sin(mt)

    input_data = {
        "x": x.reshape([-1, 1]),
        "y": y.reshape([-1, 1]),
    }
    label_data = u_solution_func(input_data).reshape([x.shape[0], -1])
    output_data = solver.predict(input_data, return_numpy=True)
    output_data = {k: v.reshape([x.shape[0], -1]) for k, v in output_data.items()}
    plot(x, y, input_data, output_data, label_data)


def evaluate(cfg: DictConfig):
    # load model
    model = ppsci.load_model(cfg.pretrained_model_path)
    # set geometry
    geom = {
        "disk": ppsci.geometry.Disk(np.array([0, 0]), np.array([1]), np.array([[0]])),
    }

    def u_solution_func(
        out: Dict[str, Union[paddle.Tensor, np.ndarray]]
    ) -> Union[paddle.Tensor, np.ndarray]:
        if isinstance(out["x"], paddle.Tensor):
            return paddle.abs(1 - (out["x"] ** 2 + out["y"] ** 2)) ** (
                1 + cfg.ALPHA / 2
            )
        return np.abs(1 - (out["x"] ** 2 + out["y"] ** 2)) ** (1 + cfg.ALPHA / 2)

    # set validator
    l2rel_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["disk"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": cfg.NPOINT_EVAL,
        },
        ppsci.loss.MSELoss(),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Metric",
    )
    validator = {l2rel_metric.name: l2rel_metric}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    # train model
    solver.train()

    # visualize prediction after finished training
    theta = np.arange(0, 2 * math.pi, 0.04, dtype=paddle.get_default_dtype())
    rho = np.arange(0, 1, 0.005, dtype=paddle.get_default_dtype())
    mt, mr = np.meshgrid(theta, rho)
    x = mr * np.cos(mt)
    y = mr * np.sin(mt)

    input_data = {
        "x": x.reshape([-1, 1]),
        "y": y.reshape([-1, 1]),
    }

    label_data = u_solution_func(input_data).reshape([x.shape[0], -1])
    output_data = solver.predict(input_data, return_numpy=True)
    output_data = {k: v.reshape([x.shape[0], -1]) for k, v in output_data.items()}

    plot(x, y, input_data, output_data, label_data)


@hydra.main(
    version_base=None, config_path="./conf", config_name="fractional_poisson_2d.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
