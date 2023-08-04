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

from typing import Dict
from typing import Union

import numpy as np
import paddle

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set training hyper-parameters
    EPOCHS = 20000 if not args.epochs else args.epochs
    ITERS_PER_EPOCH = 1

    # set output directory
    OUTPUT_DIR = "./output/Volterra_IDE" if not args.output_dir else args.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("x", "y"), ("u",), 4, 20)
    model.register_output_transform(
        lambda in_, out: (1 - (in_["x"] ** 2 + in_["y"] ** 2)) * out["u"]
    )

    # set geometry
    geom = {"disk": ppsci.geometry.Disk((0, 0), 1)}

    # set equation
    ALPHA = 1.8

    def kernel_func(x, s):
        return np.exp(s - x)

    equation = {"fpde": ppsci.equation.FPDE(ALPHA, geom["disk"], [8, 100])}

    # set constraint
    NPOINT_INTERIOR = 100
    NPOINT_BC = 1

    def u_solution_func(
        out: Dict[str, Union[paddle.Tensor, np.ndarray]]
    ) -> Union[paddle.Tensor, np.ndarray]:
        if isinstance(out["x"], paddle.Tensor):
            return (paddle.abs(1 - out["x"] ** 2 + out["y"] ** 2)) ** (1 + ALPHA / 2)
        return (np.abs(1 - out["x"] ** 2 + out["y"] ** 2)) ** (1 + ALPHA / 2)

    # set input transform
    def quad_transform(in_: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get sampling points for integral.

        Args:
            in_ (Dict[str, np.ndarray]): Raw input dict.

        Returns:
            Dict[str, np.ndarray]: Input dict contained sampling points.
        """
        x = in_["x"]  # N points.
        x = equation["fpde"].get_x(x)  # NxQ
        return {
            **in_,
            "x": x,
        }

    fpde_constraint = ppsci.constraint.InteriorConstraint(
        {},
        {"fpde": u_solution_func},
        geom["disk"],
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "transforms": (
                    {
                        "FunctionalTransform": {
                            "transform_func": quad_transform,
                        },
                    },
                ),
            },
            "batch_size": NPOINT_INTERIOR,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        criteria=lambda x, y: ~geom["disk"].on_boundary(np.hstack((x, y))),
        name="FPDE",
    )
    bc = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["disk"],
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": NPOINT_BC,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        name="BC",
    )
    # wrap constraints together
    constraint = {
        fpde_constraint.name: fpde_constraint,
        bc.name: bc,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(1e-3)(model)

    # set validator
    NPOINT_EVAL = 100
    l2rel_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": lambda in_: np.exp(-in_["x"]) * np.cosh(in_["x"])},
        geom["disk"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": NPOINT_EVAL,
        },
        ppsci.loss.L2RelLoss(),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Metric",
    )
    validator = {l2rel_metric.name: l2rel_metric}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=400,
        equation=equation,
        geom=geom,
        validator=validator,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()

    # visualize prediction after finished training
    # input_data = geom["disk"].sample_interior(1000)
    # input_data = {k: v for k, v in input_data.items() if k in geom["disk"].dim_keys}

    # label_data = u_solution_func(input_data)
    # output_data = solver.predict(input_data)
    # output_data = {
    #     k: v.numpy()
    #     for k, v in output_data.items()
    # }

    # plt.plot(input_data, label_data, "-", label=r"$u(t)$")
    # plt.plot(input_data, output_data, "o", label="pred", markersize=4.0)
    # plt.legend()
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$u$")
    # plt.title(r"$u-t$")
    # plt.savefig("./Volterra_IDE.png", dpi=200)
