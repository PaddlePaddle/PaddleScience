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

# Reference: https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py

from typing import Dict

import numpy as np
import paddle
from matplotlib import pyplot as plt

import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)

    # set output directory
    OUTPUT_DIR = "./output_Volterra_IDE" if not args.output_dir else args.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("x",), ("u",), 3, 20)

    # set geometry
    BOUNDS = (0, 5)
    geom = {"timedomain": ppsci.geometry.TimeDomain(BOUNDS[0], BOUNDS[1])}

    # set equation
    QUAD_DEG = 20
    NPOINT_INTERIOR = 12
    NPOINT_IC = 1

    def kernel_func(x, s):
        return np.exp(s - x)

    def func(out):
        x, u = out["x"], out["u"]
        return jacobian(u, x) + u

    equation = {
        "volterra": ppsci.equation.Volterra(
            BOUNDS[0],
            NPOINT_INTERIOR,
            QUAD_DEG,
            kernel_func,
            func,
        )
    }

    # set constraint
    ITERS_PER_EPOCH = 1
    # set input transform
    def quad_transform(in_: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """Get sampling points for integral.

        Args:
            in_ (Dict[str, paddle.Tensor]): Raw input dict.

        Returns:
            Dict[str, paddle.Tensor]: Input dict contained sampling points.
        """
        x = in_["x"]  # N points.
        x_quad = equation["volterra"].get_quad_points(x).reshape([-1, 1])  # NxQ
        x_quad = paddle.concat((x, x_quad), axis=0)  # M+MxQ: [M|Q1|Q2,...,QM|]
        return {
            **in_,
            "x": x_quad,
        }

    # interior constraint
    ide_constraint = ppsci.constraint.InteriorConstraint(
        equation["volterra"].equations,
        {"volterra": 0},
        geom["timedomain"],
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
        evenly=True,
        name="EQ",
    )

    # initial condition
    def u_solution_func(in_):
        if isinstance(in_["x"], paddle.Tensor):
            return paddle.exp(-in_["x"]) * paddle.cosh(in_["x"])
        return np.exp(-in_["x"]) * np.cosh(in_["x"])

    ic = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["timedomain"],
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": NPOINT_IC,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        criteria=geom["timedomain"].on_initial,
        name="IC",
    )
    # wrap constraints together
    constraint = {
        ide_constraint.name: ide_constraint,
        ic.name: ic,
    }

    # set training hyper-parameters
    EPOCHS = 1 if not args.epochs else args.epochs

    # set optimizer
    optimizer = ppsci.optimizer.LBFGS(
        learning_rate=1,
        max_iter=15000,
        max_eval=1250,
        tolerance_grad=1e-8,
        tolerance_change=0,
        history_size=100,
    )(model)

    # set validator
    NPOINT_EVAL = 100
    l2rel_validator = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["timedomain"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": NPOINT_EVAL,
        },
        ppsci.loss.L2RelLoss(),
        evenly=True,
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Validator",
    )
    validator = {l2rel_validator.name: l2rel_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=1,
        equation=equation,
        geom=geom,
        validator=validator,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()

    # visualize prediction after finished training
    input_data = geom["timedomain"].uniform_points(100)
    label_data = u_solution_func({"x": input_data})
    output_data = solver.predict({"x": input_data}, return_numpy=True)["u"]

    plt.plot(input_data, label_data, "-", label=r"$u(t)$")
    plt.plot(input_data, output_data, "o", label=r"$\hat{u}(t)$", markersize=4.0)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.title(r"$u-t$")
    plt.savefig("./Volterra_IDE.png", dpi=200)
