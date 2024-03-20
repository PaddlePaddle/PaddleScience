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

import importlib.util
import traceback
from typing import Dict
from typing import Sequence
from typing import Union

import paddle

from ppsci.utils import logger

__all__ = [
    "run_check",
    "run_check_mesh",
    "dynamic_import_to_globals",
]


def run_check() -> None:
    """Check whether PaddleScience is installed correctly and running successfully on
    your system.

    Examples:
        >>> import ppsci
        >>> ppsci.utils.run_check()  # doctest: +SKIP
    """
    # test demo code below.
    import ppsci

    try:
        ppsci.utils.set_random_seed(42)
        ppsci.utils.logger.init_logger()
        model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 3, 16, "tanh")

        equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, False)}

        geom = {"rect": ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))}

        ITERS_PER_EPOCH = 5
        train_dataloader_cfg = {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": ITERS_PER_EPOCH,
        }

        NPOINT_PDE = 8**2
        pde_constraint = ppsci.constraint.InteriorConstraint(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["rect"],
            {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            weight_dict={
                "continuity": 0.0001,
                "momentum_x": 0.0001,
                "momentum_y": 0.0001,
            },
            name="EQ",
        )
        constraint = {pde_constraint.name: pde_constraint}

        residual_validator = ppsci.validate.GeometryValidator(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["rect"],
            {
                "dataset": "NamedArrayDataset",
                "total_size": 8**2,
                "batch_size": 32,
                "sampler": {"name": "BatchSampler"},
            },
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            metric={"MSE": ppsci.metric.MSE(False)},
            name="Residual",
        )
        validator = {residual_validator.name: residual_validator}

        EPOCHS = 2
        optimizer = ppsci.optimizer.Adam(0.001)(model)
        solver = ppsci.solver.Solver(
            model,
            constraint,
            None,
            optimizer,
            None,
            EPOCHS,
            ITERS_PER_EPOCH,
            device=paddle.device.get_device(),
            equation=equation,
            validator=validator,
        )
        solver.train()
        solver.eval(EPOCHS)
    except Exception as e:
        traceback.print_exc()
        logger.error(
            f"PaddleScience meets some problem with \n {repr(e)} \nplease check whether "
            "Paddle's version and PaddleScience's version are both correct."
        )
    else:
        logger.message("PaddleScience is installed successfully.✨ 🍰 ✨")


def run_check_mesh() -> None:
    """Check whether geometry packages is installed correctly and `ppsci.geometry.Mesh`
    can running successfully on your system.

    Examples:
        >>> import ppsci
        >>> ppsci.utils.run_check_mesh()  # doctest: +SKIP
    """
    # test demo code below.
    if importlib.util.find_spec("open3d") is None:
        raise ModuleNotFoundError(
            "Please install open3d first as "
            "https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/#143-pip"
        )
    if importlib.util.find_spec("pysdf") is None:
        raise ModuleNotFoundError(
            "Please install pysdf first as "
            "https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/#143-pip"
        )
    if importlib.util.find_spec("pymesh") is None:
        raise ModuleNotFoundError(
            "Please install pymesh first as "
            "https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/#143-pip"
        )

    import numpy as np
    import pymesh

    import ppsci

    try:
        ppsci.utils.set_random_seed(42)
        ppsci.utils.logger.init_logger()
        model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 3, 16, "tanh")

        equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, False)}

        # create a 1x1x1 simple cube geometry
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )  # 8 vertices for mesh
        faces = np.array(
            [
                [4, 7, 5],
                [4, 6, 7],
                [0, 2, 4],
                [2, 6, 4],
                [0, 1, 2],
                [1, 3, 2],
                [1, 5, 7],
                [1, 7, 3],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 1],
                [1, 4, 5],
            ]
        )  # 12 triangle faces for mesh
        box_mesh = pymesh.form_mesh(vertices, faces)
        geom = {"rect": ppsci.geometry.Mesh(box_mesh)}

        ITERS_PER_EPOCH = 5
        train_dataloader_cfg = {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": ITERS_PER_EPOCH,
        }

        NPOINT_PDE = 8**2
        pde_constraint = ppsci.constraint.InteriorConstraint(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["rect"],
            {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
            ppsci.loss.MSELoss("sum"),
            weight_dict={
                "continuity": "sdf",
                "momentum_x": "sdf",
                "momentum_y": "sdf",
            },
            name="EQ",
        )
        constraint = {pde_constraint.name: pde_constraint}

        residual_validator = ppsci.validate.GeometryValidator(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["rect"],
            {
                "dataset": "NamedArrayDataset",
                "total_size": 8**2,
                "batch_size": 32,
                "sampler": {"name": "BatchSampler"},
            },
            ppsci.loss.MSELoss("sum"),
            metric={"MSE": ppsci.metric.MSE(False)},
            name="Residual",
        )
        validator = {residual_validator.name: residual_validator}

        EPOCHS = 2
        optimizer = ppsci.optimizer.Adam(0.001)(model)
        solver = ppsci.solver.Solver(
            model,
            constraint,
            None,
            optimizer,
            None,
            EPOCHS,
            ITERS_PER_EPOCH,
            device=paddle.device.get_device(),
            equation=equation,
            validator=validator,
        )
        solver.train()
        solver.eval(EPOCHS)
    except Exception as e:
        traceback.print_exc()
        logger.error(
            f"PaddleScience meets some problem with \n {repr(e)} \nplease check whether "
            "open3d, pysdf, pybind11, PyMesh are all installed correctly."
        )
    else:
        logger.message("ppsci.geometry.Mesh module running successfully.✨ 🍰 ✨")


def dynamic_import_to_globals(
    names: Union[str, Sequence[str]], alias: Dict[str, str] = None
) -> bool:
    """Import module and add it to globals() by given names dynamically.

    Args:
        names (Union[str, Sequence[str]]): Module name or sequence of module names.
        alias (Dict[str, str]): Alias name of module when imported into globals().

    Returns:
        bool: Whether given names all exist.
    """
    if isinstance(names, str):
        names = (names,)

    if alias is None:
        alias = {}

    for name in names:
        # find module in environment by it's name and alias(if given)
        module_spec = importlib.util.find_spec(name)
        if module_spec is None and name in alias:
            module_spec = importlib.util.find_spec(alias[name])

        # log error and return False if module do not exist
        if not module_spec:
            logger.error(f"Module {name} should be installed first.")
            return False

        # module exist, add to globals() if not in globals()
        add_name = name
        if add_name in alias:
            add_name = alias[add_name]
        if add_name not in globals():
            globals()[add_name] = importlib.import_module(name)

    return True
