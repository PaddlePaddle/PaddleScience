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

import importlib.util
import traceback
from typing import Dict
from typing import Tuple
from typing import Union

import paddle

from ppsci.utils import logger


def run_check() -> None:
    """Check whether PaddleScience is installed correctly and running successfully on
    your system.

    Examples:
        >>> import ppsci
        >>> ppsci.utils.run_check()  # doctest: +SKIP
    """
    # test demo code below.
    import logging

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
        optimizer = ppsci.optimizer.Adam(0.001)((model,))
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
        logging.warning(
            f"PaddleScience meets some problem with \n {repr(e)} \nplease check whether "
            "Paddle's version and PaddleScience's version are both correct."
        )
    else:
        print("PaddleScience is installed successfully.✨ 🍰 ✨")


def dynamic_import_to_globals(
    names: Union[str, Tuple[str, ...]], alias: Dict[str, str] = None
) -> bool:
    """Import module and add it to globals() by given names dynamically.

    Args:
        names (Union[str, Tuple[str, ...]]): Module name or list of module names.
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
