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
from typing import Dict
from typing import Tuple
from typing import Union

from ppsci.utils import logger


def run_check() -> None:
    """Check whether PaddleScience is installed correctly and running successfully on
    your system.

    Examples:
        >>> import ppsci
        >>> ppsci.utils.run_check()
        Runing test code [1/2] [1/5]
        Runing test code [1/2] [2/5]
        Runing test code [1/2] [3/5]
        Runing test code [1/2] [4/5]
        Runing test code [1/2] [5/5]
        Runing test code [2/2] [1/5]
        Runing test code [2/2] [2/5]
        Runing test code [2/2] [3/5]
        Runing test code [2/2] [4/5]
        Runing test code [2/2] [5/5]
        PaddleScience is installed successfully.✨ 🍰 ✨
    """

    # test demo code below.
    import logging

    import ppsci

    try:
        model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 9, 50, "tanh", False, False)

        equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, False)}

        geom = {"rect": ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))}

        iters_per_epoch = 5
        train_dataloader_cfg = {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": iters_per_epoch,
        }

        npoint_pde = 99**2
        pde_constraint = ppsci.constraint.InteriorConstraint(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["rect"],
            {**train_dataloader_cfg, "batch_size": npoint_pde},
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            weight_dict={
                "continuity": 0.0001,
                "momentum_x": 0.0001,
                "momentum_y": 0.0001,
            },
            name="EQ",
        )

        epochs = 2
        optimizer = ppsci.optimizer.Adam(0.001)((model,))
        for _epoch in range(1, epochs + 1):
            for _iter_id in range(1, iters_per_epoch + 1):
                input_dict, label_dict, weight_dict = next(pde_constraint.data_iter)
                for v in input_dict.values():
                    v.stop_gradient = False
                evaluator = ppsci.utils.ExpressionSolver(
                    pde_constraint.input_keys, pde_constraint.output_keys, model
                )
                for output_name, output_formula in pde_constraint.output_expr.items():
                    if output_name in label_dict:
                        evaluator.add_target_expr(output_formula, output_name)

                output_dict = evaluator(input_dict)
                loss = pde_constraint.loss(output_dict, label_dict, weight_dict)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                print(
                    f"Runing test code [{_epoch}/{epochs}]"
                    f" [{_iter_id}/{iters_per_epoch}]"
                )
    except Exception as e:
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
