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

import copy

from ppsci.constraint.base import Constraint
from ppsci.constraint.boundary_constraint import BoundaryConstraint
from ppsci.constraint.initial_constraint import InitialConstraint
from ppsci.constraint.integral_constraint import IntegralConstraint
from ppsci.constraint.interior_constraint import InteriorConstraint
from ppsci.constraint.periodic_constraint import PeriodicConstraint
from ppsci.constraint.supervised_constraint import SupervisedConstraint
from ppsci.loss import build_loss
from ppsci.utils import logger
from ppsci.utils import misc

__all__ = [
    "Constraint",
    "BoundaryConstraint",
    "InitialConstraint",
    "IntegralConstraint",
    "InteriorConstraint",
    "PeriodicConstraint",
    "SupervisedConstraint",
]


def build_constraint(cfg, equation_dict, geom_dict):
    """Build constraint(s).

    Args:
        cfg (List[AttrDict]): Constraint config list.
        equation_dict (Dct[str, Equation]): Equation(s) in dict.
        geom_dict (Dct[str, Geometry]): Geometry(ies) in dict.

    Returns:
        Dict[str, constraint]: Constraint(s) in dict.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    global_dataloader_cfg = cfg["dataloader"]
    constraint_cfg = cfg["content"]

    constraint_dict = misc.PrettyOrderedDict()
    for _item in constraint_cfg:
        constraint_cls = next(iter(_item.keys()))
        _constraint_cfg = _item[constraint_cls]
        constraint_name = _constraint_cfg.get("name", constraint_cls)

        # select equation
        if isinstance(_constraint_cfg["output_expr"], str):
            equation_name = _constraint_cfg.pop("output_expr")
            _constraint_cfg["output_expr"] = equation_dict[equation_name].equations

        # select geometry
        geom_name = _constraint_cfg.pop("geom")
        _constraint_cfg["geom"] = geom_dict[geom_name]

        # update complete dataloader config
        local_dataloader_cfg = _constraint_cfg["dataloader"]
        local_dataloader_cfg.update(global_dataloader_cfg)

        # build loss
        _constraint_cfg["loss"] = build_loss(_constraint_cfg["loss"])

        # instantiate constraint
        _constraint_cfg["dataloader_cfg"] = _constraint_cfg.pop("dataloader")
        constraint_dict[constraint_name] = eval(constraint_cls)(**_constraint_cfg)

        logger.debug(str(constraint_dict[constraint_name]))

    return constraint_dict
