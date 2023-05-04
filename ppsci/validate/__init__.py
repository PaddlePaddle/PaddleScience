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

from ppsci.loss import build_loss
from ppsci.metric import build_metric
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.validate.base import Validator
from ppsci.validate.geo_validator import GeometryValidator
from ppsci.validate.sup_validator import SupervisedValidator

__all__ = [
    "Validator",
    "GeometryValidator",
    "SupervisedValidator",
]


def build_validator(cfg, equation_dict, geom_dict):
    """Build validator(s).

    Args:
        cfg (List[AttrDict]): Validator(s) config list.
        geom_dict (Dct[str, Geometry]): Geometry(ies) in dict.
        equation_dict (Dct[str, Equation]): Equation(s) in dict.

    Returns:
        Dict[str, Validator]: Validator(s) in dict.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    global_dataloader_cfg = cfg["dataloader"]
    validator_cfg = cfg["content"]

    validator_dict = misc.PrettyOrderedDict()
    for _item in validator_cfg:
        validator_cls = next(iter(_item.keys()))
        _validator_cfg = _item[validator_cls]
        validator_name = _validator_cfg.get("name", validator_cls)
        # select geometry
        geom_name = _validator_cfg.pop("geom")
        _validator_cfg["geom"] = geom_dict[geom_name]

        # update complete dataloader config
        local_dataloader_cfg = _validator_cfg["dataloader"]
        local_dataloader_cfg.update(global_dataloader_cfg)

        # select equation
        for name, expr in _validator_cfg["output_expr"].items():
            if isinstance(expr, str) and expr in equation_dict:
                _validator_cfg["output_expr"][name] = equation_dict[expr].equations[
                    name
                ]

        # build loss
        _validator_cfg["loss"] = build_loss(_validator_cfg["loss"])

        # build metric
        _validator_cfg["metric"] = build_metric(_validator_cfg["metric"])

        # instantiate validator
        _validator_cfg["dataloader_cfg"] = _validator_cfg.pop("dataloader")
        validator_dict[validator_name] = eval(validator_cls)(**_validator_cfg)

        logger.debug(str(validator_dict[validator_name]))

    return validator_dict
