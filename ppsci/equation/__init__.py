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

from ppsci.equation.pde import PDE
from ppsci.equation.pde import Biharmonic
from ppsci.equation.pde import Laplace
from ppsci.equation.pde import NavierStokes
from ppsci.equation.pde import NormalDotVec
from ppsci.equation.pde import Poisson
from ppsci.equation.pde import Vibration
from ppsci.utils import logger
from ppsci.utils import misc

__all__ = [
    "PDE",
    "Biharmonic",
    "Laplace",
    "NavierStokes",
    "NormalDotVec",
    "Poisson",
    "Vibration",
    "build_equation",
]


def build_equation(cfg):
    """Build equation(s)

    Args:
        cfg (List[AttrDict]): Equation(s) config list.

    Returns:
        Dict[str, Equation]: Equation(s) in dict.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    eq_dict = misc.PrettyOrderedDict()
    for _item in cfg:
        eq_cls = next(iter(_item.keys()))
        eq_cfg = _item[eq_cls]
        eq_name = eq_cfg.pop("name", eq_cls)
        eq_dict[eq_name] = eval(eq_cls)(**eq_cfg)

        logger.debug(str(eq_dict[eq_name]))

    return eq_dict
