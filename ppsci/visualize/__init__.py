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

from ppsci.visualize.base import Visualizer
from ppsci.visualize.visualizer import Visualizer3D
from ppsci.visualize.visualizer import VisualizerScatter1D
from ppsci.visualize.visualizer import VisualizerVtu
from ppsci.visualize.vtu import save_vtu_from_dict

__all__ = [
    "Visualizer",
    "VisualizerScatter1D",
    "VisualizerVtu",
    "Visualizer3D",
    "save_vtu_from_dict",
]


def build_visualizer(cfg):
    """Build visualizer(s).

    Args:
        cfg (List[AttrDict]): Visualizer(s) config list.
        geom_dict (Dct[str, Geometry]): Geometry(ies) in dict.
        equation_dict (Dct[str, Equation]): Equation(s) in dict.

    Returns:
        Dict[str, Visualizer]: Visualizer(s) in dict.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    visualizer_dict = {}
    for _item in cfg:
        visualizer_cls = next(iter(_item.keys()))
        visualizer_cfg = _item[visualizer_cls]
        visualizer = eval(visualizer_cls)(**visualizer_cfg)

        visualizer_name = visualizer_cfg.get("name", visualizer_cls)
        if visualizer_name in visualizer_dict:
            raise ValueError(f"Name of visualizer({visualizer_name}) should be unique")
        visualizer_dict[visualizer_name] = visualizer

    return visualizer_dict
