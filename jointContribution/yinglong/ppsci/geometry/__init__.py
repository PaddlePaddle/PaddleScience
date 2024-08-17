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

from ppsci.geometry.geometry import Geometry
from ppsci.geometry.geometry_1d import Interval
from ppsci.geometry.geometry_2d import Disk
from ppsci.geometry.geometry_2d import Polygon
from ppsci.geometry.geometry_2d import Rectangle
from ppsci.geometry.geometry_2d import Triangle
from ppsci.geometry.geometry_3d import Cuboid
from ppsci.geometry.geometry_3d import Sphere
from ppsci.geometry.geometry_nd import Hypercube
from ppsci.geometry.geometry_nd import Hypersphere
from ppsci.geometry.mesh import Mesh
from ppsci.geometry.pointcloud import PointCloud
from ppsci.geometry.timedomain import TimeDomain
from ppsci.geometry.timedomain import TimeXGeometry
from ppsci.utils import logger
from ppsci.utils import misc

__all__ = [
    "build_geometry",
    "Cuboid",
    "Disk",
    "Geometry",
    "Hypercube",
    "Hypersphere",
    "Interval",
    "Mesh",
    "Polygon",
    "Rectangle",
    "Sphere",
    "TimeDomain",
    "TimeXGeometry",
    "Triangle",
    "PointCloud",
]


def build_geometry(cfg):
    """Build geometry(ies)

    Args:
        cfg (List[AttrDict]): Geometry config list.

    Returns:
        Dict[str, Geometry]: Geometry(ies) in dict.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    geom_dict = misc.PrettyOrderedDict()
    for _item in cfg:
        geom_cls = next(iter(_item.keys()))
        geom_cfg = _item[geom_cls]
        geom_name = geom_cfg.pop("name", geom_cls)
        if geom_cls == "TimeXGeometry":
            time_cfg = geom_cfg.pop("TimeDomain")
            geom_cls = next(iter(geom_cfg.keys()))
            geom_dict[geom_name] = TimeXGeometry(
                TimeDomain(**time_cfg), eval(geom_cls)(**geom_cfg[geom_cls])
            )
        else:
            geom_dict[geom_name] = eval(geom_cls)(**geom_cfg)

        logger.debug(str(geom_dict[geom_name]))
    return geom_dict
