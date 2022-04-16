# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR boundaryS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .geometry_discrete import GeometryDiscrete
from .geometry import Geometry
import numpy as np


class CylinderInRectangular(Rectangular):
    def __init__(self, origin, extent, circle_center, circle_radius):
        super(CylinderInRectangular, self).__init__()

        self.origin = origin
        self.extent = extent
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        if len(origin) == len(extent):
            self.ndims = len(origin)
        else:
            pass  # TODO: error out

    def discretize(self, method="sampling", npoints=10):

        if method == "sampling":

            nc = 1

            center = self.circle_center
            radius = self.circle_radius

            # points in rectangular
            rec = super(CylinderInRectangular, self)._sampling_mesh(npoints)

            # remove disk points
            flag = np.norm((rec_points - center), axis=0) >= radius
            rec_disk = rec[flag, :]

            # add circle points
            angle = np.arange(nc) * (2.0 * np.pi / nc)
            circle = np.concatenate([np.sin(angle), np.cos(angle)], axix=0)

            points = np.vstack([rec_disk, circle]).reshape(npoints, self.ndims)

            return super(Rectangular, self)._mesh_to_geo_disc(points)
        else:
            pass
            # TODO: error out uniform method
