# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from .. import config
from .geometry_discrete import GeometryDiscrete
import numpy as np
import vtk
import matplotlib.pyplot as plt
import paddle


# Geometry
class Geometry:
    def __init__(self):
        self.criteria = dict()  # criteria (lambda) defining boundary
        self.normal = dict()  # boundary normal direction
        self._dtype = config._dtype

    def add_boundary(self, name, criteria, normal=None):
        """
        Add (specify) bounday in geometry

        Parameters:
            name (string): Boundary name
            criteria (lambda function): Lambda function to define boundary.

        Example:
            >>> import paddlescience as psci
            >>> rec = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> rec.add_boundary("top", criteria=lambda x, y : y==1.0) # top boundary
        """

        self.criteria[name] = criteria
        self.normal[name] = normal

    def delete_boundary(self, name):
        """
        Delete bounday in geometry

        Parameters:
            name (string): Boundary name

        Example:
            >>> import paddlescience as psci
            >>> rec = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> rec.add_boundary("top", criteria=lambda x, y : y==1.0) # top boundary
            >>> rec.delete_boundary("top") # delete top boundary
        """

        if name in self.criteria:
            del self.criteria[name]

        if name in self.normal:
            del self.normal[name]

    def clear_boundary(self):
        """
        Delete all the boundaries in geometry

        Example:
            >>> import paddlescience as psci
            >>> rec = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> rec.add_boundary("top", criteria=lambda x, y : y==1.0)  # top boundary
            >>> rec.add_boundary("down", criteria=lambda x, y : y==0.0) # down boundary
            >>> rec.clear_boundary()
        """

        self.criteria.clear()
        self.normal.clear()

    # select boundaries from all points and construct disc geometry
    def _mesh_to_geo_disc(self, points, padding=True):

        geo_disc = GeometryDiscrete()

        npoints = len(points)

        # list of point's columns, used as input of criterial (lambda)
        data = list()
        for n in range(self.ndims):
            data.append(points[:, n])

        # init as True
        flag_i = np.full(npoints, True, dtype='bool')

        # boundary points
        for name in self.criteria.keys():

            # flag bounday points
            flag_b = self.criteria[name](*data)

            # extract
            flag_ib = flag_i & flag_b
            geo_disc.boundary[name] = points[flag_ib, :]

            # set extracted points as False
            flag_i[flag_ib] = False

            # TODO: normal
            normal = self.normal[name]
            normal_disc = None
            geo_disc.normal[name] = normal_disc

        # extract remain points, i.e. interior points
        geo_disc.interior = points[flag_i, :]

        # padding
        if padding:
            nproc = paddle.distributed.get_world_size()
            geo_disc.padding(nproc)

        return geo_disc
