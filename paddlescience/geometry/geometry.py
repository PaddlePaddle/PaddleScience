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
import pyvista as pv


# Geometry
class Geometry:
    def __init__(self):
        self.criteria = dict()  # criteria (lambda) defining boundary
        self.mesh_file = dict()
        self.normal = dict()  # boundary normal direction
        self._dtype = config._dtype

    def add_boundary(self, name, criteria=None, normal=None, filename=None):
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

        if criteria != None:
            self.criteria[name] = criteria
            self.normal[name] = normal

        if filename != None:
            self.mesh_file[name] = filename

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

        if name in self.mesh_file:
            del self.mesh_file[name]

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
        self.mesh_file.clear()

    def _is_inside_mesh(self, points, filename):
        flag_inside_mesh = np.full(len(points), False, dtype='bool')
        mesh_model = pv.read(filename)
        for i in range(len(points)):
            # TODO The vertice[0,0,0] is not applicable in all cases
            point = [[0, 0, 0], points[i]]
            point_poly = pv.PolyData(point)
            select = point_poly.select_enclosed_points(mesh_model)
            flag_inside_mesh[i] = select['SelectedPoints'][1]
        return flag_inside_mesh

    def _get_points_from_meshfile(self, filename):
        mesh_model = pv.read(filename)
        return mesh_model.points

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

        # boundary points defined by criterial 
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

        # boundary points defined by mesh_file
        for name in self.mesh_file.keys():

            # flag boundary points which inside the mesh
            flag_inside_mesh = self._is_inside_mesh(points,
                                                    self.mesh_file[name])

            # set extracted points as False
            flag_i[flag_inside_mesh] = False

            # add boundary points
            geo_disc.boundary[name] = self._get_points_from_meshfile(
                self.mesh_file[name])

            # TODO: normal

        # extract remain points, i.e. interior points
        geo_disc.interior = points[flag_i, :]

        # padding
        if padding:
            nproc = paddle.distributed.get_world_size()
            geo_disc.padding(nproc)

        return geo_disc
