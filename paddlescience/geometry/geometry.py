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
from pysdf import SDF


# Geometry
class Geometry:
    def __init__(self):
        self.criteria = dict()  # criteria (lambda) defining boundary
        self.tri_mesh = dict()
        self.normal = dict()  # boundary normal direction
        self.pv_mesh = None
        self._dtype = config._dtype

    def add_boundary(self, name, criteria=None, normal=None, filename=None):
        """
        Add (specify) bounday in geometry

        Parameters:
            name (string): Boundary name
            criteria (lambda function): Lambda function to define boundary.
            filename (string): Read mesh file to define boundary. The mesh file needs to meet two conditions: 1. It must be watertight. 2. It must be a `Triangular Mesh`.

        Example:
            >>> import paddlescience as psci
            >>> rec = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> rec.add_boundary("top", criteria=lambda x, y : y==1.0) # top boundary
            >>> rec.add_boundary("geo_boundary", filename="geo_boundary.stl")
        """

        if criteria != None:
            self.criteria[name] = criteria
            self.normal[name] = normal

        if filename != None:
            self.tri_mesh[name] = filename

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

        if name in self.tri_mesh:
            del self.tri_mesh[name]

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
        self.tri_mesh.clear()

    def _is_inside_mesh(self, points, tri_mesh):

        if isinstance(tri_mesh, str):
            mesh_model = pv.read(tri_mesh)
        else:
            mesh_model = tri_mesh.pv_mesh

        # The mesh must be manifold and need to be triangulate
        if mesh_model.is_manifold is False and mesh_model.is_all_triangles is False:
            assert 0, "The mesh must be watertight and need to be Triangulate mesh."

        # The all the faces of mesh must be triangles
        faces_as_array = mesh_model.faces.reshape(
            (mesh_model.n_faces, 4))[:, 1:]

        sdf = SDF(mesh_model.points, faces_as_array, False)

        origin_contained = sdf.contains(points)

        return origin_contained

    def _get_points_from_meshfile(self, tri_mesh):

        if isinstance(tri_mesh, str):
            mesh_model = pv.read(tri_mesh)
        else:
            mesh_model = tri_mesh.pv_mesh

        # TODO(liu-xiandong): Need to increase sampling points on the boundary
        return mesh_model.points

    def __sub__(self, other):
        self.tri_mesh['subtraction' + str(len(self.tri_mesh))] = other
        return self

    # select boundaries from all points and construct disc geometry
    def _mesh_to_geo_disc(self, points, padding=True, npoints_need=100):

        geo_disc = GeometryDiscrete(geometry=self)

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
            normal_disc = normal
            geo_disc.normal[name] = normal_disc

        # boundary points defined by mesh_file
        for name in self.tri_mesh.keys():

            # flag boundary points which inside the mesh
            flag_inside_mesh = self._is_inside_mesh(points,
                                                    self.tri_mesh[name])

            # set extracted points as False
            flag_i[flag_inside_mesh] = False

            # add boundary points
            geo_disc.boundary[name] = self._get_points_from_meshfile(
                self.tri_mesh[name])

            # TODO: normal

        # extract remain points, i.e. interior points
        geo_disc.interior = points[flag_i, :]

        # TODO: Note that the currently generated points are inaccurate 
        # and will be fixed in the future

        # padding
        if padding:
            nproc = paddle.distributed.get_world_size()
            geo_disc.padding(nproc)

        return geo_disc

    def _sampling_refinement(self, dist, npoints, geo=None):
        pass
