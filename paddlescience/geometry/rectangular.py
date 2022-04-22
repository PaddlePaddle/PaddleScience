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

from .geometry_discrete import GeometryDiscrete
from .geometry import Geometry
import numpy as np
import math


# Rectangular
class Rectangular(Geometry):
    #     """
    #     Two dimentional rectangular

    #     Parameters:
    #         origin: cordinate of left-bottom point of rectangular
    #         extent: extent of rectangular

    #     Example:
    #         >>> import paddlescience as psci
    #         >>> geo = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))

    #     """

    def __init__(self, origin, extent):
        super(Rectangular, self).__init__()

        self.origin = origin
        self.extent = extent
        if len(origin) == len(extent):
            self.ndims = len(origin)
        else:
            pass  # TODO: error out

    def discretize(self, method="uniform", npoints=100):

        # TODO: scalar / list

        if method == "uniform":
            if np.isscalar(npoints):
                # npoints^{1/ndims}
                n = int(math.pow(npoints, 1.0 / self.ndims))
                nl = [n for i in range(self.ndims)]
            else:
                nl = npoints
            points = self._uniform_mesh(nl)
        elif method == "sampling":
            points = self._sampling_mesh(npoints)
            # TODO: npoints as list

        return super(Rectangular, self)._mesh_to_geo_disc(points)

    def _sampling_mesh(self, npoints):

        steps = list()

        if self.ndims == 1:
            pass  # TODO

        elif self.ndims == 2:

            # TODO: npoint should be larger than 9

            ne = int(np.sqrt(npoints - 4 - 4))  # number of points in edge
            ni = npoints - 4 * ne - 4  # number of internal points 

            x1, y1 = self.origin
            x2, y2 = self.extent

            # interior
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent, ni))

            # four boundary: down, top, left, right
            origin = [x1, y1]
            extent = [x2, y1]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y2]
            extent = [x2, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y1]
            extent = [x1, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y1]
            extent = [x2, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            # four vertex
            steps.append(np.array([x1, y1], dtype="float32"))
            steps.append(np.array([x1, y2], dtype="float32"))
            steps.append(np.array([x2, y1], dtype="float32"))
            steps.append(np.array([x2, y2], dtype="float32"))

        elif self.ndims == 3:

            # TODO: exact number of points

            n = int(math.pow(npoints, 1.0 / 3.0))

            nf = n * n  # number of points in face
            ne = n - 2  # number of points in edge
            ni = npoints - 6 * nf - 12 * ne - 8  # number of points internal

            # print(npoints, n, nf, ne, ni)

            x1, y1, z1 = self.origin
            x2, y2, z2 = self.extent

            # interior
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent, ni))

            # six faces: down, top, left, right, front, back
            origin = [x1, y1, z1]
            extent = [x2, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [x1, y1, z2]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [x1, y1, z1]
            extent = [x1, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [x2, y1, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [x1, y1, z1]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [x1, y2, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            # twelve edges
            origin = [x1, y1, z1]
            extent = [x2, y1, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y1, z1]
            extent = [x2, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y2, z1]
            extent = [x1, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y2, z1]
            extent = [x1, y1, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y1, z2]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y1, z2]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y2, z2]
            extent = [x1, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y2, z2]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y1, z1]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y1, z1]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x2, y2, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [x1, y1, z1]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            # eight vertex
            steps.append(np.array([x1, y1, z1], dtype="float32"))
            steps.append(np.array([x2, y1, z1], dtype="float32"))
            steps.append(np.array([x2, y2, z1], dtype="float32"))
            steps.append(np.array([x1, y2, z1], dtype="float32"))
            steps.append(np.array([x1, y1, z2], dtype="float32"))
            steps.append(np.array([x2, y1, z2], dtype="float32"))
            steps.append(np.array([x2, y2, z2], dtype="float32"))
            steps.append(np.array([x1, y2, z2], dtype="float32"))
        else:
            pass
            # TODO: error out

        return np.vstack(steps).reshape(npoints, self.ndims)

    def _sampling_mesh_interior(self, origin, extent, n):

        steps = list()
        for i in range(self.ndims):
            if origin[i] == extent[i]:
                steps.append(np.full(n, origin[i], dtype="float32"))
            else:
                steps.append(
                    np.random.uniform(origin[i], extent[i], n).astype(
                        "float32"))

        return np.dstack(steps).reshape((n, self.ndims))

    def _uniform_mesh(self, npoints):

        steps = list()
        for i in range(self.ndims):
            steps.append(
                np.linspace(
                    self.origin[i],
                    self.extent[i],
                    npoints[i],
                    endpoint=True,
                    dtype='float32'))

        # meshgrid and stack to cordinates
        if (self.ndims == 1):
            points = steps[0]
        if (self.ndims == 2):
            mesh = np.meshgrid(steps[1], steps[0], sparse=False, indexing='ij')
            points = np.stack(
                (mesh[1].reshape(-1), mesh[0].reshape(-1)), axis=-1)
        elif (self.ndims == 3):
            mesh = np.meshgrid(
                steps[2], steps[1], steps[0], sparse=False, indexing='ij')
            points = np.stack(
                (mesh[2].reshape(-1), mesh[1].reshape(-1),
                 mesh[0].reshape(-1)),
                axis=-1)

        return points


# cube 
Cube = Rectangular


# CircleInRectangular
class CircleInRectangular(Rectangular):
    def __init__(self, origin, extent, circle_center, circle_radius):
        super(CircleInRectangular, self).__init__(origin, extent)

        self.origin = origin
        self.extent = extent
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        if len(origin) == len(extent):
            self.ndims = len(origin)
        else:
            pass  # TODO: error out

    def discretize(self, method="sampling", npoints=20):

        if method == "sampling":

            # TODO: better nc and nr
            # TODO: exact nr using area info
            nc = int(np.sqrt(npoints))  # npoints in circle
            nr = npoints - nc  # npoints in rectangular

            center = np.array(self.circle_center, dtype="float32")
            radius = np.array(self.circle_radius, dtype="float32")

            # rectangular points
            rec = super(CircleInRectangular, self)._sampling_mesh(nr)

            # remove circle points
            flag = np.linalg.norm((rec - center), axis=1) >= radius
            rec_cir = rec[flag, :]

            # add circle boundary points
            angle = np.arange(nc) * (2.0 * np.pi / nc)

            # TODO: when circle is larger than rec
            x = (np.sin(angle).reshape((nc, 1)) * radius).astype("float32")
            y = (np.cos(angle).reshape((nc, 1)) * radius).astype("float32")
            cir_b = np.concatenate([x, y], axis=1)
            ncr = len(rec_cir) + len(cir_b)
            points = np.vstack([rec_cir, cir_b]).reshape(ncr, self.ndims)

            return super(CircleInRectangular, self)._mesh_to_geo_disc(points)
        else:
            # TODO: better error out
            print("ERROR: ",
                  type(self).__name__,
                  "does not support uniform discretization.")
            exit()


# CylinderInCube
class CylinderInCube(Rectangular):
    def __init__(self, origin, extent, circle_center, circle_radius):
        super(CylinderInCube, self).__init__(origin, extent)

        self.origin = origin
        self.extent = extent
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        if len(origin) == len(extent):
            self.ndims = len(origin)
        else:
            pass  # TODO: error out

    def discretize(self, method="sampling", npoints=1000):

        if method == "sampling":

            # TODO: better nc and nr
            nc = int(np.sqrt(npoints))
            nr = npoints - nc
            nz = int(math.pow(npoints, 1.0 / 3.0))

            center = np.array(self.circle_center, dtype="float32")
            radius = np.array(self.circle_radius, dtype="float32")

            # cube points
            cube = super(CylinderInCube, self)._sampling_mesh(nr)

            # remove cylinder points
            flag = np.linalg.norm((cube[:, 0:2] - center), axis=1) >= radius
            cube_cyl = cube[flag, :]

            # TODO : points inside / outside cube

            # add cylinder boundary points
            angle = np.arange(nc) * (2.0 * np.pi / nc)
            x = (np.sin(angle).reshape((1, nc)) * radius).astype("float32")
            y = (np.cos(angle).reshape((1, nc)) * radius).astype("float32")
            z = np.random.uniform(self.origin[2], self.extent[2],
                                  nz).astype("float32")
            x_rpt = np.tile(x, nz).reshape((nc * nz, 1))  # repeat x
            y_rpt = np.tile(y, nz).reshape((nc * nz, 1))  # repeat y
            z_rpt = np.repeat(z, nc).reshape((nc * nz, 1))  # repeat z
            cyl_b = np.concatenate([x_rpt, y_rpt, z_rpt], axis=1)  # [x, y, z]

            points = np.vstack([cube_cyl, cyl_b])

            return super(CylinderInCube, self)._mesh_to_geo_disc(points)
        else:
            pass
            # TODO: error out uniform method
