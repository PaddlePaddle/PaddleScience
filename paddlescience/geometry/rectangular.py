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
import math


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

    def discretize(self, method="uniform", npoints=10):

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

            nb = int(np.sqrt(npoints - 4 - 4))  # number of internal points
            ni = npoints - 4 * nb - 4  # number of points in each boundary

            # interior
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent, ni))

            # four boundary: down, top, left, right
            origin = [self.origin[0], self.origin[1]]
            extent = [self.extent[0], self.origin[1]]
            steps.append(self._sampling_mesh_interior(origin, extent, nb))

            origin = [self.origin[0], self.extent[1]]
            extent = [self.extent[0], self.extent[1]]
            steps.append(self._sampling_mesh_interior(origin, extent, nb))

            origin = [self.origin[0], self.origin[1]]
            extent = [self.origin[0], self.extent[1]]
            steps.append(self._sampling_mesh_interior(origin, extent, nb))

            origin = [self.extent[0], self.origin[1]]
            extent = [self.extent[0], self.extent[1]]
            steps.append(self._sampling_mesh_interior(origin, extent, nb))

            # four vertex
            steps.append(np.array([self.origin[0], self.origin[1]]))
            steps.append(np.array([self.origin[0], self.extent[1]]))
            steps.append(np.array([self.extent[0], self.origin[1]]))
            steps.append(np.array([self.extent[0], self.extent[1]]))

        elif self.ndims == 3:

            nb = int(np.sqrt(npoints - 4 - 4))  # number of internal points
            ni = npoints - 4 * nb - 4  # number of points in each boundary

            # interior
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent, ni))

            # six faces: down, top, left, right, front, back
            origin = [self.origin[0], self.origin[1], self.origin[2]]
            extent = [self.extent[0], self.extent[1], self.origin[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [self.origin[0], self.origin[1], self.extent[2]]
            extent = [self.extent[0], self.extent[1], self.extent[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [self.origin[0], self.origin[1], self.origin[2]]
            extent = [self.origin[0], self.extent[1], self.extent[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [self.extent[0], self.origin[1], self.origin[2]]
            extent = [self.extent[0], self.extent[1], self.extent[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [self.origin[0], self.origin[1], self.origin[2]]
            extent = [self.origin[0], self.extent[1], self.extent[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            origin = [self.origin[0], self.extent[1], self.origin[2]]
            extent = [self.extent[0], self.extent[1], self.extent[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, nf))

            # twelve edges
            origin = [self.origin[0], self.origin[1], self.origin[2]]
            extent = [self.extent[0], self.origin[1], self.origin[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            origin = [self.extent[0], self.origin[1], self.origin[2]]
            extent = [self.extent[0], self.extent[1], self.origin[2]]
            steps.append(self._sampling_mesh_interior(origin, extent, ne))

            # eight vertex

        else:
            pass
            # TODO: error out

        return np.vstack(steps).reshape(npoints, self.ndims)

    def _sampling_mesh_interior(self, origin, extent, n):

        steps = list()
        for i in range(self.ndims):
            if origin[i] == extent[i]:
                steps.append(np.full(n, origin[i]))
            else:
                # print(np.random.uniform(origin[i], extent[i], n))
                steps.append(np.random.uniform(origin[i], extent[i], n))

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
            nc = int(np.sqrt(npoints))
            nr = npoints - nc

            center = np.array(self.circle_center)
            radius = self.circle_radius

            # rectangular points
            rec = super(CircleInRectangular, self)._sampling_mesh(nr)

            # remove circle points
            flag = np.linalg.norm((rec - center), axis=1) >= radius
            rec_cir = rec[flag, :]

            # add circle boundary points
            angle = np.arange(nc) * (2.0 * np.pi / nc)
            x = np.sin(angle).reshape((nc, 1))
            y = np.cos(angle).reshape((nc, 1))
            cir_b = np.concatenate([x, y], axis=1)
            points = np.vstack([rec_cir, cir_b]).reshape(npoints, self.ndims)

            return super(Rectangular, self)._mesh_to_geo_disc(points)
        else:
            pass
            # TODO: error out uniform method


class CylinderInCube(Rectangular):
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
            nc = int(np.sqrt(npoints))
            nr = npoints - nc

            center = np.array(self.circle_center)
            radius = self.circle_radius

            # cube points
            rec = super(CircleInRectangular, self)._sampling_mesh(nr)

            # remove cylinder points
            flag = np.linalg.norm((rec - center), axis=1) >= radius
            rec_cyl = rec[flag, :]

            # add cylinder boundary points
            angle = np.arange(nc) * (2.0 * np.pi / nc)
            x = np.sin(angle).reshape((nc, 1))
            y = np.cos(angle).reshape((nc, 1))
            z = np.random.uniform(origin[2], extent[2], nz)
            cir_b = np.concatenate([x, y], axis=1)

            # TODO: stack x, y, z

            points = np.vstack([rec_cyl, cyl_b]).reshape(npoints, self.ndims)

            return super(Rectangular, self)._mesh_to_geo_disc(points)
        else:
            pass
            # TODO: error out uniform method
