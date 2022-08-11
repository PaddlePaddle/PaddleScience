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
from scipy.stats import qmc
from pysdf import SDF

__all__ = ['Rectangular', 'Cube', 'CircleInRectangular', 'CylinderInCube']


# Rectangular
class Rectangular(Geometry):
    """
    Two dimentional rectangular or three dimentional cube

    Parameters:
        origin: Cordinate of left-bottom point of rectangular
        extent: Extent of rectangular

    Example:
        >>> import paddlescience as psci
        >>> geo2d = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
        >>> geo3d = psci.geometry.Rectangular(origin=(0.0,0.0,0.0), extent=(2.0,2.0,2.0))
    """

    def __init__(self, origin, extent):
        super(Rectangular, self).__init__()

        if np.isscalar(origin):
            self.origin = [origin]  # scalar to list
        else:
            self.origin = origin

        if np.isscalar(extent):
            self.extent = [extent]  # scalar to list
        else:
            self.extent = extent

        # ndims
        if len(self.origin) == len(self.extent):
            self.ndims = len(self.origin)
        else:
            pass  # TODO: error out

    def discretize(self, method="uniform", npoints=100, padding=True):
        """
        Discretize rectangular

        Parameters:
            method ("uniform" / "sampling" / "quasi_halton" / "quasi_sobol"/ "quasi_lhs"): Discretize rectangular using method "uniform", "sampling", "quasi_halton", "quasi_sobol" or "quasi_lhs".
            npoints (integer / integer list): Number of points 

        Example:
            >>> import paddlescience as psci
            >>> geo = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> geo.discretize(method="uniform", npoints=100)
            >>> geo.discretize(method="uniform", npoints=[10, 20])
            >>> geo.discretize(method="sampling", npoints=200)
            >>> geo.discretize(method="quasi_halton", npoints=200)
            >>> geo.discretize(method="quasi_sobol", npoints=200)
            >>> geo.discretize(method="quasi_lhs", npoints=200)
        """

        if method == "uniform":
            points = self._uniform_mesh(npoints)
        elif method == "sampling":
            points = self._sampling_mesh(npoints)
        elif method == "quasi_halton":
            points = self._sampling_halton(npoints)
        elif method == "quasi_sobol":
            points = self._sampling_sobol(npoints)
        elif method == "quasi_lhs":
            points = self._sampling_lhs(npoints)
        else:
            assert 0, "The discretize method can only be uniform, sampling or quasi sampler."

        return super(Rectangular, self)._mesh_to_geo_disc(points, padding,
                                                          npoints)

    def _sampling_refinement(self, dist, npoints, geo=None):
        # construct the sdf of the geo
        geo = geo.pv_mesh

        if geo.is_manifold is False and geo.is_all_triangles is False:
            assert 0, "The mesh must be watertight and need to be a Triangulate mesh."

        faces_as_array = geo.faces.reshape((geo.n_faces, 4))[:, 1:]

        f = SDF(geo.points, faces_as_array, False)

        points = []
        num_points = 0
        num_iters = 0

        while True:
            # If we get enough points, we stop the loop.
            if num_points >= npoints:
                break

            # Generate enough points
            sampler = qmc.Halton(d=self.ndims, scramble=False)
            sample = sampler.random(n=2 * (num_iters + 1) *
                                    (npoints - num_points))
            l_bounds = self.origin
            u_bounds = self.extent
            result = qmc.scale(sample, l_bounds, u_bounds)
            result = np.array(result).astype(self._dtype)
            sdf_multi_point = f(result)

            # Get the points which meet the requirements
            sdf_flag = (sdf_multi_point < 0) & (sdf_multi_point >= -dist)
            result = result[sdf_flag, :]
            points.append(result)

            # Update the loop message
            num_points += len(result)
            num_iters += 1

        points = np.vstack(points)
        return points[0:npoints, :]

    def _sampling_boundary(self, npoints):
        steps = list()

        if self.ndims == 1:
            steps.append(np.array(self.origin[0], dtype=self._dtype))
            steps.append(np.array(self.extent[0], dtype=self._dtype))
        elif self.ndims == 2:
            # nx: number of points on x-axis
            # ny: number of points on y-axis
            # nx * ny = npoints 
            # nx / ny = lx / ly
            lx = self.extent[0] - self.origin[0]
            ly = self.extent[1] - self.origin[1]
            ny = np.sqrt(float(npoints) * ly / lx)
            nx = float(npoints) / ny
            nx = int(nx)
            ny = int(ny)

            x1, y1 = self.origin
            x2, y2 = self.extent

            # four boundary: down, top, left, right
            origin = [x1, y1]
            extent = [x2, y1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y2]
            extent = [x2, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y1]
            extent = [x1, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x2, y1]
            extent = [x2, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            # four vertex
            steps.append(np.array([x1, y1], dtype=self._dtype))
            steps.append(np.array([x1, y2], dtype=self._dtype))
            steps.append(np.array([x2, y1], dtype=self._dtype))
            steps.append(np.array([x2, y2], dtype=self._dtype))
        elif self.ndims == 3:

            # nx: number of points on x-axis
            # ny: number of points on y-axis
            # nz: number of points on z-axis
            # nx * ny * nz = npoints 
            # nx / lx = ny / ly = nz / lz
            lx = self.extent[0] - self.origin[0]
            ly = self.extent[1] - self.origin[1]
            lz = self.extent[2] - self.origin[2]
            nz = math.pow(float(npoints + 1) * lz**2 / (lx * ly), 1.0 / 3.0)
            nx = nz * lx / lz
            ny = nz * ly / lz
            nx = int(nx)
            ny = int(ny)
            nz = int(nz)

            x1, y1, z1 = self.origin
            x2, y2, z2 = self.extent

            # six faces: down, top, left, right, front, back
            origin = [x1, y1, z1]
            extent = [x2, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * ny))

            origin = [x1, y1, z2]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * ny))

            origin = [x1, y1, z1]
            extent = [x1, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny * nz))

            origin = [x2, y1, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny * nz))

            origin = [x1, y1, z1]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * nz))

            origin = [x1, y2, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * nz))

            # twelve edges
            origin = [x1, y1, z1]
            extent = [x2, y1, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x2, y1, z1]
            extent = [x2, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x2, y2, z1]
            extent = [x1, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y2, z1]
            extent = [x1, y1, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x1, y1, z2]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x2, y1, z2]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x2, y2, z2]
            extent = [x1, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y2, z2]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x1, y1, z1]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            origin = [x2, y1, z1]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            origin = [x2, y2, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            origin = [x1, y1, z1]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            # eight vertex
            steps.append(np.array([x1, y1, z1], dtype=self._dtype))
            steps.append(np.array([x2, y1, z1], dtype=self._dtype))
            steps.append(np.array([x2, y2, z1], dtype=self._dtype))
            steps.append(np.array([x1, y2, z1], dtype=self._dtype))
            steps.append(np.array([x1, y1, z2], dtype=self._dtype))
            steps.append(np.array([x2, y1, z2], dtype=self._dtype))
            steps.append(np.array([x2, y2, z2], dtype=self._dtype))
            steps.append(np.array([x1, y2, z2], dtype=self._dtype))

        return np.vstack(steps)

    def _sampling_halton(self, npoints):

        sampler = qmc.Halton(d=self.ndims, scramble=False)

        sample = sampler.random(n=npoints)

        l_bounds = self.origin
        u_bounds = self.extent

        result = qmc.scale(sample, l_bounds, u_bounds)

        result = np.array(result).astype(self._dtype)

        boundary_points = self._sampling_boundary(npoints)

        return np.concatenate((result, boundary_points), axis=0)

    def _sampling_sobol(self, npoints):

        sampler = qmc.Sobol(d=self.ndims, scramble=False)

        # log_points = np.ceil(np.log2(npoints))
        # sample = sampler.random_base2(m=log_points)
        sample = sampler.random(n=npoints)

        l_bounds = self.origin
        u_bounds = self.extent

        result = qmc.scale(sample, l_bounds, u_bounds)

        result = np.array(result).astype(self._dtype)

        boundary_points = self._sampling_boundary(npoints)

        return np.concatenate((result, boundary_points), axis=0)

    def _sampling_lhs(self, npoints):

        sampler = qmc.LatinHypercube(d=self.ndims)

        sample = sampler.random(n=npoints)

        l_bounds = self.origin
        u_bounds = self.extent

        result = qmc.scale(sample, l_bounds, u_bounds)

        result = np.array(result).astype(self._dtype)

        boundary_points = self._sampling_boundary(npoints)

        return np.concatenate((result, boundary_points), axis=0)

    def _sampling_mesh(self, npoints):

        steps = list()

        if self.ndims == 1:
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent,
                                             npoints))
            steps.append(np.array(self.origin[0], dtype=self._dtype))
            steps.append(np.array(self.extent[0], dtype=self._dtype))

        elif self.ndims == 2:

            # nx: number of points on x-axis
            # ny: number of points on y-axis
            # nx * ny = npoints 
            # nx / ny = lx / ly
            lx = self.extent[0] - self.origin[0]
            ly = self.extent[1] - self.origin[1]
            ny = np.sqrt(float(npoints) * ly / lx)
            nx = float(npoints) / ny
            nx = int(nx)
            ny = int(ny)

            x1, y1 = self.origin
            x2, y2 = self.extent

            # interior
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent,
                                             npoints))

            # four boundary: down, top, left, right
            origin = [x1, y1]
            extent = [x2, y1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y2]
            extent = [x2, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y1]
            extent = [x1, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x2, y1]
            extent = [x2, y2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            # four vertex
            steps.append(np.array([x1, y1], dtype=self._dtype))
            steps.append(np.array([x1, y2], dtype=self._dtype))
            steps.append(np.array([x2, y1], dtype=self._dtype))
            steps.append(np.array([x2, y2], dtype=self._dtype))

        elif self.ndims == 3:

            # nx: number of points on x-axis
            # ny: number of points on y-axis
            # nz: number of points on z-axis
            # nx * ny * nz = npoints 
            # nx / lx = ny / ly = nz / lz
            lx = self.extent[0] - self.origin[0]
            ly = self.extent[1] - self.origin[1]
            lz = self.extent[2] - self.origin[2]
            nz = math.pow(float(npoints + 1) * lz**2 / (lx * ly), 1.0 / 3.0)
            nx = nz * lx / lz
            ny = nz * ly / lz
            nx = int(nx)
            ny = int(ny)
            nz = int(nz)

            x1, y1, z1 = self.origin
            x2, y2, z2 = self.extent

            # interior
            ni = npoints
            steps.append(
                self._sampling_mesh_interior(self.origin, self.extent, ni))

            # six faces: down, top, left, right, front, back
            origin = [x1, y1, z1]
            extent = [x2, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * ny))

            origin = [x1, y1, z2]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * ny))

            origin = [x1, y1, z1]
            extent = [x1, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny * nz))

            origin = [x2, y1, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny * nz))

            origin = [x1, y1, z1]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * nz))

            origin = [x1, y2, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx * nz))

            # twelve edges
            origin = [x1, y1, z1]
            extent = [x2, y1, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x2, y1, z1]
            extent = [x2, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x2, y2, z1]
            extent = [x1, y2, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y2, z1]
            extent = [x1, y1, z1]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x1, y1, z2]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x2, y1, z2]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x2, y2, z2]
            extent = [x1, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nx))

            origin = [x1, y2, z2]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, ny))

            origin = [x1, y1, z1]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            origin = [x2, y1, z1]
            extent = [x2, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            origin = [x2, y2, z1]
            extent = [x2, y2, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            origin = [x1, y1, z1]
            extent = [x1, y1, z2]
            steps.append(self._sampling_mesh_interior(origin, extent, nz))

            # eight vertex
            steps.append(np.array([x1, y1, z1], dtype=self._dtype))
            steps.append(np.array([x2, y1, z1], dtype=self._dtype))
            steps.append(np.array([x2, y2, z1], dtype=self._dtype))
            steps.append(np.array([x1, y2, z1], dtype=self._dtype))
            steps.append(np.array([x1, y1, z2], dtype=self._dtype))
            steps.append(np.array([x2, y1, z2], dtype=self._dtype))
            steps.append(np.array([x2, y2, z2], dtype=self._dtype))
            steps.append(np.array([x1, y2, z2], dtype=self._dtype))
        else:
            pass
            # TODO: error out

        return np.vstack(steps)

    def _sampling_mesh_interior(self, origin, extent, n):

        # return np.random.uniform(low=origin, high=extent, size=(n, self.ndims))

        steps = list()
        for i in range(self.ndims):
            if origin[i] == extent[i]:
                steps.append(np.full(n, origin[i], dtype=self._dtype))
            else:
                steps.append(
                    np.random.uniform(origin[i], extent[i], n).astype(
                        self._dtype))

        return np.dstack(steps).reshape((n, self.ndims))

    def _uniform_mesh(self, npoints, origin=None, extent=None):

        if origin is None:
            origin = self.origin
        if extent is None:
            extent = self.extent

        if np.isscalar(npoints):

            # nx: number of points on x-axis
            # ny: number of points on y-axis
            # nz: number of points on z-axis
            if self.ndims == 1:
                nd = [npoints]
            elif self.ndims == 2:
                # nx * ny = npoints 
                # nx / ny = lx / ly
                lx = self.extent[0] - self.origin[0]
                ly = self.extent[1] - self.origin[1]
                ny = np.sqrt(float(npoints) * ly / lx)
                nx = float(npoints) / ny
                nd = [int(nx), int(ny)]
            elif self.ndims == 3:
                # nx * ny * nz = npoints 
                # nx / lx = ny / ly = nz / lz
                lx = self.extent[0] - self.origin[0]
                ly = self.extent[1] - self.origin[1]
                lz = self.extent[2] - self.origin[2]
                nz = math.pow(
                    float(npoints + 1) * lz**2 / (lx * ly), 1.0 / 3.0)
                nx = nz * lx / lz
                ny = nz * ly / lz
                nd = [int(nx), int(ny), int(nz)]
        else:
            nd = npoints

        steps = list()
        for i in range(self.ndims):
            steps.append(
                np.linspace(
                    origin[i],
                    extent[i],
                    nd[i],
                    endpoint=True,
                    dtype=self._dtype))

        # meshgrid and stack to cordinates
        if (self.ndims == 1):
            points = steps[0].reshape((-1, 1))
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


Cube = Rectangular


# CircleInRectangular
class CircleInRectangular(Rectangular):
    """
    Two dimentional rectangular removing one circle

    Parameters:
        origin (list of float): Cordinate of left-bottom point of rectangular
        extent (list of float): Extent of rectangular
        circle_center (list of float): Center of circle
        circle_radius (float): Radius of circle

    Example:
        >>> import paddlescience as psci
        >>> geo2d = psci.geometry.CircleInRectangular(origin=(0.0,0.0), extent=(1.0,1.0), circle_center=(0.5,0.5), circle_radius=0.1)
   """

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

    def discretize(self, method="sampling", npoints=20, padding=True):
        """
        Discretize CircleInRectangular

        Parameters:
            method (string): Currently, only "sampling" method is supported
            npoints (integer): Number of points

        Example:
            >>> import paddlescience as psci
            >>> geo = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> geo.discretize(method="uniform", npoints=100)
        """

        if method == "sampling":

            # TODO: better nc and nr
            # TODO: exact nr using area info
            nc = int(np.sqrt(npoints))  # npoints in circle
            nr = npoints - nc  # npoints in rectangular

            center = np.array(self.circle_center, dtype=self._dtype)
            radius = np.array(self.circle_radius, dtype=self._dtype)

            # rectangular points
            rec = super(CircleInRectangular, self)._sampling_mesh(nr)

            # remove circle points
            flag = np.linalg.norm((rec - center), axis=1) >= radius
            rec_cir = rec[flag, :]

            # add circle boundary points
            angle = np.arange(nc) * (2.0 * np.pi / nc)

            # TODO: when circle is larger than rec
            x = (np.sin(angle).reshape((nc, 1)) * radius).astype(self._dtype)
            y = (np.cos(angle).reshape((nc, 1)) * radius).astype(self._dtype)
            cir_b = np.concatenate([x, y], axis=1)
            ncr = len(rec_cir) + len(cir_b)
            points = np.vstack([rec_cir, cir_b]).reshape(ncr, self.ndims)

            return super(CircleInRectangular, self)._mesh_to_geo_disc(points,
                                                                      padding)
        else:
            # TODO: better error out
            print("ERROR: ",
                  type(self).__name__,
                  "does not support uniform discretization.")
            exit()


# CylinderInCube
class CylinderInCube(Rectangular):
    """
    Three dimentional cube removing one cylinder

    Parameters:
        origin (list of float): Cordinate of left-bottom point of rectangular
        extent (list of float): Extent of rectangular
        circle_center (list of float): Center of circle
        circle_radius (float): Radius of circle

    Example:
        >>> import paddlescience as psci
        >>> geo2d = psci.geometry.CircleInRectangular(origin=(0.0,0.0), extent=(1.0,1.0), circle_center=(0.5,0.5), circle_radius=0.1)
   """

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

    def discretize(self, method="sampling", npoints=1000, padding=True):
        """
        Discretize CylinderInCube

        Parameters:
            method (string): Currently, "uniform and "sampling" methods are supported
            npoints (integer): Number of points

        Example:
            >>> import paddlescience as psci
            >>> geo = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
            >>> geo.discretize(method="sampling", npoints=100)
        """

        lx = float(self.extent[0] - self.origin[0])
        ly = float(self.extent[1] - self.origin[1])
        lz = float(self.extent[2] - self.origin[2])
        center = np.array(self.circle_center, dtype=self._dtype)
        radius = np.array(self.circle_radius, dtype=self._dtype)
        ratio_area = np.sqrt(3.14 * radius**2 / (lx * ly))
        ratio_perimeter = (3.14 * radius) / (lx + ly)

        if np.isscalar(npoints):
            # nx: number of points on x-axis
            # ny: number of points on y-axis
            # nz: number of points on z-axis
            # nx * ny * nz = npoints 
            # nx / lx = ny / ly = nz / lz
            lx = self.extent[0] - self.origin[0]
            ly = self.extent[1] - self.origin[1]
            lz = self.extent[2] - self.origin[2]
            nz = math.pow(float(npoints + 1) * lz**2 / (lx * ly), 1.0 / 3.0)
            nx = nz * lx / lz
            ny = nz * ly / lz
            nx = int(nx)
            ny = int(ny)
            nz = int(nz)

            # number of points in cube
            ncube = int(npoints * (1.0 + ratio_area)**2)
        else:
            nx = npoints[0]
            ny = npoints[1]
            nz = npoints[2]
            # number of points in cube
            ncube = npoints

        # number of points in circle     
        nc = int(2 * (nx + ny) * ratio_perimeter)

        if method == "sampling":
            # cube points
            cube = super(CylinderInCube, self)._sampling_mesh(ncube)
        elif method == "uniform":
            cube = super(CylinderInCube, self)._uniform_mesh(ncube)

            org = list(self.origin)
            ext = list(self.extent)
            ext[1] = self.origin[1]
            nface = ncube.copy()
            nface[1] = 1
            nface[2] *= 10
            face1 = super(CylinderInCube, self)._uniform_mesh(
                nface, origin=org, extent=ext)

            org = list(self.origin)
            org[1] = self.extent[1]
            ext = list(self.extent)
            nface = ncube.copy()
            nface[1] = 1
            nface[2] *= 10
            face2 = super(CylinderInCube, self)._uniform_mesh(
                nface, origin=org, extent=ext)

            org = list(self.origin)
            ext = list(self.extent)
            ext[0] = self.origin[0]
            nface = ncube.copy()
            nface[0] = 1
            nface[2] *= 10
            face3 = super(CylinderInCube, self)._uniform_mesh(
                nface, origin=org, extent=ext)

            org = list(self.origin)
            org[0] = self.extent[0]
            ext = list(self.extent)
            nface = ncube.copy()
            nface[0] = 1
            nface[2] *= 10
            face4 = super(CylinderInCube, self)._uniform_mesh(
                nface, origin=org, extent=ext)

            cube = np.vstack([cube, face1, face2, face3, face4])

        else:
            pass  # TODO: error out

        # remove cylinder points
        flag = np.linalg.norm((cube[:, 0:2] - center), axis=1) >= radius
        cube_cyl = cube[flag, :]

        # TODO : points inside / outside cube

        # add cylinder boundary points
        angle = np.arange(nc) * (2.0 * np.pi / nc)
        x = (np.sin(angle).reshape((1, nc)) * radius).astype(self._dtype)
        y = (np.cos(angle).reshape((1, nc)) * radius).astype(self._dtype)
        z = np.linspace(self.origin[2], self.extent[2], nz).astype(self._dtype)
        x_rpt = np.tile(x, nz).reshape((nc * nz, 1))  # repeat x
        y_rpt = np.tile(y, nz).reshape((nc * nz, 1))  # repeat y
        z_rpt = np.repeat(z, nc).reshape((nc * nz, 1))  # repeat z
        cyl_b = np.concatenate([x_rpt, y_rpt, z_rpt], axis=1)  # [x, y, z]

        points = np.vstack([cube_cyl, cyl_b])

        return super(CylinderInCube, self)._mesh_to_geo_disc(points, padding)
