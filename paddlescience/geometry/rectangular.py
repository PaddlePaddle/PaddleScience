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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .geometry_discrete import GeometryDiscrete
from .geometry import Geometry
import numpy as np
import vtk
import matplotlib.pyplot as plt


# Rectangular
class Rectangular(Geometry):
    """
    Two dimentional rectangular

    Parameters:
        space_origin: cordinate of left-bottom point of rectangular
        space_extent: extent of rectangular

    Example:
        >>> import paddlescience as psci
        >>> geo = psci.geometry.Rectangular(space_origin=(0.0,0.0), space_extent=(1.0,1.0))

    """

    # init function
    def __init__(self,
                 time_dependent=False,
                 time_origin=None,
                 time_extent=None,
                 space_origin=None,
                 space_extent=None):
        super(Rectangular, self).__init__(time_dependent, time_origin,
                                          time_extent, space_origin,
                                          space_extent)

        # check time inputs 
        if (time_dependent == True):
            if (time_origin == None or not np.isscalar(time_origin)):
                print("ERROR: Please check the time_origin")
                exit()
            if (time_extent == None or not np.isscalar(time_extent)):
                print("ERROR: Please check the time_extent")
                exit()
        else:
            if (time_origin != None):
                print(
                    "Errror: The time_origin need to be None when time_dependent is false"
                )
                exit()
            if (time_extent != None):
                print(
                    "Errror: The time_extent need to be None when time_dependent is false"
                )
                exit()

        # check space inputs and set dimension
        self.space_origin = (space_origin, ) if (
            np.isscalar(space_origin)) else space_origin
        self.space_extent = (space_extent, ) if (
            np.isscalar(space_extent)) else space_extent

        lso = len(self.space_origin)
        lse = len(self.space_extent)
        self.space_ndims = lso
        if (lso != lse):
            print(
                "ERROR: Please check dimention of space_origin and space_extent."
            )
            exit()
        elif lso == 1:
            self.space_shape = "rectangular_1d"
        elif lso == 2:
            self.space_shape = "rectangular_2d"
        elif lso == 3:
            self.space_shape = "rectangular_3d"
        else:
            print("ERROR: Rectangular supported is should be 1d/2d/3d.")

    # domain sampling discretize
    def sampling_discretize(self,
                            time_nsteps=None,
                            space_point_size=None,
                            space_nsteps=None):
        # TODO
        # check input
        self.space_point_size = (space_point_size, ) if (
            np.isscalar(space_point_size)) else space_point_size

        self.space_nsteps = (space_nsteps, ) if (
            np.isscalar(space_nsteps)) else space_nsteps

        # discretization time space with linspace
        steps = []
        if self.time_dependent == True:
            time_steps = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)

        # sampling in space discretization
        space_points = []
        for i in range(space_point_size):
            current_point = []
            for j in range(self.space_ndims):
                # get a random value in [space_origin[j], space_extent[j]]
                random_value = self.space_origin[j] + (
                    self.space_extent[j] - self.space_origin[j]
                ) * np.random.random_sample()
                current_point.append(random_value)
            space_points.append(current_point)

        # add boundry value
        if (self.space_ndims == 1):
            nbc = 2
            space_points.append(self.space_origin[-1])
            space_points.append(self.space_extent[-1])
            bc_index = np.ndarray(2, dtype=int)
            bc_index[0] = space_point_size
            bc_index[1] = space_point_size + 1
        elif (self.space_ndims == 2):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nbc = nx * ny - (nx - 2) * (ny - 2)
            bc_index = np.ndarray(nbc, dtype=int)
            nbc = 0
            x_start = self.space_origin[0]
            delta_x = (self.space_extent[0] - self.space_origin[0]) / (nx - 1)
            y_start = self.space_origin[1]
            delta_y = (self.space_extent[1] - self.space_origin[1]) / (ny - 1)
            for j in range(ny):
                for i in range(nx):
                    if (j == 0 or j == ny - 1 or i == 0 or i == nx - 1):
                        x = x_start + i * delta_x
                        y = y_start + j * delta_y
                        space_points.append([x, y])
                        bc_index[nbc] = space_point_size + nbc
                        nbc += 1
        elif (self.space_ndims == 3):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            nbc = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
            bc_index = np.ndarray(nbc, dtype=int)
            nbc = 0
            x_start = self.space_origin[0]
            delta_x = (self.space_extent[0] - self.space_origin[0]) / (nx - 1)
            y_start = self.space_origin[1]
            delta_y = (self.space_extent[1] - self.space_origin[1]) / (ny - 1)
            z_start = self.space_origin[2]
            delta_z = (self.space_extent[2] - self.space_origin[2]) / (nz - 1)
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if (k == 0 or k == nz - 1 or j == 0 or j == ny - 1 or
                                i == 0 or i == nx - 1):
                            x = x_start + i * delta_x
                            y = y_start + j * delta_y
                            z = z_start + k * delta_z
                            space_points.append([x, y, z])
                            bc_index[nbc] = space_point_size + nbc
                            nbc += 1
        space_domain = np.array(space_points)

        # bc_index with time-domain
        nbc = len(bc_index)
        if self.time_dependent == True:
            bc_offset = np.arange(time_nsteps).repeat(len(bc_index))
            bc_offset = bc_offset * len(space_domain)
            bc_index = np.tile(bc_index, time_nsteps)
            bc_index = bc_index + bc_offset

        # IC index
        if self.time_dependent == True:
            ic_index = bc_index[0:nbc]

        # return discrete geometry
        geo_disc = GeometryDiscrete()
        domain = []
        if self.time_dependent == True:
            # Get the time-space domain which combine the time domain and space domain
            for time in time_steps:
                current_time = time * np.ones(
                    (len(space_domain), 1), dtype=np.float32)
                current_domain = np.concatenate(
                    (current_time, space_domain), axis=-1)
                domain.append(current_domain.tolist())
            time_size = len(time_steps)
            space_domain_size = space_domain.shape[0]
            domain_dim = len(space_domain[0]) + 1
            domain = np.array(domain).reshape(
                (time_size * space_domain_size, domain_dim))

        if self.time_dependent == True:
            geo_disc.set_domain(
                time_domain=time_steps,
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent,
                time_space_domain=domain)
            geo_disc.set_bc_index(bc_index)
            geo_disc.set_ic_index(ic_index)
        else:
            geo_disc.set_domain(
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent)
            geo_disc.set_bc_index(bc_index)

        vtk_obj_name, vtk_obj, vtk_data_size = self.obj_vtk()
        geo_disc.set_vtk_obj(vtk_obj_name, vtk_obj, vtk_data_size)

        return geo_disc

    # domain discretize
    def discretize(self, time_nsteps=None, space_nsteps=None):

        # check input
        self.space_nsteps = (space_nsteps, ) if (
            np.isscalar(space_nsteps)) else space_nsteps

        # discretization time space with linspace
        steps = []
        if self.time_dependent == True:
            time_steps = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)

        # discretization each space dimention with linspace
        for i in range(self.space_ndims):
            steps.append(
                np.linspace(
                    self.space_origin[i],
                    self.space_extent[i],
                    self.space_nsteps[i],
                    endpoint=True))

        # meshgrid and stack to cordinates
        if self.time_dependent == True:
            nsteps = time_nsteps
            ndims = self.space_ndims + 1
        else:
            nsteps = 1
            ndims = self.space_ndims
        for i in self.space_nsteps:
            nsteps *= i

        if (self.space_ndims == 1):
            space_domain = steps[0]
        if (self.space_ndims == 2):
            mesh = np.meshgrid(steps[1], steps[0], sparse=False, indexing='ij')
            space_domain = np.stack(
                (mesh[1].reshape(-1), mesh[0].reshape(-1)), axis=-1)
        elif (self.space_ndims == 3):
            mesh = np.meshgrid(
                steps[2], steps[1], steps[0], sparse=False, indexing='ij')
            space_domain = np.stack(
                (mesh[2].reshape(-1), mesh[1].reshape(-1),
                 mesh[0].reshape(-1)),
                axis=-1)

        # bc_index
        if (self.space_ndims == 1):
            bc_index = np.ndarray(2, dtype=int)
            bc_index[0] = 0
            bc_index[1] = self.space_nsteps[-1]
        elif (self.space_ndims == 2):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nbc = nx * ny - (nx - 2) * (ny - 2)
            bc_index = np.ndarray(nbc, dtype=int)
            nbc = 0
            for j in range(ny):
                for i in range(nx):
                    if (j == 0 or j == ny - 1 or i == 0 or i == nx - 1):
                        bc_index[nbc] = j * nx + i
                        nbc += 1
        elif (self.space_ndims == 3):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            nbc = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
            bc_index = np.ndarray(nbc, dtype=int)
            nbc = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if (k == 0 or k == nz - 1 or j == 0 or j == ny - 1 or
                                i == 0 or i == nx - 1):
                            bc_index[nbc] = k * nx * ny + j * nx + i
                            nbc += 1

        # bc_index with time-domain
        nbc = len(bc_index)
        if self.time_dependent == True:
            bc_offset = np.arange(time_nsteps).repeat(len(bc_index))
            bc_offset = bc_offset * len(space_domain)
            bc_index = np.tile(bc_index, time_nsteps)
            bc_index = bc_index + bc_offset

        # IC index
        if self.time_dependent == True:
            ic_index = bc_index[0:nbc]

        # return discrete geometry
        geo_disc = GeometryDiscrete()
        domain = []
        if self.time_dependent == True:
            # Get the time-space domain which combine the time domain and space domain
            for time in time_steps:
                current_time = time * np.ones(
                    (len(space_domain), 1), dtype=np.float32)
                current_domain = np.concatenate(
                    (current_time, space_domain), axis=-1)
                domain.append(current_domain.tolist())
            time_size = len(time_steps)
            space_domain_size = space_domain.shape[0]
            domain_dim = len(space_domain[0]) + 1
            domain = np.array(domain).reshape(
                (time_size * space_domain_size, domain_dim))

        if self.time_dependent == True:
            geo_disc.set_domain(
                time_domain=time_steps,
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent,
                time_space_domain=domain)
            geo_disc.set_bc_index(bc_index)
            geo_disc.set_ic_index(ic_index)
        else:
            geo_disc.set_domain(
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent)
            geo_disc.set_bc_index(bc_index)

        vtk_obj_name, vtk_obj, vtk_data_size = self.obj_vtk()
        geo_disc.set_vtk_obj(vtk_obj_name, vtk_obj, vtk_data_size)

        # mpl_obj, mpl_data_shape = self.obj_mpl()
        # geo_disc.set_mpl_obj(mpl_obj, mpl_data_shape)

        return geo_disc

    # visu vtk
    def obj_vtk(self):
        # prepare plane obj 2d
        if self.space_ndims == 2:
            vtkobjname = "vtkPlanceSource"
            self.plane = vtk.vtkPlaneSource()
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            self.plane.SetResolution(nx - 1, ny - 1)
            self.plane.SetOrigin(
                [self.space_origin[0], self.space_origin[1], 0])
            self.plane.SetPoint1(
                [self.space_extent[0], self.space_origin[1], 0])
            self.plane.SetPoint2(
                [self.space_origin[0], self.space_extent[1], 0])
            self.plane.Update()
            vtk_data_size = self.plane.GetOutput().GetNumberOfPoints()
            return vtkobjname, self.plane, vtk_data_size
        elif self.space_ndims == 3:
            vtkobjname = "vtkImageData"
            self.img = vtk.vtkImageData()
            self.img.SetOrigin(self.space_origin[0], self.space_origin[1],
                               self.space_origin[2])
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            self.img.SetDimensions(nx, ny, nz)
            vtk_data_size = self.img.GetNumberOfPoints()
            return vtkobjname, self.img, vtk_data_size


# # visu matplotlib
# def obj_mpl(self):
#     # prepare plan obj 2d
#     if self.space_ndims == 2:
#         fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
#     return self.ax, (self.space_nsteps[0], self.space_nsteps[1])
