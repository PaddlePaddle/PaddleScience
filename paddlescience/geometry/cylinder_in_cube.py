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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .geometry_discrete import GeometryDiscrete
from .geometry import Geometry
from .rectangular import Rectangular
import numpy as np
import vtk
import matplotlib.pyplot as plt
import math


# CylinderInRectangular
class CylinderInRectangular(Rectangular):
    """
    Three dimentional cylinder in a cube, the height of the circle is the same as that of the cube

    Parameters:
        time_dependent: does it depend on time
        time_origin: start time
        time_extent: finish time
        space_origin: cordinate of left-bottom point of rectangular
        space_extent: extent of rectangular
        circle_center: coordinate point of the center of the circle
        circle_radius: circle radius

    Example:
        >>> import paddlescience as psci
        >>> geo = psci.geometry.CylinderInRectangular(
                time_dependent=True,
                time_origin=0,
                time_extent=0.5,
                space_origin=(-0.05, -0.05, -0.05),
                space_extent=(0.05, 0.05, 0.05),
                circle_center=(0, 0),
                circle_radius=0.02)

    """

    # init function
    def __init__(self,
                 time_dependent=False,
                 time_origin=None,
                 time_extent=None,
                 space_origin=None,
                 space_extent=None,
                 circle_center=None,
                 circle_radius=None):
        super(CylinderInRectangular, self).__init__(time_dependent,
                                                    time_origin, time_extent,
                                                    space_origin, space_extent)

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

        # check the circle is reasonable
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        if (len(self.circle_center) != 2):
            print("Error: The circle is 2D ")
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
        elif lso == 2:
            self.space_shape = "rectangular_2d"
        elif lso == 3:
            self.space_shape = "rectangular_3d"
        else:
            print("ERROR: Rectangular supported is should be 2d/3d.")

    # domain sampling discretize
    def sampling_discretize(self,
                            time_nsteps=None,
                            space_npoints=None,
                            space_nsteps=None,
                            circle_bc_size=None,
                            real_data=None):
        # check input
        self.space_npoints = (space_npoints, ) if (
            np.isscalar(space_npoints)) else space_npoints

        self.space_nsteps = (space_nsteps, ) if (
            np.isscalar(space_nsteps)) else space_nsteps

        # discretization time space with linspace
        steps = []
        if self.time_dependent == True:
            time_steps = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)

        # sampling in space discretization
        space_points = []
        current_gen_num = 0
        while current_gen_num < space_npoints:
            current_point = []
            for j in range(self.space_ndims):
                # get a random value in [space_origin[j], space_extent[j]]
                random_value = self.space_origin[j] + (self.space_extent[j] -
                                                       self.space_origin[j]
                                                       ) * np.random.rand()
                current_point.append(random_value)
            # if the point is in circle, do not use it
            x_x0 = current_point[0] - self.circle_center[0]
            y_y0 = current_point[1] - self.circle_center[1]
            if (x_x0**2 + y_y0**2 > self.circle_radius**2):
                space_points.append(current_point)
                current_gen_num += 1

        # more point for the cylinder around
        # current_gen_num = 0
        # while current_gen_num < space_npoints:
        #     current_point = []
        #     for j in range(self.space_ndims):
        #         # get a random value in [space_origin[j], space_extent[j]]
        #         random_value = self.space_origin[j] + (self.space_extent[j] -
        #                                                self.space_origin[j]
        #                                                ) * np.random.rand()
        #         current_point.append(random_value)
        #     # if the point is in circle, do not use it
        #     x_x0 = current_point[0] - self.circle_center[0]
        #     y_y0 = current_point[1] - self.circle_center[1]
        #     is_not_in_cycle = (x_x0**2 + y_y0**2 > self.circle_radius**2)
        #     # if the point is in more area
        #     x_left = self.circle_center[0] - 2*self.circle_radius
        #     x_right = self.circle_center[0] + 2*self.circle_radius
        #     y_down = self.circle_center[1] - 2*self.circle_radius
        #     y_up = self.circle_center[1] + 2*self.circle_radius 
        #     is_in_more_area_x = ( current_point[0]> x_left and current_point[0]< x_right)
        #     is_in_more_area_y = ( current_point[1] > y_down and current_point[1] < y_up)
        #     # if the point is OK
        #     if (is_not_in_cycle and (is_in_more_area_x or is_in_more_area_y)):
        #         space_points.append(current_point)
        #         current_gen_num += 1

        # space_npoints = 2 * space_npoints

        # add boundry value in rectangular
        if (self.space_ndims == 2):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nbc = nx * ny - (nx - 2) * (ny - 2)
            bc_index = []
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
                        # if the point is in circle, do not use it
                        x_x0 = x - self.circle_center[0]
                        y_y0 = y - self.circle_center[1]
                        if (x_x0**2 + y_y0**2 > self.circle_radius**2):
                            space_points.append([x, y])
                            bc_index.append(space_npoints + nbc)
                            nbc += 1
        elif (self.space_ndims == 3):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            nbc = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
            bc_index = []
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
                            # if the point is in circle, do not use it
                            x_x0 = x - self.circle_center[0]
                            y_y0 = y - self.circle_center[1]
                            if (x_x0**2 + y_y0**2 > self.circle_radius**2):
                                space_points.append([x, y, z])
                                bc_index.append(space_npoints + nbc)
                                nbc += 1

        # add boundry value in cylinder
        current_gen_circle_num = 0
        bc_start = space_npoints + len(bc_index)
        if (self.space_ndims == 3):
            circle_bc_size *= self.space_nsteps[2]
        while current_gen_circle_num < circle_bc_size:
            # generate a random x in [x0-r,x0+r]
            x = (self.circle_center[0] - self.circle_radius
                 ) + np.random.rand() * self.circle_radius * 2
            y1 = self.circle_center[1] - math.sqrt(self.circle_radius**2 - (
                x - self.circle_center[0])**2)
            y2 = self.circle_center[1] + math.sqrt(self.circle_radius**2 - (
                x - self.circle_center[0])**2)
            if (self.space_ndims == 2):
                space_points.append([x, y1])
                space_points.append([x, y2])
                bc_index.append(bc_start + current_gen_circle_num)
                bc_index.append(bc_start + current_gen_circle_num + 1)
                current_gen_circle_num += 2
            elif (self.space_ndims == 3):
                z_steps = np.linspace(
                    self.space_origin[2],
                    self.space_extent[2],
                    self.space_nsteps[2],
                    endpoint=True)
                for i in range(len(z_steps)):
                    z = z_steps[i]
                    space_points.append([x, y1, z])
                    space_points.append([x, y2, z])
                    bc_index.append(bc_start + current_gen_circle_num)
                    bc_index.append(bc_start + current_gen_circle_num + 1)
                    current_gen_circle_num += 2

        # add real data
        # real_data_len = len(real_data)
        space_points = np.array(space_points)
        if real_data is not None:
            real_xy = real_data[:, 0:self.space_ndims]
            space_points = np.concatenate((space_points, real_xy), axis=0)

        bc_index = np.array(bc_index)
        space_domain = space_points

        # bc_index with time-domain
        nbc = len(bc_index)
        if self.time_dependent == True:
            bc_offset = np.arange(time_nsteps).repeat(len(bc_index))
            bc_offset = bc_offset * len(space_domain)
            bc_index = np.tile(bc_index, time_nsteps)
            bc_index = bc_index + bc_offset

        # IC index
        if self.time_dependent == True:
            ic_index = np.arange(len(space_domain))

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

        return geo_disc
