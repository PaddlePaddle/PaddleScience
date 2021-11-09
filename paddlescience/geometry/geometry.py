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
import numpy as np


# Geometry
class Geometry:
    def __init__(self,
                 time_dependent=False,
                 time_origin=None,
                 time_extent=None,
                 space_origin=None,
                 space_extent=None):

        # TODO
        # 1D input could be integer or list of integer, need to check and transform

        # TODO check inputs  

        # time information
        self.time_dependent = time_dependent
        self.time_origin = time_origin
        self.time_extent = time_extent

        # space information
        self.space_shape = ""
        self.space_ndims = len(space_origin)
        self.space_origin = space_origin
        self.space_extent = space_extent
        self.space_radius = space_extent

    def discretize(self, time_nsteps, space_nsteps):
        geo_disc = GeometryDiscrete()
        return GeometryDiscrete()


# Rectangular
class Rectangular(Geometry):

    # init function
    def __init__(self,
                 time_dependent=False,
                 time_origin=None,
                 time_extent=None,
                 space_origin=None,
                 space_extent=None):
        super(Rectangular,
              self).__init__(time_dependent, time_origin, time_extent,
                             space_origin, space_extent)
        self.space_shape = "rectangular"

    # domain discretize
    def discretize(self, time_nsteps=None, space_nsteps=None):

        geo_disc = GeometryDiscrete()

        # domain discretize
        steps = []

        # time discretization
        if self.time_dependent == True:
            geo_disc.set_time_nsteps(time_nsteps)
            ts = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)
            geo_disc.set_time_steps(ts)
            steps.append(ts)

        # space discretization
        for i in range(self.space_ndims):
            steps.append(
                np.linspace(
                    self.space_origin[i],
                    self.space_extent[i],
                    space_nsteps[i],
                    endpoint=True))

#       # meshgrid
#       mesh = np.meshgrid(steps[0], steps[1], sparse=False, indexing='xy')
#       steps = []
#       for ms in mesh:
#           steps.append(ms.flatten())

# TODO: better code for the discretization
# 2D is supported for the moment

        if self.time_dependent == True:
            nsteps = time_nsteps
            for i in space_nsteps:
                nsteps *= i
            ndims = self.space_ndims + 1
        else:
            nsteps = 1
            for i in space_nsteps:
                nsteps *= i
            ndims = self.space_ndims

        domain = np.zeros((nsteps, ndims))  # domain
        for i in range(space_nsteps[1]):
            for j in range(space_nsteps[0]):
                domain[i * space_nsteps[0] + j][0] = steps[0][j]
                domain[i * space_nsteps[0] + j][1] = steps[1][i]

        geo_disc.set_space_nsteps(space_nsteps)
        geo_disc.set_steps(domain, self.space_origin, self.space_extent)

        # bc_index TODO optimize
        nbc = 0
        for i in range(space_nsteps[1]):
            for j in range(space_nsteps[0]):
                if (i == 0) or (i == space_nsteps[1] - 1) or (j == 0) or (
                        j == space_nsteps[0] - 1):
                    nbc += 1

        bc_index = np.zeros(nbc, dtype=int)  # BC index
        nbc = 0
        for i in range(space_nsteps[1]):
            for j in range(space_nsteps[0]):
                if (i == 0) or (i == space_nsteps[1] - 1) or (j == 0) or (
                        j == space_nsteps[0] - 1):
                    bc_index[nbc] = i * space_nsteps[0] + j
                    nbc += 1
        geo_disc.set_bc_index(bc_index)

        # print(domain)
        # print(bc_index)

        # IC index TODO

        return geo_disc
