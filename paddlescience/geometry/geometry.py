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
import vtk
import matplotlib.pyplot as plt


# Geometry
class Geometry:
    def __init__(self,
                 time_dependent=False,
                 time_origin=None,
                 time_extent=None,
                 space_origin=None,
                 space_extent=None):

        # TODO check inputs  

        # time information
        self.time_dependent = time_dependent
        self.time_origin = time_origin
        self.time_extent = time_extent

        # space information
        self.space_shape = None
        self.space_origin = None
        self.space_extent = None

        # self.space_radius = (space_radius, ) if (np.isscalar(space_radius)) else space_radius

    def sampling_discretize(self, time_nsteps, space_point_size, space_nsteps):
        geo_disc = GeometryDiscrete()
        return GeometryDiscrete()

    def discretize(self, time_nsteps, space_nsteps):
        geo_disc = GeometryDiscrete()
        return GeometryDiscrete()
