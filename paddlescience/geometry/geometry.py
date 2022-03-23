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
import numpy as np
import vtk
import matplotlib.pyplot as plt


# Geometry
class Geometry:
    def __init__(self):
        self.condition = dict()  # condition (lambda) defining boundary
        self.normal = dict()  # boundary normal direction

    def add_boundary(self, name, condition, normal):

        self.condition[name] = condition
        self.normal[name] = normal

    def delete_boundary(self, name):

        if name in self.condition:
            del self.condition[name]

        if name in self.normal:
            del self.normal[name]

    def clear_boundary(self):

        self.condition.clear()
        self.normal.clear()

    # def sampling_discretize(self, time_nsteps, space_point_size, space_nsteps):
    #     geo_disc = GeometryDiscrete()
    #     return GeometryDiscrete()

    # def discretize(self, time_nsteps, space_nsteps):
    #     geo_disc = GeometryDiscrete()
    #     return GeometryDiscrete()
