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
import pyvista as pv

__all__ = ['PolyData']


# Rectangular
class PolyData(Geometry):
    # """
    # Two dimentional rectangular or three dimentional cube

    # Parameters:
    #     origin: Cordinate of left-bottom point of rectangular
    #     extent: Extent of rectangular

    # Example:
    #     >>> import paddlescience as psci
    #     >>> geo2d = psci.geometry.Rectangular(origin=(0.0,0.0), extent=(1.0,1.0))
    #     >>> geo3d = psci.geometry.Rectangular(origin=(0.0,0.0,0.0), extent=(2.0,2.0,2.0))
    # """

    def __init__(self, vertices, faces):
        super(PolyData, self).__init__()

        triangle = pv.PolyData(vertices, faces)
        triangle = triangle.triangulate()
        self.pv_mesh = triangle
