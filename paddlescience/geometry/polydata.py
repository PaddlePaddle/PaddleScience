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
import pyvista as pv

__all__ = ['PolyData']


# Rectangular
class PolyData(Geometry):
    """
    Three dimentional PolyData

    Parameters:
        vertices(integer numpy.ndarray | float numpy.ndarray | double numpy.ndarray): Cordinate of points. Note that only 3D point data is supported currently.
        faces(integer numpy.ndarray): Face connectivity array. Note that faces must contain padding indicating the number of points in the face. And it needs to satisfy the right-hand rule. 

    Example:
        >>> import paddlescience as psci
        >>> vertices = np.array([[3, 3, 0], [7, 3, 0], 
        >>>    [5, 7, 0], [3, 3, 0.5], 
        >>>    [7, 3, 0.5], [5, 7, 0.5]])
        >>> # Right-hand rule
        >>> faces = np.hstack(
        >>>    [
        >>>        [3, 0, 2, 1],  # triangle
        >>>        [3, 3, 4, 5],  # triangle
        >>>        [4, 0, 3, 5, 2], # square
        >>>        [4, 1, 2, 5, 4], # square
        >>>        [4, 0, 1, 4, 3],  # square
        >>>    ]
        >>> )
        >>> triangle = psci.geometry.PolyData(vertices, faces)
    """

    def __init__(self, vertices, faces):
        super(PolyData, self).__init__()

        triangle = pv.PolyData(vertices, faces)
        triangle = triangle.triangulate()
        self.pv_mesh = triangle
