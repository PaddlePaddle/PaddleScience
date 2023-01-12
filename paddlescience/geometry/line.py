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

__all__ = ['Line']


# line
class Line(Geometry):
    '''
    1D line shape

    Parameters:
    origin: Coridinate of start point of line
    extent: Extent of line

    Example:
        >>> import paddlescience as psci
        >>> geo1d = psci.geometry.line(origin=(0,), extent=(1,))
    '''
    def __init__(self, origin, extent):
        super(Line, self).__init__()
        
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
    def discretize(self, method='uniform', npoints=100, padding=True):
        """
        Discretize line

        Parameters:
            method ("uniform" ): Discretize line using method "uniform".
            npoints (integer / integer list): Number of points 

        Example:
            >>> import paddlescience as psci
            >>> geo = psci.geometry.line(origin=(0,0), extent=(1,0))
            >>> geo.discretize(method="uniform", npoints=100)
        """   

        if method == "uniform":
            points = self._uniform_mesh(npoints)

        return super(Line, self)._mesh_to_geo_disc(points, padding)


    def _uniform_mesh(self, npoints, origin=None, extent=None):

        if origin is None:
            origin = self.origin
        if extent is None:
            extent = self.extent

        if np.isscalar(npoints):
            if self.ndims == 1:
                nd = [npoints]
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
        return points
