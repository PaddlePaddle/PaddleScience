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

import numpy as np
import paddle
from collections import OrderedDict

class GeometryDiscrete:
    """
    Geometry Discrete
    """

    def __init__(self):

        # TODO: data structure uniformation
        self.interior = None
        self.boundary = OrderedDict()
        self.normal = OrderedDict()
        self.user = None

    def __str__(self):
        return "TODO: Print for DiscreteGeometry"

    def add_customized_points(self, cordinate):
        """
        Add cutomized points (cordinate) to geometry
    
        Parameters:
            cord(array): Cordinate of customized points

        Example:
            >>> cord = numpy.array([[0,0,0],[0,1,2]])
            >>> geo_disc.add_customized_points(cordinate=cord)
        """
        self.user = cord

    def padding(self, nprocs=1):

        # interior
        if type(self.interior) is np.ndarray:
            self.interior = self.__padding_array(nprocs, self.interior)

        # bc
        for name_b in self.boundary.keys():
            if type(self.boundary[name_b]) is np.ndarray:
                self.boundary[name_b] = self.__padding_array(
                    nprocs, self.boundary[name_b])

        # user
        if type(self.user) is np.ndarray:
            self.user = self.__padding_array(nprocs, self.user)

        # TODO: normal

    def __padding_array(self, nprocs, array):
        npad = (nprocs - len(array) % nprocs) % nprocs  # pad npad elements
        datapad = array[-1, :].reshape((-1, array[-1, :].shape[0]))
        for i in range(npad):
            array = np.append(array, datapad, axis=0)
        return array

    def split(self, nprocs=1):

        dp = list()
        for i in range(nprocs):
            dp.append(self.sub(nprocs, i))
        return dp

    def sub(self, nprocs, n):

        subp = GeometryDiscrete()

        # interior
        ni = int(len(self.interior) / nprocs)
        s = ni * n
        e = ni * (n + 1)
        subp.interior = self.interior[s:e, :]

        # boundary
        for name, b in self.boundary.items():
            nb = int(len(b) / nprocs)
            s = nb * n
            e = nb * (n + 1)
            subp.boundary[name] = b[s:e, :]

        # user
        if self.user is not None:
            nd = int(len(self.user) / nprocs)
            s = nd * n
            e = nd * (n + 1)
            subp.user = self.user[s:e, :]

        return subp
