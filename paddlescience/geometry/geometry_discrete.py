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


class GeometryDiscrete:
    """
    Geometry Discrete
    """

    def __init__(self):

        # TODO: data structure uniformation
        self.interior = None
        self.boundary = dict()
        self.normal = dict()
        self.data = None

    def __str__(self):
        return "TODO: Print for DiscreteGeometry"

    def padding(self, nprocs=1):

        # interior
        self.interior = self.__padding_array(nprocs, self.interior)
        # bc
        for name_b in self.boundary.keys():
            self.boundary[name_b] = self.__padding_array(nprocs,
                                                         self.boundary[name_b])
        # data
        self.data = self.__padding_array(nprocs, self.data)
        # TODO: normal

    def __padding_array(self, nprocs, array):
        npad = (len(array) % nprocs) % nprocs  # pad npad elements
        datapad = array[-1, :]
        for i in range(npad):
            array = np.append(array, datapad, axis=0)
        return array
