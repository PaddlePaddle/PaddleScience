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

import numpy as np
import paddle


class GeometryDiscrete:
    """
    Geometry Discrete
    """

    def __init__(self):
        # time discretization
        self.time_dependent = False
        self.time_domain_size = -1
        self.time_domain = None
        # space discretization
        self.space_dims = 2
        self.space_domain_size = -1
        self.space_domain = None
        # time-space domain after discretization
        self.domain_dims = None
        self.domain_size = -1
        self.domain = None
        # time IC index
        self.ic_index = None
        # space BC index
        self.bc_index = None
        # visu vtk obj
        self.vtk_obj = None
        self.vtk_num_points = -1

    # set time domain
    def set_time_domain(self, time_domain):
        self.time_dependent = True
        self.time_domain = time_domain
        self.time_domain_size = len(time_domain)

    # set space domain
    def set_space_domain(self, space_domain):
        self.space_domain = space_domain
        self.space_domain_size = len(space_domain)
        self.space_dims = len(space_domain[0])

    # set domain
    def set_domain(self,
                   time_domain=None,
                   time_origin=None,
                   time_extent=None,
                   space_domain=None,
                   space_origin=None,
                   space_extent=None,
                   time_space_domain=None):
        # time domain
        if time_domain is not None:
            self.set_time_domain(time_domain)
        # space domain
        if space_domain is not None:
            self.set_space_domain(space_domain)
        # time-space domain
        if time_space_domain is not None:
            self.domain = time_space_domain
            self.domain_size = len(time_space_domain)
            self.domain_dims = len(time_space_domain[0])
        self.space_origin = space_origin
        self.space_extent = space_extent

    # set bc index
    def set_bc_index(self, bc_index):
        self.bc_index = bc_index

    # set ic index
    def set_ic_index(self, ic_index):
        self.ic_index = ic_index

    # get bc index
    def get_bc_index(self):
        """
        Get bounday index 

        Returns
        -------
        bc_index: numpy array
            Bounday index indicating which points are on the boundary.
        """
        return self.bc_index

    # get bc index
    def get_ic_index(self):
        """
        Get init index 

        Returns
        -------
        ic_index: numpy array
        """
        return self.ic_index

    # get domain size
    def get_domain_size(self):
        if (self.domain_size != -1):
            return self.domain_size
        else:
            return self.space_domain_size

    # get time domain
    def get_time_domain(self):
        return self.time_domain

    # get space domain
    def get_space_domain(self):
        """
        Get space domain cordinates

        Returns
        -------
        space_domain: numay array
            Space domain cordinates
        """
        return self.space_domain

    # get domain
    def get_domain(self):
        if (self.domain is not None):
            return self.domain
        else:
            return self.space_domain

    # set time steps
    def set_time_steps(self, time_steps):
        self.time_steps = time_steps

    # set space steps
    def set_space_steps(self, space_steps):
        self.space_steps = space_steps

    def to_tensor(self):
        self.space_domain = paddle.to_tensor(
            self.space_domain, dtype="float32")
        self.space_domain.stop_gradient = False
        for batch_id in range(self.num_batch):
            self.bc_index[batch_id] = paddle.to_tensor(
                self.bc_index[batch_id], dtype='int64')
            if self.ic_index is not None:
                self.ic_index[batch_id] = paddle.to_tensor(
                    self.ic_index[batch_id], dtype='int64')
        if self.domain is not None:
            self.domain = paddle.to_tensor(self.domain, dtype="float32")
            self.domain.stop_gradient = False

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batch = self.get_domain_size() // batch_size
        if self.get_domain_size() % batch_size != 0:
            self.num_batch = self.num_batch + 1

        new_bc_index = [[] for _ in range(self.num_batch)]
        for idx in self.bc_index:
            new_bc_index[idx // batch_size].append(idx)
        self.bc_index = [np.array(el) for el in new_bc_index]

        if self.ic_index is not None:
            new_ic_index = [[] for _ in range(self.num_batch)]
            for idx in self.ic_index:
                new_ic_index[idx // batch_size].append(idx)
            self.ic_index = [np.array(el) for el in new_ic_index]

    def get_num_batch(self):
        return self.num_batch

    def set_vtk_obj(self, vtk_obj_name, vtk_obj, vtk_data_size):
        self.vtk_obj_name = vtk_obj_name
        self.vtk_obj = vtk_obj
        self.vtk_data_size = vtk_data_size

    def get_vtk_obj(self):
        return self.vtk_obj_name, self.vtk_obj, self.vtk_data_size

    # def set_mpl_obj(self, mpl_obj, mpl_data_shape):
    #     self.mpl_obj = mpl_obj
    #     self.mpl_data_shape = mpl_data_shape

    # def get_mpl_obj(self):
    #     return self.mpl_obj, self.space_domain self.vtk_data_shape
