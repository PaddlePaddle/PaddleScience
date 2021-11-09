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
    def __init__(self):

        # time discretization
        self.time_dependent = False
        self.time_nsteps = 0
        # self.time_steps 

        # space discretization
        self.space_dims = 2
        # self.space_nsteps
        # self.space_steps 

        # time IC index
        self.ic_index = None

        # space BC index
        self.bc_index = None

    # set num of time steps
    def set_time_nsteps(self, time_nsteps):
        self.time_nsteps = time_nsteps

    # set time steps
    def set_time_steps(self, time_steps):
        self.time_dependent = True
        self.time_steps = time_steps
        self.time_nsteps = len(time_steps)

    # set num of space steps
    def set_space_nsteps(self, space_nsteps):
        self.space_nsteps = space_nsteps

    # set space steps
    def set_space_steps(self, space_steps):
        self.space_steps = space_steps
        self.space_dims = len(space_steps[0])
        # self.space_nsteps = len(space_steps)

    # set steps

    def set_steps(self, steps, origin=None, extent=None):
        self.steps = steps
        self.mesh = np.copy(steps)
        self.space_origin = origin
        self.space_extent = extent

    # set ic index

    def set_ic_index(self, ic_index):
        self.ic_index = ic_index

    # set bc index
    def set_bc_index(self, bc_index):
        self.bc_index = bc_index

    # get bc index
    def get_bc_index(self):
        return self.bc_index

    # get num of steps
    def get_nsteps(self):
        if self.time_dependent == True:
            nsteps = self.time_nsteps
            for i in space_nsteps:
                nsteps *= i
        else:
            nsteps = 1
            for i in self.space_nsteps:
                nsteps *= i
        return nsteps

    # get steps
    def get_step(self):
        return self.steps

    def to_tensor(self):
        self.steps = paddle.to_tensor(self.steps, dtype="float32")
        self.steps.stop_gradient = False
        for batch_id in range(self.num_batch):
            self.bc_index[batch_id] = paddle.to_tensor(
                self.bc_index[batch_id], dtype='int64')
            if self.ic_index is not None:
                self.ic_index[batch_id] = paddle.to_tensor(
                    self.ic_index[batch_id], dtype='int64')

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batch = self.get_nsteps() // batch_size
        if self.get_nsteps() % batch_size != 0:
            self.num_batch = self.num_batch + 1

        new_bc_index = [[] for _ in range(self.num_batch)]
        for idx in self.bc_index:
            new_bc_index[idx // batch_size].append(idx)
        self.bc_index = [np.array(el) for el in new_bc_index]

        if self.ic_index is not None:
            new_ic_index = [[] for _ in range(self.num_batch)]
            for idx in self.ic_index:
                new_ic_index[idx % batch_size].append(idx)
            self.ic_index = [np.array(el) for el in new_ic_index]

    def get_num_batch(self):
        return self.num_batch
