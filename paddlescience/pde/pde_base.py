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

# from ..formula import MathOperator

import paddle

#NOTE: do not change these list
first_order_rslts = ['u', 'v', 'w', 'p']
first_order_derivatives = [['du/dt', 'du/dx', 'du/dy', 'du/dz'],
                           ['dv/dt', 'dv/dx', 'dv/dy', 'dv/dz'],
                           ['dw/dt', 'dw/dx', 'dw/dy', 'dw/dz'],
                           ['dp/dt', 'dp/dx', 'dp/dy', 'dp/dz']]

second_order_derivatives = [[['d2u/dt2', 'd2u/dtdx', 'd2u/dtdy', 'd2u/dtdz'],
                             ['d2u/dxdt', 'd2u/dx2', 'd2u/dxdy', 'd2u/dxdz'],
                             ['d2u/dydt', 'd2u/dydx', 'd2u/dy2', 'd2u/dydz'],
                             ['d2u/dzdt', 'd2u/dzdx', 'd2u/dzdy', 'd2u/dz2']],
                            [['d2v/dt2', 'd2v/dtdx', 'd2v/dtdy', 'd2v/dtdz'],
                             ['d2v/dxdt', 'd2v/dx2', 'd2v/dxdy', 'd2v/dxdz'],
                             ['d2v/dydt', 'd2v/dydx', 'd2v/dy2', 'd2v/dydz'],
                             ['d2v/dzdt', 'd2v/dzdx', 'd2v/dzdy', 'd2v/dz2']],
                            [['d2w/dt2', 'd2w/dtdx', 'd2w/dtdy', 'd2w/dtdz'],
                             ['d2w/dxdt', 'd2w/dx2', 'd2w/dxdy', 'd2w/dxdz'],
                             ['d2w/dydt', 'd2w/dydx', 'd2w/dy2', 'd2w/dydz'],
                             ['d2w/dzdt', 'd2w/dzdx', 'd2w/dzdy', 'd2w/dz2']],
                            [['d2p/dt2', 'd2p/dtdx', 'd2p/dtdy', 'd2p/dtdz'],
                             ['d2p/dxdt', 'd2p/dx2', 'd2p/dxdy', 'd2p/dxdz'],
                             ['d2p/dydt', 'd2p/dydx', 'd2p/dy2', 'd2p/dydz'],
                             ['d2p/dzdt', 'd2p/dzdx', 'd2p/dzdy', 'd2p/dz2']]]


class PDEItem:
    def __init__(self, coefficient, args):
        self.coefficient = coefficient
        self.derivative = []
        for arg in args:
            self.derivative.append(arg)


class PDE:
    def __init__(self, num_pdes=1, time_dependent=False):
        # super(MathOperator, self).__init__()

        # time dependent / independent
        self.time_dependent = time_dependent

        # whether or not need 2nd order derivate
        self.need_2nd_derivatives = True

        # pde definition
        self.num_pdes = num_pdes
        self.pdes = []
        for i in range(self.num_pdes):
            self.pdes.append([])

    def add_item(self, pde_index, coefficient, *args):
        # if derivative not in first_order_derivatives:
        #     self.need_2nd_derivatives = True
        self.pdes[pde_index].append(PDEItem(coefficient, args))

    def get_pde(self, idx):
        return self.pdes[idx]

    def set_ic_value(self, ic_value, ic_check_dim=None):
        self.ic_value = ic_value
        self.ic_check_dim = ic_check_dim

    def set_bc_value(self, bc_value, bc_check_dim=None):
        """
            Set boudary value (Dirichlet boundary condition) to PDE

            Parameters:
                bc_value: array of values
                bc_check_dim (list):  Optional, default None. If is not None, this list contains the dimensions to set boundary condition values on. If is None, boundary condition values are set on all dimentions of network output. 
        """
        self.bc_value = bc_value
        self.bc_check_dim = bc_check_dim
        # print(self.bc_value)

    def discretize(self):
        pass  # TODO

    def to_tensor(self):
        # time
        if self.time_dependent == True:
            self.ic_value = paddle.to_tensor(self.ic_value, dtype='float32')
            self.ic_check_dim = paddle.to_tensor(
                self.ic_check_dim,
                dtype='int64') if self.ic_check_dim is not None else None
        # space
        self.bc_value = paddle.to_tensor(self.bc_value, dtype='float32')
        self.bc_check_dim = paddle.to_tensor(
            self.bc_check_dim,
            dtype='int64') if self.bc_check_dim is not None else None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
