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


class PDE:
    def __init__(self, num_pdes=1, time_dependent=False):
        # super(MathOperator, self).__init__()

        # time dependent / independent
        self.time_dependent = time_dependent

        # discretize method (work for time-dependent equation)
        self.time_discretize_method = None

        self.independent_variable = list()
        self.dependent_variable = list()
        self.dependent_variable_1 = list()

        # Equation list
        self.equations = list()

        # right-hand side
        self.rhs = list()

        # Geometry
        self.geometry = None

        # Boundary condition list
        self.bc = dict()

    def add_geometry(self, geo):

        self.geometry = geo

    def add_bc(self, name, *args):

        if name not in self.bc:
            self.bc[name] = list()

        for arg in args:
            arg.to_formula(self.independent_variable)
            self.bc[name].append(arg)

    def discretize(self, method):
        pass

    # def add_item(self, pde_index, coefficient, *args):
    #     # if derivative not in first_order_derivatives:
    #     #     self.need_2nd_derivatives = True
    #     self.pdes[pde_index].append(PDEItem(coefficient, args))

    # def get_pde(self, idx):
    #     return self.pdes[idx]

    # def set_ic_value(self, ic_value, ic_check_dim=None):
    #     self.ic_value = ic_value
    #     self.ic_check_dim = ic_check_dim

    # def set_bc_value(self, bc_value, bc_check_dim=None):
    #     """
    #         Set boudary value (Dirichlet boundary condition) to PDE

    #         Parameters:
    #             bc_value: array of values
    #             bc_check_dim (list):  Optional, default None. If is not None, this list contains the dimensions to set boundary condition values on. If is None, boundary condition values are set on all dimentions of network output. 
    #     """
    #     self.bc_value = bc_value
    #     self.bc_check_dim = bc_check_dim
    #     # print(self.bc_value)

    # def discretize(self):
    #     pass  # TODO

    # def to_tensor(self):
    #     # time
    #     if self.time_dependent == True:
    #         self.ic_value = paddle.to_tensor(self.ic_value, dtype='float32')
    #         self.ic_check_dim = paddle.to_tensor(
    #             self.ic_check_dim,
    #             dtype='int64') if self.ic_check_dim is not None else None
    #     # space
    #     self.bc_value = paddle.to_tensor(self.bc_value, dtype='float32')
    #     self.bc_check_dim = paddle.to_tensor(
    #         self.bc_check_dim,
    #         dtype='int64') if self.bc_check_dim is not None else None

    # def set_batch_size(self, batch_size):
    #     self.batch_size = batch_size
