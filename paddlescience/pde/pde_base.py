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

import numpy as np
from collections import OrderedDict


class PDE:
    def __init__(self, num_equations=1, time_dependent=False, weight=None):
        # super(MathOperator, self).__init__()

        # time dependent / independent
        self.time_dependent = time_dependent

        # independent variable
        # dependent variable on current time step n
        # dependent variable on next time step n+1
        self.independent_variable = list()
        self.dependent_variable = list()
        self.dependent_variable_n = list()

        # parameter in pde
        self.parameter = list()

        # equation
        self.equations = list()

        # right-hand side
        self.rhs = list()

        # boundary condition
        self.bc = OrderedDict()

        # geometry
        self.geometry = None

        # weight
        self.weight = weight

        # rhs disc
        self.rhs_disc = list()

        # weight disc
        self.weight_disc = list()

        # discretize method (for time-dependent)
        self.time_disc_method = None

        # u_n_disc
        self.u_n_disc = [None for i in range(num_equations)]

    def add_geometry(self, geo):
        self.geometry = geo

    def add_bc(self, name, *args):
        if name not in self.bc:
            self.bc[name] = list()

        for arg in args:
            arg.to_formula(self.independent_variable)
            self.bc[name].append(arg)
