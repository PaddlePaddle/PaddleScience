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

import sympy


class BC:
    def __init__(self, name, weight=1.0):
        self.category = None
        self.name = name
        self.rhs = 0.0
        self.weight = weight  # none, scale or lambda
        self.rhs_disc = 0.0
        self.weight_disc = 1.0


class Free(BC):
    def __init__(self, name, weight):
        super(Free, self).__init__(name, weight)
        self.category = "Free"

    def formula(self, indvar):
        return 0.0


class Dirichlet(BC):
    def __init__(self, name, rhs=0.0, weight=1.0):
        super(Dirichlet, self).__init__(name, weight)
        self.category = "Dirichlet"
        self.formula = None
        self.rhs = rhs

    def formula(self, indvar):
        self.formula = sympy.Function(self.name)(*indvar)


class Neumann(BC):
    def __init__(self, name, rhs=0.0, weight=1.0):
        super(Neumann, self).__init__(name, weight)
        self.category = "Neumann"
        self.rhs = rhs

    def formula(self, indvar):
        n = sympy.Symbol('n')
        u = sympy.Function(self.name)(n)(*indvar)
        return u.diff(n)


class Robin(BC):
    def __init__(self, name, rhs=0.0, weight=1.0):
        super(Robin, self).__init__(name, weight)
        self.category = "Robin"
        self.rhs = rhs

    def formula(self, indvar):
        n = sympy.Symbol('n')
        u = sympy.Function(self.name)(n)(*indvar)
        return u + u.diff(n)


# if __name__ == "__main__":

#     # set geometry and boundary
#     geo = pcsi.geometry.Rectangular(origine=(0.0, 0.0), extent=(1.0, 1.0))
#     geo.add_boundary(name="top", condition=lambda x, y: y == 1.0, normal=(0.0, 1.0))
#     geo.add_boundary(name="down", condition=lambda x, y: y == 0.0, normal=(0.0, -1.0))

#     # define N-S
#     pde = psci.pde.NavierStokes(nu=0.1, rho=1.0, dim=2, time_dependent=False)

#     # set bounday condition
#     bctop_u = psci.bc.Dirichlet('u', 0)
#     bctop_v = psci.bc.Dirichlet('v', 0)

#     bcdown_u = psci.bc.Newmann('u', 0)

#     # bounday and bondary condition to pde
#     pde.add_bc("top", bctop_u, bctop_v)
