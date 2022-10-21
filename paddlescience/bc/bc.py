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
import copy


class BC:
    def __init__(self, name, weight=1.0):
        self.category = None
        self.name = name
        self.rhs = 0.0
        self.weight = weight  # none, scale or lambda
        self.rhs_disc = 0.0
        self.weight_disc = 1.0
        self.normal_disc = 1.0


class Free(BC):
    def __init__(self, name, weight):
        super(Free, self).__init__(name, weight)
        self.category = "Free"

    def to_formula(self, indvar):
        return None


class Dirichlet(BC):
    """
    Dirichlet boundary condition
 
    Parameters:
        name (string): name of dependent variable.
        rhs (float or lambda function): right-hand side of Dirichlet boundary condition. The default value is 0.0.
        weight (optional, float or lambda function): weight in computing boundary loss. The default value is 1.0.

    Example
        >>> import paddlescience as psci
        >>> bc1 = psci.bc.Dirichlet("u", rhs=0.0)
        >>> bc2 = psci.bc.Dirichlet("u", rhs=lambda x, y: cos(x)*cosh(y))
    """

    def __init__(self, name, rhs=0.0, weight=1.0):
        super(Dirichlet, self).__init__(name, weight)
        self.category = "Dirichlet"
        self.formula = None
        self.rhs = rhs

    def to_formula(self, indvar):
        self.formula = sympy.Function(self.name)(*indvar)

    def discretize(self, indvar):
        bc_disc = copy.deepcopy(self)
        bc_disc.to_formula(indvar)
        return bc_disc


class Neumann(BC):
    """
    Neumann boundary condition
 
    Parameters:
        name (string): Name of dependent variable
        rhs (float or lambda function): right-hand side of Neumann boundary condition. The default value is 0.0.
        weight (optional, float or lambda function): weight for computing boundary loss. The default value is 1.0.

    Example
        >>> import paddlescience as psci
        >>> bc1 = psci.bc.Neumann("u", rhs=1.0)
        >>> bc2 = psci.bc.Neumann("u", rhs=lambda x, y: 1.0)
    """

    def __init__(self, name, rhs=0.0, weight=1.0):
        super(Neumann, self).__init__(name, weight)
        self.category = "Neumann"
        self.rhs = rhs

    def to_formula(self, indvar):
        n = sympy.Symbol('n')
        u = sympy.Function(self.name)(*indvar)
        self.formula = sympy.Derivative(u, n)

    def discretize(self, indvar):
        bc_disc = copy.deepcopy(self)
        bc_disc.to_formula(indvar)
        return bc_disc


class Robin(BC):
    """
    Robin boundary condition
 
    Parameters:
        name (string): Name of dependent variable
        rhs (float or lambda function): right-hand side of Neumann boundary condition. The default value is 0.0.
        weight (optional, float or lambda function): weight for computing boundary loss. The default value is 1.0.

    Example:
        >>> import paddlescience as psci
        >>> bc1 = psci.bc.Robin("u", rhs=0.0)
        >>> bc2 = psci.bc.Robin("u", rhs=lambda x, y: 0.0)
    """

    def __init__(self, name, rhs=0.0, weight=1.0):
        super(Robin, self).__init__(name, weight)
        self.category = "Robin"
        self.rhs = rhs

    def to_formula(self, indvar):
        n = sympy.Symbol('n')
        u = sympy.Function(self.name)(*indvar)
        self.formula = u + sympy.Derivative(u, n)

    def discretize(self, indvar):
        bc_disc = copy.deepcopy(self)
        bc_disc.to_formula(indvar)
        return bc_disc
