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


class IC:
    """
    Initial condition for time-dependent equation
 
    Parameters:
        name (string): name of dependent variable.
        rhs (float or lambda function): right-hand side of initial boundary condition. The default value is 0.0.
        weight (optional, float or lambda function): weight for computing initial loss. The default value is 1.0.

    Example:
        >>> import paddlescience as psci
        >>> ic1 = psci.ic.IC("u")
        >>> ic2 = psci.ic.IC("u", rhs=0.0)
        >>> ic3 = psci.ic.IC("u", rhs=lambda x, y: cos(x)*cosh(y))
    """

    def __init__(self, name, rhs=None, weight=1.0):
        self.name = name
        self.rhs = rhs
        self.weight = weight
        self.rhs_disc = rhs
        self.weight_disc = weight

    def to_formula(self, indvar):
        self.formula = sympy.Function(self.name)(*indvar)

    def discretize(self, indvar):
        ic_disc = copy.deepcopy(self)
        ic_disc.to_formula(indvar)
        return ic_disc
