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

from .pde_base import PDE
import sympy
import numpy as np

__all__ = ['Laplace']


# Laplace equation
class Laplace(PDE):
    """
    Laplace Equation
    
    Parameters:
        dim (integer): equation's dimention. 1, 2 and 3 are supported.
        weight (optional, float or list of float): weight used in computing equation loss. The default value is 1.0.

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.pde.Laplace(dim=2)
    """

    def __init__(self, dim=2, weight=1.0):
        super(Laplace, self).__init__(1, weight=weight)

        if dim == 1:
            # independent variable
            x = sympy.Symbol('x')

            # dependent variable
            u = sympy.Function('u')(x, )

            # variables in order
            self.indvar = [x]
            self.dvar = [u]

            # order
            self.order = 2

            # equations and rhs
            self.equations = [u.diff(x).diff(x)]
            self.rhs = [0.0]

        elif dim == 2:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable
            u = sympy.Function('u')(x, y)

            # variables in order
            self.indvar = [x, y]
            self.dvar = [u]

            # order
            self.order = 2

            # equations and rhs
            self.equations = [u.diff(x).diff(x) + u.diff(y).diff(y)]
            self.rhs = [0.0]

        elif dim == 3:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            # dependent variable
            u = sympy.Function('u')(x, y, z)

            # variables in order
            self.indvar = [x, y, z]
            self.dvar = [u]

            # order
            self.order = 2

            # equations and rhs
            self.equations = [
                u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z)
            ]
            self.rhs = [0.0]
