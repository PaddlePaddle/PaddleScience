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

        if dim == 1:
            # independent and dependent variable
            x = sympy.Symbol('x')
            u = sympy.Function('u')(x, )
            super(Laplace, self).__init__([x], [u], weight)
            self.add_equation(u.diff(x).diff(x))

        elif dim == 2:
            # independent and dependent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            u = sympy.Function('u')(x, y)
            super(Laplace, self).__init__([x, y], [u], weight)
            self.add_equation(u.diff(x).diff(x) + u.diff(y).diff(y))

        elif dim == 3:
            # independent and dependent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')
            u = sympy.Function('u')(x, y, z)
            super(Laplace, self).__init__([x, y, z], [u], weight)
            self.add_equation(
                u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z))
