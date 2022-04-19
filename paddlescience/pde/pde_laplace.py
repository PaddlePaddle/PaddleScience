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

from .pde_base import PDE

__all__ = ['Laplace']


class Laplace(PDE):
    """
    Two dimentional Laplace Equation
    
    .. math::
        \\frac{\\partial^2 u}{\\partial x^2} + \\frac{\\partial^2 u}{\\partial y^2} = 0.

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.pde.Laplace2D()
    """

    def __init__(self, dim=2, weight=1.0):
        super(Laplace, self).__init__(1, weight=weight)

        if dim == 2:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable
            u = sympy.Function('u')(x, y)

            # variables in order
            self.independent_variable = [x, y]
            self.dependent_variable = [u]

            # order
            self.order = 2

            # equations and rhs
            self.equations = [u.diff(x).diff(x) + u.diff(y).diff(y)]
            self.rhs = 0

        elif dim == 3:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            # dependent variable
            u = sympy.Function('u')(x, y, z)

            # variables in order
            self.independent_variable = [x, y, z]
            self.dependent_variable = [u]

            # order
            self.order = 2

            # equations and rhs
            self.equations = [
                u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z)
            ]
            self.rhs = 0
