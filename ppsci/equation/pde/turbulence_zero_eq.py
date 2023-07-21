# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sympy import Function
from sympy import Min
from sympy import Symbol
from sympy import sqrt

from ppsci.equation.pde import base
from ppsci.utils.paddle_printer import SympyToPaddle


class ZeroEquation(base.PDE):
    r"""Prandtl's mixing-length model
    
    $$
    \begin{cases}
        \nu_t = \rho l_{m}^2 G \\
        G = \left[
                \dfrac{\partial U_i}{\partial x_j} 
            \left(
                \dfrac{\partial U_i}{\partial x_j} + 
                \dfrac{\partial U_j}{\partial x_i}
            \right) \right]^{1/2} \\
        l_m = min \left(karman_constant * normal_distance, max_distance_ratio * max_distance) \\
    \end{cases}
    $$

    Args:
        nu (float/Callable): The dynamic viscosity of the fluid. If Callable, it can accept x and y as input and output the drag coefficient value.
        max_distance (float): The maximum distance used to calculate the mixing length.
        rho (float, optional): Density, default is 1.0.
        dim (int, optional): The dimension of the computational domain, default is 3.

    Returns:
        Dynamic Eddy-Viscosity + Dynamic Fluid Viscosity

    Examples:
        >>> import ppsci
        >>> ppsci.equation.ZeroEquation(0.01, 1)
    """

    def __init__(self, nu, max_distance, rho=1.0, dim=3):
        # set params
        super().__init__()
        self.dim = dim
        self.nu = nu
        self.expr = None

        # model coefficients
        self.max_distance = max_distance
        self.karman_constant = 0.419
        self.max_distance_ratio = 0.09

        def nu_symbol():
            # make input variables
            x, y = Symbol("x"), Symbol("y")
            input_variables = {"x": x, "y": y}
            u = Function("u")(*input_variables)
            v = Function("v")(*input_variables)
            nu = Function("nu")(*input_variables) if callable(self.nu) else self.nu
            normal_distance = Function("sdf")(*input_variables)

            mixing_length = Min(
                self.karman_constant * normal_distance,
                self.max_distance_ratio * self.max_distance,
            )

            G = 2 * u.diff(x) ** 2 + 2 * v.diff(y) ** 2 + (u.diff(y) + v.diff(x)) ** 2

            sympy_expr = nu + rho * mixing_length**2 * sqrt(G)
            self.expr = sympy_expr
            return SympyToPaddle(sympy_expr, "nu_symbol")

        self.add_equation("nu_symbol", nu_symbol())
