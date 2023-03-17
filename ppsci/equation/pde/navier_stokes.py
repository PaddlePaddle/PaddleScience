"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sympy
from sympy.parsing import sympy_parser as sp_parser

from ppsci.equation.pde import base
from ppsci.gradient import hessian
from ppsci.gradient import jacobian


class NavierStokes(base.PDE):
    """NavierStokes

    Args:
        nu (float): Dynamic viscosity.
        rho (float): Density.
        dim (int): Dimension of equation.
        time (bool): Whether the euqation is time-dependent.
    """

    def __init__(self, nu, rho, dim, time):
        super().__init__()

        # Sympy Code
        if isinstance(nu, float):
            nu = sympy.Number(nu)
        elif isinstance(nu, str):
            nu = sp_parser.parse_expr(nu)
        else:
            raise ValueError(f"type of nu({type(nu)}) must be float or str")

        if isinstance(rho, float):
            rho = sympy.Number(rho)
        elif isinstance(rho, str):
            rho = sp_parser.parse_expr(rho)
        else:
            raise ValueError(f"type of rho({type(rho)}) must be float or str")

        # independent variable
        t, x, y, z = self.create_symbols("t x y z")
        invars = [x, y, z][:dim]
        if time:
            invars = [t] + invars

        # dependent variable
        u = self.create_function("u", invars)
        v = self.create_function("v", invars)
        w = self.create_function("w", invars) if dim == 3 else sympy.Number(0)
        p = self.create_function("p", invars)

        # continuity equation
        continuity = u.diff(x) + v.diff(y) + w.diff(z)

        # momentum equation
        momentum_x = (
            u.diff(t)
            + u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)
            - nu / rho * u.diff(x, 2)
            - nu / rho * u.diff(y, 2)
            - nu / rho * u.diff(z, 2)
            + 1.0 / rho * p.diff(x)
        )

        momentum_y = (
            v.diff(t)
            + u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)
            - nu / rho * v.diff(x, 2)
            - nu / rho * v.diff(y, 2)
            - nu / rho * v.diff(z, 2)
            + 1.0 / rho * p.diff(y)
        )

        momentum_z = (
            w.diff(t)
            + u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            - nu / rho * w.diff(x, 2)
            - nu / rho * w.diff(y, 2)
            - nu / rho * w.diff(z, 2)
            + 1.0 / rho * p.diff(z)
        )

        self.equations["continuity"] = continuity
        self.equations["momentum_x"] = momentum_x
        self.equations["momentum_y"] = momentum_y
        if dim == 3:
            self.equations["momentum_z"] = momentum_z
        # =======

        # Paddle API Code
        # def continuity(d):
        #     x, y = d["x"], d["y"]
        #     u, v = d["u"], d["v"]

        #     continuity = jacobian(u, x) + jacobian(v, y)
        #     return continuity

        # def momentum_x(d):
        #     t, x, y = d["t"], d["x"], d["y"]
        #     u, v, p = d["u"], d["v"], d["p"]
        #     momentum_x = jacobian(u, t) + u * jacobian(u, x) + v * jacobian(u, y) - \
        #         nu / rho * hessian(u, x) - nu / rho * hessian(u, y) + 1.0 / rho * jacobian(p, x)
        #     return momentum_x

        # def momentum_y(d):
        #     t, x, y = d["t"], d["x"], d["y"]
        #     u, v, p = d["u"], d["v"], d["p"]
        #     momentum_y = jacobian(v, t) + u * jacobian(v, x) + v * jacobian(v, y) - \
        #         nu / rho * hessian(v, x) - nu / rho * hessian(v, y) + 1.0 / rho * jacobian(p, y)
        #     return momentum_y

        # self.equations["continuity"] = continuity
        # self.equations["momentum_x"] = momentum_x
        # self.equations["momentum_y"] = momentum_y
        # # =======
