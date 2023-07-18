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

from typing import Callable
from typing import Union
from ppsci.utils.paddle_printer import SympyToPaddle

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base
from sympy import Symbol, Function


class NavierStokes(base.PDE):
    r"""Class for navier-stokes equation(Incompressible).

    $$
    \begin{cases}
        \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z} = 0 \\
        \dfrac{\partial u}{\partial t} + u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} + w\dfrac{\partial w}{\partial z} =
            - \dfrac{1}{\rho}\dfrac{\partial p}{\partial x}
            + \nu(
                \dfrac{\partial ^2 u}{\partial x ^2}
                + \dfrac{\partial ^2 u}{\partial y ^2}
                + \dfrac{\partial ^2 u}{\partial z ^2}
            ) \\
        \dfrac{\partial v}{\partial t} + u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} + w\dfrac{\partial w}{\partial z} =
            - \dfrac{1}{\rho}\dfrac{\partial p}{\partial y}
            + \nu(
                \dfrac{\partial ^2 v}{\partial x ^2}
                + \dfrac{\partial ^2 v}{\partial y ^2}
                + \dfrac{\partial ^2 v}{\partial z ^2}
            ) \\
        \dfrac{\partial w}{\partial t} + u\dfrac{\partial w}{\partial x} + v\dfrac{\partial w}{\partial y} + w\dfrac{\partial w}{\partial z} =
            - \dfrac{1}{\rho}\dfrac{\partial p}{\partial z}
            + \nu(
                \dfrac{\partial ^2 w}{\partial x ^2}
                + \dfrac{\partial ^2 w}{\partial y ^2}
                + \dfrac{\partial ^2 w}{\partial z ^2}
            ) \\
    \end{cases}
    $$

    Args:
        nu (Union[float, Callable]): Dynamic viscosity.
        rho (float): Density.
        dim (int): Dimension of equation.
        time (bool): Whether the euqation is time-dependent.
        sympy_expression (bool): Whether the euqation is expressed by sympy(It means paddle_printer.py will be used).

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.NavierStokes(0.1, 1.0, 3, False)
    """

    def __init__(self, nu: Union[float, Callable], rho: float, dim: int, time: bool, sympy_expression: bool=False):
        super().__init__()
        self.nu = nu
        self.rho = rho
        self.dim = dim
        self.time = time

        def continuity_compute_func(out):
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            continuity = jacobian(u, x) + jacobian(v, y)
            if self.dim == 3:
                z, w = out["z"], out["w"]
                continuity += jacobian(w, z)
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def momentum_x_compute_func(out):
            nu = self.nu(out) if callable(self.nu) else self.nu
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_x = (
                + u * jacobian(u, x)
                + v * jacobian(u, y)
                - nu * hessian(u, x)
                - nu * hessian(u, y)
                + 1 / rho * jacobian(p, x)
            )
            if self.time:
                t = out["t"]
                momentum_x += jacobian(u, t)
            if self.dim == 3:
                z, w = out["z"], out["w"]
                momentum_x += w * jacobian(u, z)
                momentum_x -= nu  * hessian(u, z)
            return momentum_x

        if sympy_expression is False:
            self.add_equation("momentum_x", momentum_x_compute_func)

        def momentum_y_compute_func(out):
            nu = self.nu(out) if callable(self.nu) else self.nu
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_y = (
                + u * jacobian(v, x)
                + v * jacobian(v, y)
                - nu  * hessian(v, x)
                - nu  * hessian(v, y)
                + 1 / rho * jacobian(p, y)
            )
            if self.time:
                t = out["t"]
                momentum_y += jacobian(v, t)
            if self.dim == 3:
                z, w = out["z"], out["w"]
                momentum_y += w * jacobian(v, z)
                momentum_y -= nu / rho * hessian(v, z)
            return momentum_y

        if sympy_expression is False:
            self.add_equation("momentum_y", momentum_y_compute_func)


        def momentum_x_symbol_func():
            x, y = Symbol("x"), Symbol("y")
            input_variables = {"x": x, "y": y}
            u = Function("u")(*input_variables)
            v = Function("v")(*input_variables)
            p = Function("p")(*input_variables)
            nu = Function("nu")(*input_variables) if callable(self.nu) else self.nu
            sympy_expr = (
                + u * (u.diff(x))
                + v * (u.diff(y)) 
                + 1 / rho * p.diff(x) 
                - (nu * u.diff(x)).diff(x) 
                - (nu * u.diff(y)).diff(y)
            )

            if self.time:
                t = Symbol("t")
                input_variables.update({"t", t})
                sympy_expr += u.diff(t)

            if self.dim == 3:
                z = Symbol("z")
                input_variables.update({"z", z})
                w = Function("w")(*input_variables)
                sympy_expr += w * u.diff(z)
                sympy_expr -= (nu * u.diff(z)).diff(z)
            
            return SympyToPaddle(sympy_expr, "momentum_x")

        if sympy_expression is True:
            self.add_equation("momentum_x", momentum_x_symbol_func())


        def momentum_y_symbol_func():
            x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
            input_variables = {"x": x, "y": y}
            u = Function("u")(*input_variables)
            v = Function("v")(*input_variables)
            p = Function("p")(*input_variables)
            nu = Function("nu")(*input_variables) if callable(self.nu) else self.nu
            sympy_expr = (
                + u * v.diff(x)
                + v * v.diff(y)
                + 1 / rho * p.diff(y)
                - (nu * v.diff(x)).diff(x)
                - (nu * v.diff(y)).diff(y)
            )

            if self.time:
                t = Symbol("t")
                input_variables.update({"t", t})
                sympy_expr += v.diff(t)

            if self.dim == 3:
                z = Symbol("z")
                input_variables.update({"z", z})
                w = Function("w")(*input_variables)
                sympy_expr += w * v.diff(z)
                sympy_expr -= (nu * v.diff(z)).diff(z)
            
            return SympyToPaddle(sympy_expr, "momentum_y")

        if sympy_expression is True:
            self.add_equation("momentum_y", momentum_y_symbol_func())
        

        if self.dim == 3:

            def momentum_z_compute_func(out):
                nu = self.nu(out) if callable(self.nu) else self.nu
                x, y, z = out["x"], out["y"], out["z"]
                u, v, w, p = out["u"], out["v"], out["w"], out["p"]
                momentum_z = (
                    + u * jacobian(w, x)
                    + v * jacobian(w, y)
                    + w * jacobian(w, z)
                    - nu / rho * hessian(w, x)
                    - nu / rho * hessian(w, y)
                    - nu / rho * hessian(w, z)
                    + 1 / rho * jacobian(p, z)
                )
                if self.time:
                    t = out["t"]
                    momentum_z += jacobian(w, t)
                return momentum_z

            if sympy_expression is False:
                self.add_equation("momentum_z", momentum_z_compute_func)
