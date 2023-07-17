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
    r"""Class for navier-stokes equation.

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

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.NavierStokes(0.1, 1.0, 3, False)
    """

    def __init__(self, nu: Union[float, Callable], rho: float, dim: int, time: bool):
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
                u * jacobian(u, x)
                + v * jacobian(u, y)
                - nu / rho * hessian(u, x)
                - nu / rho * hessian(u, y)
                + 1 / rho * jacobian(p, x)
            )
            if self.time:
                t = out["t"]
                momentum_x += jacobian(u, t)
            if self.dim == 3:
                z, w = out["z"], out["w"]
                momentum_x += w * jacobian(u, z)
                momentum_x -= nu / rho * hessian(u, z)
            return momentum_x

        # self.add_equation("momentum_x", momentum_x_compute_func)

        def momentum_y_compute_func(out):
            nu = self.nu(out) if callable(self.nu) else self.nu
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_y = (
                u * jacobian(v, x)
                + v * jacobian(v, y)
                - nu / rho * hessian(v, x)
                - nu / rho * hessian(v, y)
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

        # self.add_equation("momentum_y", momentum_y_compute_func)


        def momentum_x_symbol_func():
            x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
            input_variables = {"x": x, "y": y}
            u = Function("u")(*input_variables)
            v = Function("v")(*input_variables)
            p = Function("p")(*input_variables)
            nu = Function("nu")(*input_variables) if callable(self.nu) else self.nu
            mu = nu * rho
            sympy_expr = (
            u * ((rho * u).diff(x)) +
            v * ((rho * u).diff(y)) 
            + p.diff(x) 
            - (mu * u.diff(x)).diff(x) 
            - (mu * u.diff(y)).diff(y))
            
            return SympyToPaddle(sympy_expr, "momentum_x")
        self.add_equation("momentum_x", momentum_x_symbol_func())


        def momentum_y_symbol_func():
            x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
            input_variables = {"x": x, "y": y}
            u = Function("u")(*input_variables)
            v = Function("v")(*input_variables)
            p = Function("p")(*input_variables)
            nu = Function("nu")(*input_variables) if callable(self.nu) else self.nu
            mu = nu * rho
            sympy_expr = (
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                )
                + p.diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
            )
            
            return SympyToPaddle(sympy_expr, "momentum_y")
        
        self.add_equation("momentum_y", momentum_y_symbol_func())
        

        if self.dim == 3:

            def momentum_z_compute_func(out):
                nu = self.nu(out) if callable(self.nu) else self.nu
                x, y, z = out["x"], out["y"], out["z"]
                u, v, w, p = out["u"], out["v"], out["w"], out["p"]
                momentum_z = (
                    u * jacobian(w, x)
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

            self.add_equation("momentum_z", momentum_z_compute_func)
