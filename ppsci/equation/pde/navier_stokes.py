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

from __future__ import annotations

from typing import Callable
from typing import Union

import sympy as sp

from ppsci.equation.pde import base


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

        t, x, y, z = self.create_symbols("t x y z")
        u, v, w, p = self.create_symbols("u v w p")
        if self.dim == 2:
            w = sp.Number(0)
        if not time:
            t = sp.Number(0)

        continuity = u.diff(x) + v.diff(y) + w.diff(z)
        momentum_x = (
            u.diff(t)
            + u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)
            - nu * (u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z))
            + 1 / rho * p.diff(x)
        )
        momentum_y = (
            u.diff(t)
            + u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)
            - nu * (v.diff(x).diff(x) + v.diff(y).diff(y) + v.diff(z).diff(z))
            + 1 / rho * p.diff(y)
        )
        momentum_z = (
            u.diff(t)
            + u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            - nu * (w.diff(x).diff(x) + w.diff(y).diff(y) + w.diff(z).diff(z))
            + 1 / rho * p.diff(z)
        )
        self.add_equation("continuity", continuity)
        self.add_equation("momentum_x", momentum_x)
        self.add_equation("momentum_y", momentum_y)
        self.add_equation("momentum_z", momentum_z)
