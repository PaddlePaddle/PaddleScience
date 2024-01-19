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

from typing import Optional
from typing import Tuple
from typing import Union

import sympy as sp

from ppsci.equation.pde import base


class LinearElasticity(base.PDE):
    r"""Linear elasticity equations.
    Use either (E, nu) or (lambda_, mu) to define the material properties.

    $$
    \begin{cases}
        stress\_disp_{xx} = \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z}) + 2\mu \dfrac{\partial u}{\partial x} - \sigma_{xx} \\
        stress\_disp_{yy} = \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z}) + 2\mu \dfrac{\partial v}{\partial y} - \sigma_{yy} \\
        stress\_disp_{zz} = \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z}) + 2\mu \dfrac{\partial w}{\partial z} - \sigma_{zz} \\
        stress\_disp_{xy} = \mu(\dfrac{\partial u}{\partial y} + \dfrac{\partial v}{\partial x}) - \sigma_{xy} \\
        stress\_disp_{xz} = \mu(\dfrac{\partial u}{\partial z} + \dfrac{\partial w}{\partial x}) - \sigma_{xz} \\
        stress\_disp_{yz} = \mu(\dfrac{\partial v}{\partial z} + \dfrac{\partial w}{\partial y}) - \sigma_{yz} \\
        equilibrium_{x} = \rho \dfrac{\partial^2 u}{\partial t^2} - (\dfrac{\partial \sigma_{xx}}{\partial x} + \dfrac{\partial \sigma_{xy}}{\partial y} + \dfrac{\partial \sigma_{xz}}{\partial z}) \\
        equilibrium_{y} = \rho \dfrac{\partial^2 u}{\partial t^2} - (\dfrac{\partial \sigma_{xy}}{\partial x} + \dfrac{\partial \sigma_{yy}}{\partial y} + \dfrac{\partial \sigma_{yz}}{\partial z}) \\
        equilibrium_{z} = \rho \dfrac{\partial^2 u}{\partial t^2} - (\dfrac{\partial \sigma_{xz}}{\partial x} + \dfrac{\partial \sigma_{yz}}{\partial y} + \dfrac{\partial \sigma_{zz}}{\partial z}) \\
    \end{cases}
    $$

    Args:
        E (Optional[Union[float, str]]): The Young's modulus. Defaults to None.
        nu (Optional[Union[float, str]]): The Poisson's ratio. Defaults to None.
        lambda_ (Optional[Union[float, str]]): Lamé's first parameter. Defaults to None.
        mu (Optional[Union[float, str]]): Lamé's second parameter (shear modulus). Defaults to None.
        rho (Union[float, str], optional): Mass density. Defaults to 1.
        dim (int, optional): Dimension of the linear elasticity (2 or 3). Defaults to 3.
        time (bool, optional): Whether contains time data. Defaults to False.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.LinearElasticity(
        ...     E=None, nu=None, lambda_=1e4, mu=100, dim=3
        ... )
    """

    def __init__(
        self,
        E: Optional[Union[float, str]] = None,
        nu: Optional[Union[float, str]] = None,
        lambda_: Optional[Union[float, str]] = None,
        mu: Optional[Union[float, str]] = None,
        rho: Union[float, str] = 1,
        dim: int = 3,
        time: bool = False,
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys
        self.dim = dim
        self.time = time

        t, x, y, z = self.create_symbols("t x y z")
        normal_x, normal_y, normal_z = self.create_symbols("normal_x normal_y normal_z")
        invars = (x, y)
        if time:
            invars = (t,) + invars
        if self.dim == 3:
            invars += (z,)

        u = self.create_function("u", invars)
        v = self.create_function("v", invars)
        w = self.create_function("w", invars) if dim == 3 else sp.Number(0)

        sigma_xx = self.create_function("sigma_xx", invars)
        sigma_yy = self.create_function("sigma_yy", invars)
        sigma_xy = self.create_function("sigma_xy", invars)
        sigma_zz = (
            self.create_function("sigma_zz", invars) if dim == 3 else sp.Number(0)
        )
        sigma_xz = (
            self.create_function("sigma_xz", invars) if dim == 3 else sp.Number(0)
        )
        sigma_yz = (
            self.create_function("sigma_yz", invars) if dim == 3 else sp.Number(0)
        )

        # compute lambda and mu
        if lambda_ is None:
            if isinstance(nu, str):
                nu = self.create_function(nu, invars)
            if isinstance(E, str):
                E = self.create_function(E, invars)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = self.create_function(lambda_, invars)
            if isinstance(mu, str):
                mu = self.create_function(mu, invars)

        if isinstance(rho, str):
            rho = self.create_function(rho, invars)

        self.E = E
        self.nu = nu
        self.lambda_ = lambda_
        self.mu = mu
        self.rho = rho

        # compute stress equations
        stress_disp_xx = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * u.diff(x)
            - sigma_xx
        )
        stress_disp_yy = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * v.diff(y)
            - sigma_yy
        )
        stress_disp_zz = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * w.diff(z)
            - sigma_zz
        )
        stress_disp_xy = mu * (u.diff(y) + v.diff(x)) - sigma_xy
        stress_disp_xz = mu * (u.diff(z) + w.diff(x)) - sigma_xz
        stress_disp_yz = mu * (v.diff(z) + w.diff(y)) - sigma_yz

        # compute equilibrium equations
        equilibrium_x = rho * ((u.diff(t)).diff(t)) - (
            sigma_xx.diff(x) + sigma_xy.diff(y) + sigma_xz.diff(z)
        )
        equilibrium_y = rho * ((v.diff(t)).diff(t)) - (
            sigma_xy.diff(x) + sigma_yy.diff(y) + sigma_yz.diff(z)
        )
        equilibrium_z = rho * ((w.diff(t)).diff(t)) - (
            sigma_xz.diff(x) + sigma_yz.diff(y) + sigma_zz.diff(z)
        )

        # compute traction equations
        traction_x = normal_x * sigma_xx + normal_y * sigma_xy + normal_z * sigma_xz
        traction_y = normal_x * sigma_xy + normal_y * sigma_yy + normal_z * sigma_yz
        traction_z = normal_x * sigma_xz + normal_y * sigma_yz + normal_z * sigma_zz

        # add stress equations
        self.add_equation("stress_disp_xx", stress_disp_xx)
        self.add_equation("stress_disp_yy", stress_disp_yy)
        self.add_equation("stress_disp_xy", stress_disp_xy)
        if self.dim == 3:
            self.add_equation("stress_disp_zz", stress_disp_zz)
            self.add_equation("stress_disp_xz", stress_disp_xz)
            self.add_equation("stress_disp_yz", stress_disp_yz)

        # add equilibrium equations
        self.add_equation("equilibrium_x", equilibrium_x)
        self.add_equation("equilibrium_y", equilibrium_y)
        if self.dim == 3:
            self.add_equation("equilibrium_z", equilibrium_z)

        # add traction equations
        self.add_equation("traction_x", traction_x)
        self.add_equation("traction_y", traction_y)
        if self.dim == 3:
            self.add_equation("traction_z", traction_z)
