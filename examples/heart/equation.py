# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

import paddle
import sympy as sp

from ppsci.equation.pde import base


class Hooke(base.PDE):
    r"""equations for umbrella opening force.
    Use either (E, nu) or (lambda_, mu) to define the material properties.

    $$
    \begin{pmatrix}
        t_{xx} \\ t_{yy} \\ t_{zz} \\ t_{xy} \\ t_{xz} \\ t_{yz} \\
    \end{pmatrix}
    =
    \begin{bmatrix}
        \frac{1}{E} & -\frac{\nu}{E} & -\frac{\nu}{E} & 0 & 0 & 0 \\
        -\frac{\nu}{E} & \frac{1}{E} & -\frac{\nu}{E} & 0 & 0 & 0 \\
        -\frac{\nu}{E} & -\frac{\nu}{E} & \frac{1}{E} & 0 & 0 & 0 \\
        0 & 0 & 0 & \frac{1}{G} & 0 & 0 \\
        0 & 0 & 0 & 0 & \frac{1}{G} & 0 \\
        0 & 0 & 0 & 0 & 0 & \frac{1}{G}  \\
    \end{bmatrix}
    \begin{pmatrix}
        \varepsilon _{xx} \\ \varepsilon _{yy} \\ \varepsilon _{zz} \\ \varepsilon _{xy} \\ \varepsilon _{xz} \\ \varepsilon _{yz} \\
    \end{pmatrix}
    $$

    Args:
        E (paddle.base.framework.EagerParamBase): The Young's modulus. Learnable parameter.
        nu (Union[float, str]): The Poisson's ratio.
        P (Union[float, str]): Left ventricular cavity pressure.
        dim (int, optional): Dimension of the linear elasticity (2 or 3). Defaults to 3.
        time (bool, optional): Whether contains time data. Defaults to False.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.


    Examples:
        >>> import ppsci
        >>> E = paddle.create_parameter(
        ...     shape=[],
        ...     dtype=paddle.get_default_dtype(),
        ...     default_initializer=initializer.Constant(),
        ... )
        >>> pde = ppsci.equation.Hooke(
        ...     E=E, nu=cfg.nu, P=cfg.P, dim=3
        ... )
    """

    def __init__(
        self,
        E: Union[float, str, paddle.base.framework.EagerParamBase],
        nu: Union[float, str],
        P: Union[float, str],
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

        if isinstance(nu, str):
            nu = self.create_function(nu, invars)
        if isinstance(P, str):
            P = self.create_function(P, invars)
        if isinstance(E, str):
            E = self.create_function(E, invars)
            self.E = E
        elif isinstance(E, paddle.base.framework.EagerParamBase):
            self.E = E
            self.learnable_parameters.append(self.E)
            E = self.create_symbols(self.E.name)

        self.nu = nu
        self.P = P

        # compute sigma
        sigma_xx = u.diff(x)
        sigma_yy = v.diff(y)
        sigma_zz = w.diff(z) if dim == 3 else sp.Number(0)
        sigma_xy = 0.5 * (u.diff(y) + v.diff(x))
        sigma_xz = 0.5 * (u.diff(z) + w.diff(x)) if dim == 3 else sp.Number(0)
        sigma_yz = 0.5 * (v.diff(z) + w.diff(y)) if dim == 3 else sp.Number(0)

        # compute stress tensor t
        G = E / (2 * (1 + nu))
        e = sigma_xx + sigma_yy + sigma_zz
        t_xx = 2 * G * (sigma_xx + nu / (1 - 2 * nu) * e)
        t_yy = 2 * G * (sigma_yy + nu / (1 - 2 * nu) * e)
        t_zz = 2 * G * (sigma_zz + nu / (1 - 2 * nu) * e)
        t_xy = 2 * sigma_xy * G
        t_xz = 2 * sigma_xz * G
        t_yz = 2 * sigma_yz * G

        # compute stress
        hooke_x = t_xx.diff(x) + t_xy.diff(y) + t_xz.diff(z)
        hooke_y = t_xy.diff(x) + t_yy.diff(y) + t_yz.diff(z)
        hooke_z = t_xz.diff(x) + t_yz.diff(y) + t_zz.diff(z)

        # compute traction splitly
        traction_x = t_xx * normal_x + t_xy * normal_y + t_xz * normal_z + P * normal_x
        traction_y = t_xy * normal_x + t_yy * normal_y + t_yz * normal_z + P * normal_y
        traction_z = t_xz * normal_x + t_yz * normal_y + t_zz * normal_z + P * normal_z

        # compute traction
        traction_x_ = t_xx * normal_x + t_xy * normal_y + t_xz * normal_z
        traction_y_ = t_xy * normal_x + t_yy * normal_y + t_yz * normal_z
        traction_z_ = t_xz * normal_x + t_yz * normal_y + t_zz * normal_z

        traction = (
            traction_x_ * normal_x + traction_y_ * normal_y + traction_z_ * normal_z
        )

        # add hooke equations
        self.add_equation("hooke_x", hooke_x)
        self.add_equation("hooke_y", hooke_y)
        if self.dim == 3:
            self.add_equation("hooke_z", hooke_z)

        # add traction equations
        self.add_equation("traction_x", traction_x)
        self.add_equation("traction_y", traction_y)
        if self.dim == 3:
            self.add_equation("traction_z", traction_z)

        # add combined traction equations
        self.add_equation("traction", traction)

        self._apply_detach()
