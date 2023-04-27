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

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class NavierStokes(base.PDE):
    """Class for navier-stokes equation.

    Args:
        nu (float): Dynamic viscosity.
        rho (float): Density.
        dim (int): Dimension of equation.
        time (bool): Whether the euqation is time-dependent.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.NavierStokes(0.1, 1.0, 3, False)
    """

    def __init__(self, nu: float, rho: float, dim: int, time: bool):
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
                z = out["z"]
                w = out["w"]
                continuity += jacobian(w, z)
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def momentum_x_compute_func(out):
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
                z = out["z"]
                w = out["w"]
                momentum_x += w * jacobian(u, z)
                momentum_x -= nu / rho * hessian(u, z)
            return momentum_x

        self.add_equation("momentum_x", momentum_x_compute_func)

        def momentum_y_compute_func(out):
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
                z = out["z"]
                w = out["w"]
                momentum_y += w * jacobian(v, z)
                momentum_y -= nu / rho * hessian(v, z)
            return momentum_y

        self.add_equation("momentum_y", momentum_y_compute_func)

        if self.dim == 3:

            def momentum_z_compute_func(out):
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
