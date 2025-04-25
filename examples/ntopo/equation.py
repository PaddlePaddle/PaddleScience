# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
from typing import Union

import numpy as np

from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class EEquation(base.PDE):
    r"""Linear elasticity equations.
    Use either (E, nu) or (lambda_, mu) to define the material properties.

    2D:
    $$
    \begin{cases}
        \sigma_{xx} = \mu \dfrac{\partial u}{\partial x} + \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y})
        \sigma_{xy} = \dfrac{\mu}{2}(\dfrac{\partial u}{\partial y} + \dfrac{\partial v}{\partial x})
        \sigma_{yy} = \mu \dfrac{\partial v}{\partial y} + \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y})
        E_{xy} = \dfrac{1}{2}(\dfrac{\partial u}{\partial x}\sigma_{xx} + (\dfrac{\partial u}{\partial y} + \dfrac{\partial v}{\partial x})\sigma_{xy} + \dfrac{\partial v}{\partial y}\sigma_{yy}) \\
    \end{cases}
    $$

    3D:
    $$
    \begin{cases}
        \epsilon_{xy} = \dfrac{1}{2}(\dfrac{\partial u}{\partial y} + \dfrac{\partial v}{\partial x})
        \epsilon_{xz} = \dfrac{1}{2}(\dfrac{\partial u}{\partial z} + \dfrac{\partial w}{\partial x})
        \epsilon_{zy} = \dfrac{1}{2}(\dfrac{\partial v}{\partial z} + \dfrac{\partial w}{\partial y})
        E_{xyz} = \dfrac{\lambda}{2}(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z})^2 + \mu (\dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2} + \dfrac{\partial^2 w}{\partial z^2}) + 2\mu (\epsilon_{xy}^2 + \epsilon_{xz}^2 + \epsilon_{yz}^2) \\
    \end{cases}
    $$

    Args:
        E (Optional[float]): The Young's modulus. Defaults to None.
        nu (Optional[float]): The Poisson's ratio. Defaults to None.
        lambda_ (Optional[float]): Lamé's first parameter. Defaults to None.
        mu (Optional[float]): Lamé's second parameter (shear modulus). Defaults to None.
        rho (float, optional): Mass density. Defaults to 1.
        dim (int, optional): Dimension of the linear elasticity (2 or 3). Defaults to 3.
        time (bool, optional): Whether contains time data. Defaults to False.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.LinearElasticity(
        ...     E=None, nu=None, lambda_=1e4, mu=100, dim=3
        ... )
    """

    def __init__(
        self,
        param_dict: Dict[str, Union[float, np.ndarray]],
        dim: int = 3,
        time: bool = False,
    ):
        super().__init__()
        self.param_keys = (
            "E",
            "nu",
            "lambda_",
            "mu",
            "u__x",
            "u__y",
            "u__z",
            "v__x",
            "v__y",
            "v__z",
            "w__x",
            "w__y",
            "w__z",
        )
        self.dim = dim
        self.time = time
        self.param_dict = param_dict
        for key in self.param_keys:
            if key in self.param_dict:
                setattr(self, key, self.param_dict[key])

        def init_params(out):
            for key in self.param_keys:
                if key not in self.param_dict:
                    setattr(self, key, out[key] if key in out else None)

            if self.u__x is None:
                self.u__x = jacobian(out["u"], out["x"])
            if self.u__y is None:
                self.u__y = jacobian(out["u"], out["y"])
            if self.v__x is None:
                self.v__x = jacobian(out["v"], out["x"])
            if self.v__y is None:
                self.v__y = jacobian(out["v"], out["y"])

            if self.dim == 3:
                if self.u__z is None:
                    self.u__z = jacobian(out["u"], out["z"])
                if self.v__z is None:
                    self.v__z = jacobian(out["v"], out["z"])
                if self.w__x is None:
                    self.w__x = jacobian(out["w"], out["x"])
                if self.w__y is None:
                    self.w__y = jacobian(out["w"], out["y"])
                if self.w__z is None:
                    self.w__z = jacobian(out["w"], out["z"])

        # Energy equations
        def E_xy_compute_func(out):
            init_params(out)
            sigma_xx = self.mu * self.u__x + self.lambda_ * (self.u__x + self.v__y)
            sigma_xy = 0.5 * self.mu * (self.u__y + self.v__x)
            sigma_yy = self.mu * self.v__y + self.lambda_ * (self.u__x + self.v__y)
            E_xy = 0.5 * (
                self.u__x * sigma_xx
                + (self.u__y + self.v__x) * sigma_xy
                + self.v__y * sigma_yy
            )
            return E_xy

        self.add_equation("E_xy", E_xy_compute_func)

        if self.dim == 3:

            def E_xyz_compute_func(out):
                init_params(out)
                eps12 = 0.5 * self.u__y + 0.5 * self.v__x
                eps13 = 0.5 * self.u__z + 0.5 * self.w__x
                eps23 = 0.5 * self.v__z + 0.5 * self.w__y

                trace_strain = self.u__x + self.v__y + self.w__z
                squared_diagonal = (
                    self.u__x * self.u__x
                    + self.v__y * self.v__y
                    + self.w__z * self.w__z
                )
                E_xyz = 0.5 * self.lambda_ * trace_strain * trace_strain + self.mu * (
                    squared_diagonal
                    + 2.0 * eps12 * eps12
                    + 2.0 * eps13 * eps13
                    + 2.0 * eps23 * eps23
                )
                return E_xyz

            self.add_equation("E_xyz", E_xyz_compute_func)
