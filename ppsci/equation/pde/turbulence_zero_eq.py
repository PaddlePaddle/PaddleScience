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
from ppsci.utils.paddle_printer import SympyToPaddle
import numpy as np
import paddle
from sympy import Symbol, sqrt, pprint, diff, simplify, Function, Min

class ZeroEquation(base.PDE):
    def __init__(
        self, nu, max_distance, rho=1, dim=3
    ):  # TODO add density into model
        # set params
        super().__init__()
        self.dim = dim
        self.nu = nu
        self.expr = None
        # model coefficients
        self.max_distance = max_distance
        self.karman_constant = 0.419
        self.max_distance_ratio = 0.09
        
        def nu_equation(out):
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            sdf = out["sdf"]
            normal_distance = sdf
            # mixing length
            mixing_length = np.min(
                self.karman_constant * normal_distance,
                self.max_distance_ratio * self.max_distance,
            )
            G = (
                2 * jacobian(u, x) ** 2
                + 2 * jacobian(v, y) ** 2
                + (u.diff(y) + v.diff(x)) ** 2
            )
            return nu + rho * mixing_length**2 * paddle.sqrt(G)

        # self.add_equation("nu", nu_equation)

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
            G = (
                2 * u.diff(x) ** 2
                + 2 * v.diff(y) ** 2
                + (u.diff(y) + v.diff(x)) ** 2
            )

            # sympy_expr = nu + rho * mixing_length**2 * sqrt(G)
            sympy_expr = mixing_length**2
            self.expr = sympy_expr
            return SympyToPaddle(sympy_expr, "nu_symbol")

        self.add_equation("nu_symbol", nu_symbol())
            