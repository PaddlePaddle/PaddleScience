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

import paddle
import pytest
from sympy import Function
from sympy import Number
from sympy import Symbol

import ppsci
from ppsci import equation
from ppsci.autodiff import clear
from ppsci.utils import expression

__all__ = []


class NavierStokes_sympy:
    def __init__(self, nu, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
        self.equations["momentum_x"] = (
            (rho * u).diff(t)
            + (
                u * ((rho * u).diff(x))
                + v * ((rho * u).diff(y))
                + w * ((rho * u).diff(z))
                + rho * u * (curl)
            )
            + p.diff(x)
            - (-2 / 3 * mu * (curl)).diff(x)
            - (mu * u.diff(x)).diff(x)
            - (mu * u.diff(y)).diff(y)
            - (mu * u.diff(z)).diff(z)
            - (mu * (curl).diff(x))
        )
        self.equations["momentum_y"] = (
            (rho * v).diff(t)
            + (
                u * ((rho * v).diff(x))
                + v * ((rho * v).diff(y))
                + w * ((rho * v).diff(z))
                + rho * v * (curl)
            )
            + p.diff(y)
            - (-2 / 3 * mu * (curl)).diff(y)
            - (mu * v.diff(x)).diff(x)
            - (mu * v.diff(y)).diff(y)
            - (mu * v.diff(z)).diff(z)
            - (mu * (curl).diff(y))
        )
        self.equations["momentum_z"] = (
            (rho * w).diff(t)
            + (
                u * ((rho * w).diff(x))
                + v * ((rho * w).diff(y))
                + w * ((rho * w).diff(z))
                + rho * w * (curl)
            )
            + p.diff(z)
            - (-2 / 3 * mu * (curl)).diff(z)
            - (mu * w.diff(x)).diff(x)
            - (mu * w.diff(y)).diff(y)
            - (mu * w.diff(z)).diff(z)
            - (mu * (curl).diff(z))
        )

        if self.dim == 2:
            self.equations.pop("momentum_z")


@pytest.mark.parametrize("nu", (2.0,))
@pytest.mark.parametrize("rho", (1.0,))
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("time", (False, True))
def test_navier_stokes(nu, rho, dim, time):
    """Test for navier_stokes equation."""
    # define input/output keys
    input_keys = ("x", "y", "z")[:dim]
    if time:
        input_keys = ("t",) + input_keys

    output_keys = ("u", "v")
    if dim == 3:
        output_keys += ("w",)
    output_keys += ("p",)

    # prepare input data in dict
    batch_size = 13
    input_dict = {}
    for var in input_keys:
        input_dict[var] = paddle.randn([batch_size, 1])
        input_dict[var].stop_gradient = False

    # prepare model
    model = ppsci.arch.MLP(input_keys, output_keys, 2, 10)

    # prepare python function expressions and sympy-expression in dict
    functional_expr_dict = equation.NavierStokes(nu, rho, dim, time).equations
    sympy_expr_dict = NavierStokes_sympy(nu, rho, dim, time).equations
    for target, expr in sympy_expr_dict.items():
        sympy_expr_dict[target] = expression.sympy_to_function(
            target,
            expr,
            [
                model,
            ],
        )

    # compute equation with python function
    output_dict_functional = model(input_dict)
    for name, expr in functional_expr_dict.items():
        if callable(expr):
            output_dict_functional[name] = expr(
                {**output_dict_functional, **input_dict}
            )
        else:
            raise TypeError(f"expr type({type(expr)}) is invalid")
    clear()

    # compute equation with funciton converted from sympy
    output_dict_sympy = {k: v for k, v in input_dict.items()}
    for name, _ in sympy_expr_dict.items():
        output_dict_sympy[name] = sympy_expr_dict[name](
            {**output_dict_sympy, **input_dict}
        )
    clear()

    # test for result
    for key in functional_expr_dict:
        assert paddle.allclose(
            output_dict_functional[key], output_dict_sympy[key], atol=1e-7
        )


if __name__ == "__main__":
    pytest.main()
