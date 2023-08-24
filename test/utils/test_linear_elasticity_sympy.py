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


class LinearElasticity_sympy:
    def __init__(
        self, E=None, nu=None, lambda_=None, mu=None, rho=1, dim=3, time=False
    ):

        # set params
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # displacement componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        sigma_xx = Function("sigma_xx")(*input_variables)
        sigma_yy = Function("sigma_yy")(*input_variables)
        sigma_xy = Function("sigma_xy")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
            sigma_zz = Function("sigma_zz")(*input_variables)
            sigma_xz = Function("sigma_xz")(*input_variables)
            sigma_yz = Function("sigma_yz")(*input_variables)
        else:
            w = Number(0)
            sigma_zz = Number(0)
            sigma_xz = Number(0)
            sigma_yz = Number(0)

        # material properties
        if lambda_ is None:
            if isinstance(nu, str):
                nu = Function(nu)(*input_variables)
            elif isinstance(nu, (float, int)):
                nu = Number(nu)
            if isinstance(E, str):
                E = Function(E)(*input_variables)
            elif isinstance(E, (float, int)):
                E = Number(E)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = Function(lambda_)(*input_variables)
            elif isinstance(lambda_, (float, int)):
                lambda_ = Number(lambda_)
            if isinstance(mu, str):
                mu = Function(mu)(*input_variables)
            elif isinstance(mu, (float, int)):
                mu = Number(mu)
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # set equations
        self.equations = {}

        # Stress equations
        self.equations["stress_disp_xx"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * u.diff(x)
            - sigma_xx
        )
        self.equations["stress_disp_yy"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * v.diff(y)
            - sigma_yy
        )
        self.equations["stress_disp_zz"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * w.diff(z)
            - sigma_zz
        )
        self.equations["stress_disp_xy"] = mu * (u.diff(y) + v.diff(x)) - sigma_xy
        self.equations["stress_disp_xz"] = mu * (u.diff(z) + w.diff(x)) - sigma_xz
        self.equations["stress_disp_yz"] = mu * (v.diff(z) + w.diff(y)) - sigma_yz

        # Equations of equilibrium
        self.equations["equilibrium_x"] = rho * ((u.diff(t)).diff(t)) - (
            sigma_xx.diff(x) + sigma_xy.diff(y) + sigma_xz.diff(z)
        )
        self.equations["equilibrium_y"] = rho * ((v.diff(t)).diff(t)) - (
            sigma_xy.diff(x) + sigma_yy.diff(y) + sigma_yz.diff(z)
        )
        self.equations["equilibrium_z"] = rho * ((w.diff(t)).diff(t)) - (
            sigma_xz.diff(x) + sigma_yz.diff(y) + sigma_zz.diff(z)
        )

        # Traction equations
        self.equations["traction_x"] = (
            normal_x * sigma_xx + normal_y * sigma_xy + normal_z * sigma_xz
        )
        self.equations["traction_y"] = (
            normal_x * sigma_xy + normal_y * sigma_yy + normal_z * sigma_yz
        )
        self.equations["traction_z"] = (
            normal_x * sigma_xz + normal_y * sigma_yz + normal_z * sigma_zz
        )

        # Navier equations
        self.equations["navier_x"] = (
            rho * ((u.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(x)
            - mu * ((u.diff(x)).diff(x) + (u.diff(y)).diff(y) + (u.diff(z)).diff(z))
        )
        self.equations["navier_y"] = (
            rho * ((v.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(y)
            - mu * ((v.diff(x)).diff(x) + (v.diff(y)).diff(y) + (v.diff(z)).diff(z))
        )
        self.equations["navier_z"] = (
            rho * ((w.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(z)
            - mu * ((w.diff(x)).diff(x) + (w.diff(y)).diff(y) + (w.diff(z)).diff(z))
        )

        if self.dim == 2:
            self.equations.pop("navier_z")
            self.equations.pop("stress_disp_zz")
            self.equations.pop("stress_disp_xz")
            self.equations.pop("stress_disp_yz")
            self.equations.pop("equilibrium_z")
            self.equations.pop("traction_z")


@pytest.mark.parametrize(
    "E,nu,lambda_,mu",
    (
        (2.0, 3.0, None, None),
        (None, None, 2.0, 3.0),
    ),
)
@pytest.mark.parametrize("rho", (1,))
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("time", (False, True))
def test_linearelasticity(E, nu, lambda_, mu, rho, dim, time):
    """Test for linearelasticity equation."""
    # define input/output keys
    input_keys = ("x", "y", "z")[:dim]
    if time:
        input_keys = ("t",) + input_keys

    disp_output_keys = ("u", "v")
    if dim == 3:
        disp_output_keys += ("w",)
    disp_output_keys += ("p",)

    stress_output_keys = ("sigma_xx", "sigma_yy")
    if dim == 3:
        stress_output_keys += ("sigma_zz",)
    stress_output_keys += ("sigma_xy",)
    if dim == 3:
        stress_output_keys += ("sigma_xz", "sigma_yz")

    # prepare input data in dict
    batch_size = 13
    input_dict = {}
    for var in input_keys:
        input_dict[var] = paddle.randn([batch_size, 1])
        input_dict[var].stop_gradient = False
        input_dict[f"normal_{var}"] = paddle.randn([batch_size, 1])
        input_dict[f"normal_{var}"].stop_gradient = False

    # prepare model
    disp_net = ppsci.arch.MLP(
        input_keys, disp_output_keys, 3, 16, "silu", weight_norm=True
    )
    stress_net = ppsci.arch.MLP(
        input_keys,
        stress_output_keys,
        3,
        16,
        "silu",
        weight_norm=True,
    )
    model_list = ppsci.arch.ModelList((disp_net, stress_net))

    # prepare python function expressions and sympy-expression in dict
    functional_expr_dict = equation.LinearElasticity(
        E, nu, lambda_, mu, rho, dim, time
    ).equations
    sympy_expr_dict = LinearElasticity_sympy(
        E, nu, lambda_, mu, rho, dim, time
    ).equations
    for target, expr in sympy_expr_dict.items():
        sympy_expr_dict[target] = expression.sympy_to_function(
            target, expr, [disp_net, stress_net]
        )

    # compute equation with python function
    output_dict_functional = model_list(input_dict)
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
