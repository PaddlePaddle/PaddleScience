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
import sympy as sp

import ppsci
from ppsci import equation
from ppsci.autodiff import clear
from ppsci.autodiff import hessian as H
from ppsci.autodiff import jacobian as J
from ppsci.utils import sym_to_func


class NavierStokes_sympy:
    def __init__(self, nu, rho=1, dim=3, time=True):
        # set params
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")

        # time
        t = sp.Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = sp.Function("u")(*input_variables)
        v = sp.Function("v")(*input_variables)
        if self.dim == 3:
            w = sp.Function("w")(*input_variables)
        else:
            w = sp.Number(0)

        # pressure
        p = sp.Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = sp.Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = sp.Number(nu)

        # density
        if isinstance(rho, str):
            rho = sp.Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = sp.Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        curl = sp.Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
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


class ZeroEquation_sympy:
    def __init__(
        self, nu, max_distance, rho=1, dim=3, time=True
    ):  # TODO add density into model
        # set params
        self.dim = dim
        self.time = time

        # model coefficients
        self.max_distance = max_distance
        self.karman_constant = 0.419
        self.max_distance_ratio = 0.09

        # coordinates
        x, y, z = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")

        # time
        t = sp.Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = sp.Function("u")(*input_variables)
        v = sp.Function("v")(*input_variables)
        if self.dim == 3:
            w = sp.Function("w")(*input_variables)
        else:
            w = sp.Number(0)

        # density
        if type(rho) is str:
            rho = sp.Function(rho)(*input_variables)
        elif type(rho) in [float, int]:
            rho = sp.Number(rho)

        # wall distance
        normal_distance = sp.Function("sdf")(*input_variables)

        # mixing length
        mixing_length = sp.Min(
            self.karman_constant * normal_distance,
            self.max_distance_ratio * self.max_distance,
        )
        G = (
            2 * u.diff(x) ** 2
            + 2 * v.diff(y) ** 2
            + 2 * w.diff(z) ** 2
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2
            + (v.diff(z) + w.diff(y)) ** 2
        )

        # set equations
        self.equations = {}
        self.equations["nu"] = nu + rho * mixing_length**2 * sp.sqrt(G)


class Test_NavierStokes_sympy:
    @pytest.mark.parametrize("nu", (2.0,))
    @pytest.mark.parametrize("rho", (1.0,))
    @pytest.mark.parametrize("dim", (2,))
    @pytest.mark.parametrize("time", (False, True))
    def test_nu_sympy(self, nu, rho, dim, time):
        """Test for navier_stokes equation."""
        # define input/output keys
        ze = ZeroEquation_sympy(nu=nu, rho=rho, dim=dim, max_distance=3.4, time=time)
        nu_sympy = ze.equations["nu"]

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
            if var != "t":
                input_dict[f"sdf__{var}"] = paddle.randn([batch_size, 1])
                input_dict[f"normal__{var}"] = paddle.randn([batch_size, 1])

                input_dict[f"sdf__{var}"].stop_gradient = False
                input_dict[f"normal__{var}"].stop_gradient = False

        input_dict["sdf"] = paddle.randn([batch_size, 1])
        input_dict["sdf"].stop_gradient = False

        # prepare model
        model = ppsci.arch.MLP(input_keys, output_keys, 2, 10)

        # prepare python function expressions and sympy-expression in dict
        def nu_f(out):
            karman_constant = 0.419
            max_distance_ratio = 0.09
            normal_distance = out["sdf"]
            max_distance = ze.max_distance
            mixing_length = paddle.minimum(
                karman_constant * normal_distance,
                max_distance_ratio * max_distance,
            )
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            G = 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2 + (J(u, y) + J(v, x)) ** 2
            if dim == 3:
                z, w = out["z"], out["w"]
                G += (
                    +2 * J(w, z) ** 2
                    + (J(u, z) + J(w, x)) ** 2
                    + (J(v, z) + J(w, y)) ** 2
                )
            return nu + rho * mixing_length**2 * paddle.sqrt(G)

        functional_expr_dict = equation.NavierStokes(nu_f, rho, dim, time).equations

        def continuity_f(out):
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            return 1.0 * J(u, x) + 1.0 * J(v, y)

        def momentum_x_f(out):
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            if time:
                t = out["t"]
            return (
                -(
                    1.0
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    + 2.0
                )
                * H(u, x)
                - (
                    1.0
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    + 2.0
                )
                * H(u, y)
                - (
                    1.0
                    * (
                        (J(u, y) + J(v, x)) * (2 * H(u, y) + 2 * J(J(v, x), y)) / 2
                        + 2 * J(u, x) * J(J(u, x), y)
                        + 2 * J(v, y) * H(v, y)
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    / paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    + 0.838
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.heaviside(0.306 - 0.419 * out["sdf"], paddle.zeros([]))
                    * out["sdf__y"]
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                )
                * J(u, y)
                - (
                    1.0
                    * (
                        (J(u, y) + J(v, x)) * (2 * H(v, x) + 2 * J(J(u, x), y)) / 2
                        + 2 * J(u, x) * H(u, x)
                        + 2 * J(v, y) * J(J(v, x), y)
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    / paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    + 0.838
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.heaviside(0.306 - 0.419 * out["sdf"], paddle.zeros([]))
                    * out["sdf__x"]
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                )
                * J(u, x)
                + (1.0 * u * J(u, x) + 1.0 * v * J(u, y) + J(p, x))
                + (J(u, t) if time else 0)
            )

        def momentum_y_f(out):
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            if time:
                t = out["t"]
            return (
                -(
                    1.0
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    + 2.0
                )
                * H(v, x)
                - (
                    1.0
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    + 2.0
                )
                * H(v, y)
                - (
                    1.0
                    * (
                        (J(u, y) + J(v, x)) * (2 * H(u, y) + 2 * J(J(v, x), y)) / 2
                        + 2 * J(u, x) * J(J(u, x), y)
                        + 2 * J(v, y) * H(v, y)
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    / paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    + 0.838
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.heaviside(0.306 - 0.419 * out["sdf"], paddle.zeros([]))
                    * out["sdf__y"]
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                )
                * J(v, y)
                - (
                    1.0
                    * (
                        (J(u, y) + J(v, x)) * (2 * H(v, x) + 2 * J(J(u, x), y)) / 2
                        + 2 * J(u, x) * H(u, x)
                        + 2 * J(v, y) * J(J(v, x), y)
                    )
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                    ** 2
                    / paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    + 0.838
                    * paddle.sqrt(
                        (J(u, y) + J(v, x)) ** 2 + 2 * J(u, x) ** 2 + 2 * J(v, y) ** 2
                    )
                    * paddle.heaviside(0.306 - 0.419 * out["sdf"], paddle.zeros([]))
                    * out["sdf__x"]
                    * paddle.minimum(
                        paddle.full_like(out["sdf"], 0.306), 0.419 * out["sdf"]
                    )
                )
                * J(v, x)
                + (1.0 * u * J(v, x) + 1.0 * v * J(v, y) + J(p, y))
                + (J(v, t) if time else 0)
            )

        functional_expr_dict["continuity"] = continuity_f
        functional_expr_dict["momentum_x"] = momentum_x_f
        functional_expr_dict["momentum_y"] = momentum_y_f

        sympy_expr_dict = NavierStokes_sympy(nu_sympy, rho, dim, time).equations
        for target, expr in sympy_expr_dict.items():
            sympy_expr_dict[target] = sym_to_func.sympy_to_function(
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
        for name, expr in sympy_expr_dict.items():
            tmp = expr(output_dict_sympy)
            output_dict_sympy[name] = tmp
        clear()

        # test for result
        for key in functional_expr_dict:
            assert paddle.allclose(
                output_dict_functional[key], output_dict_sympy[key], atol=1e-7
            ), f"{key} not equal."

    @pytest.mark.parametrize("nu", (2.0,))
    @pytest.mark.parametrize("rho", (1.0,))
    @pytest.mark.parametrize("dim", (2,))
    @pytest.mark.parametrize("time", (False, True))
    def test_nu_constant(self, nu, rho, dim, time):
        """Test for navier_stokes equation."""
        # define input/output keys
        nu_sympy = nu

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
            if var != "t":
                input_dict[f"sdf__{var}"] = paddle.randn([batch_size, 1])
                input_dict[f"normal__{var}"] = paddle.randn([batch_size, 1])

                input_dict[f"sdf__{var}"].stop_gradient = False
                input_dict[f"normal__{var}"].stop_gradient = False

        input_dict["sdf"] = paddle.randn([batch_size, 1])
        input_dict["sdf"].stop_gradient = False

        # prepare model
        model = ppsci.arch.MLP(input_keys, output_keys, 2, 10)

        # prepare python function expressions and sympy-expression in dict
        functional_expr_dict = equation.NavierStokes(nu, rho, dim, time).equations

        sympy_expr_dict = NavierStokes_sympy(nu_sympy, rho, dim, time).equations
        for target, expr in sympy_expr_dict.items():
            sympy_expr_dict[target] = sym_to_func.sympy_to_function(
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
        tmp = {k: v for k, v in output_dict_sympy.items()}
        for name, expr in sympy_expr_dict.items():
            output = expr(tmp)
            output_dict_sympy[name] = output
        clear()

        # test for result
        for key in functional_expr_dict:
            assert paddle.allclose(
                output_dict_functional[key], output_dict_sympy[key], atol=1e-7
            ), f"{key} not equal."


if __name__ == "__main__":
    pytest.main()
