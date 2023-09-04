import time as time_module

import paddle
import sympy as sp

from ppsci import arch
from ppsci import equation
from ppsci.autodiff import clear
from ppsci.autodiff import hessian as H
from ppsci.autodiff import jacobian as J
from ppsci.utils import sym_to_func


class NavierStokes_sympy:
    def __init__(self, nu, rho, dim, time):
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


def compute_with_sympy(input_dicts, nu, rho, dim, time, model):
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
    cost_list = []
    # prepare python function expressions and sympy-expression in dict
    sympy_expr_dict = NavierStokes_sympy(nu_sympy, rho, dim, time).equations
    for target, expr in sympy_expr_dict.items():
        sympy_expr_dict[target] = sym_to_func.sympy_to_function(
            expr,
            [
                model,
            ],
        )
    for i, input_dict in enumerate(input_dicts):
        input_dict = input_dicts[i]

        # compute equation with funciton converted from sympy
        output_dict_sympy = {k: v for k, v in input_dict.items()}
        tmp = {k: v for k, v in output_dict_sympy.items()}
        beg = time_module.perf_counter()
        for name, expr in sympy_expr_dict.items():
            output = expr(tmp)
            output_dict_sympy[name] = output
        for key in model.output_keys:
            output_dict_sympy[key] = tmp[key]
        clear()
        end = time_module.perf_counter()
        cost_list.append(end - beg)

    # test for result
    print(
        f"compute_with_sympy overhead: {sum(cost_list[10:]) / len(cost_list[10:]):.5f}"
    )
    return output_dict_sympy


def compute_with_pyfunc(input_dicts, nu, rho, dim, time, model):
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

    """Test for navier_stokes equation."""
    # define input/output keys

    # prepare input data in dict
    cost_list = []
    for i, input_dict in enumerate(input_dicts):
        input_dict = input_dicts[i]

        # prepare python function expressions in dict
        functional_expr_dict = equation.NavierStokes(nu, rho, dim, time).equations
        functional_expr_dict["continuity"] = continuity_f
        functional_expr_dict["momentum_x"] = momentum_x_f
        functional_expr_dict["momentum_y"] = momentum_y_f

        # compute equation with python function
        output_dict_functional = model(input_dict)
        beg = time_module.perf_counter()
        for name, expr in functional_expr_dict.items():
            if callable(expr):
                output_dict_functional[name] = expr(
                    {**output_dict_functional, **input_dict}
                )
            else:
                raise TypeError(f"expr type({type(expr)}) is invalid")
        clear()
        end = time_module.perf_counter()
        cost_list.append(end - beg)

    # test for result
    print(
        f"compute_with_pyfunc overhead: {sum(cost_list[10:]) / len(cost_list[10:]):.5f}"
    )
    return output_dict_functional


if __name__ == "__main__":
    input_keys = ("t", "x", "y")
    output_keys = ("u", "v", "p")
    nu = 2
    rho = 1
    dim = 2
    time = True
    model = arch.MLP(input_keys, output_keys, 4, 50)

    batch_size = 2048
    input_dicts = []
    for i in range(50):
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
        input_dicts.append(input_dict)

    output_dict_sympy = compute_with_sympy(
        input_dicts, nu=nu, rho=rho, dim=dim, time=time, model=model
    )
    output_dict_pyfunc = compute_with_pyfunc(
        input_dicts, nu=nu, rho=rho, dim=dim, time=time, model=model
    )

    for key in output_dict_pyfunc:
        if not paddle.allclose(
            output_dict_sympy[key], output_dict_pyfunc[key], atol=1e-7
        ):
            print(f"{key} {output_dict_sympy[key]}\n{output_dict_pyfunc[key]}")
        else:
            print(f"{key} check pass")
