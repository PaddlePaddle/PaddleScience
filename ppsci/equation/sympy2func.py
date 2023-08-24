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

import functools
from typing import Dict
from typing import List
from typing import Union

import paddle
import paddle.nn as nn
import sympy

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

FUNC_MAP = {
    sympy.sin: paddle.sin,
    sympy.cos: paddle.cos,
    sympy.exp: paddle.exp,
    sympy.Pow: paddle.pow,
    # sympy.sqrt: paddle.sqrt,
    sympy.log: paddle.log,
    sympy.tan: paddle.tan,
    sympy.Max: paddle.maximum,
    sympy.Min: paddle.minimum,
    sympy.Abs: paddle.abs,
    sympy.Heaviside: functools.partial(paddle.heaviside, y=paddle.zeros([])),
}


def single_derivate_func(dvar: paddle.Tensor, invar: paddle.Tensor, order: int):
    order_left = order
    while order_left > 0:
        if order_left >= 2:
            dvar = hessian(dvar, invar)
            order_left -= 2
        else:
            dvar = jacobian(dvar, invar)
            order_left -= 1
    return dvar


def cvt_to_key(sympy_node: sympy.Basic):
    if isinstance(sympy_node, sympy.Heaviside):
        return str(sympy_node)
    if isinstance(sympy_node, (sympy.Symbol, sympy.Function)):
        return sympy_node.name
    elif isinstance(sympy_node, sympy.Derivative):
        expr_str = sympy_node.args[0].name  # use 'f' instead of 'f(x,y,z)'
        for symbol, order in sympy_node.args[1:]:
            expr_str += f"__{symbol}" * order
        return expr_str
    else:
        return str(sympy_node)


class NodeBase(nn.Layer):
    """
    The base class of the node in the computational graph.

    Args:
        expr (sympy.Expr): The expression of the node.

    Returns:
        The input dictionary with the output of the node added.
    """

    def __init__(self, expr: sympy.Expr):
        super().__init__()
        self.expr = expr
        self.key = cvt_to_key(self.expr)

    def forward(self, data_dict: Dict, model_dict: Dict[str, nn.Layer] = None):
        raise NotImplementedError

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(expr: {self.expr}), type: {type(self.expr)})"
        )


class OperatorNode(NodeBase):
    """
    A node representing a sympy operator in the computational graph.

    (e.g. sin, cos, etc.)

    Args:
        expr (sympy.Expr): The expression of the node.

    Returns:
        The input dictionary with the output of the operator added.
    """

    def __init__(self, expr: Union[sympy.Add, sympy.Mul, sympy.Derivative]):
        super().__init__(expr)

    def forward(self, data_dict: Dict, model_dict: Dict[str, nn.Layer] = None):
        if self.expr.func == sympy.Add:
            data_dict[self.key] = paddle.add_n(
                [data_dict[cvt_to_key(arg)] for arg in self.expr.args]
            )
        elif self.expr.func == sympy.Mul:
            data_dict[self.key] = data_dict[cvt_to_key(self.expr.args[0])]
            for arg in self.expr.args[1:]:
                data_dict[self.key] = data_dict[self.key] * data_dict[cvt_to_key(arg)]
        elif self.expr.func == sympy.Derivative:
            if self.key in data_dict:
                return data_dict
            data_dict[self.key] = data_dict[cvt_to_key(self.expr.args[0])]
            for symbol, order in self.expr.args[1:]:
                data_dict[self.key] = single_derivate_func(
                    data_dict[self.key],
                    data_dict[cvt_to_key(symbol)],
                    order,
                )
        else:
            try:
                func = FUNC_MAP[self.expr.func]
            except KeyError:
                raise NotImplementedError(
                    f"'{self.expr.func}' operator is not supported now."
                )
            if self.expr.func == sympy.Heaviside:
                data_dict[self.key] = func(data_dict[cvt_to_key(self.expr.args[0])])
            else:
                data_dict[self.key] = func(
                    *[data_dict[cvt_to_key(arg)] for arg in self.expr.args]
                )
        return data_dict


class LayerNode(NodeBase):
    """
    A node representing a neural network in the computational graph

    Args:
        expr (sympy.core.function.UndefinedFunction): Definition symbol of the neural network.

    Returns:
        The input dictionary with the output of the neural network added.

    Note:
        For the provided network, the forward should accept a dictionary as input and return a dictionary as output.
        And the `output_keys` should be provided in the `__init__` function.

    Examples:

    """

    def __init__(self, expr: sympy.core.function.UndefinedFunction, model: nn.Layer):
        super().__init__(expr)
        self.model = model

    def forward(self, data_dict: Dict):
        if self.key in data_dict:
            return data_dict

        output_dict = self.model(data_dict)
        for key, value in output_dict.items():
            data_dict[key] = value

        return data_dict


class ConstantNode(NodeBase):
    """
    A node representing a constant in the computational graph.

    Args:
        expr (sympy.Number or sympy.NumberSymbol): The constant to be applied.

    Returns:
        The input dictionary with the constant added.

    Examples:
        >>> node = ConstantNode(sympy.pi)
        >>> node({})
        {'pi': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            3.1415927)}
    """

    def __init__(self, expr: sympy.Number or sympy.NumberSymbol):
        super().__init__(expr)
        if self.expr.is_Float:
            self.expr = float(self.expr)
        elif self.expr.is_Integer:
            self.expr = float(self.expr)
        elif self.expr.is_Boolean:
            self.expr = float(self.expr)
        elif self.expr.is_Rational:
            self.expr = float(self.expr)
        else:
            raise TypeError(
                f"expr({expr}) should be float/int/bool, but got {type(self.expr)}"
            )
        self.expr = paddle.to_tensor(self.expr)

    def forward(self, data_dict: Dict):
        data_dict[self.key] = self.expr
        return data_dict


class ComposedFunc(nn.Layer):
    """
    Compose multiple functions into one function.

    Args:
        data_dict (Dict): The input tensor dictionary.
        model_dict (Dict[str, nn.Layer]): The dictionary of the models.

    Returns:
        The dictionary of the outputs of the all calculated nodes.
    """

    def __init__(self, target: str, funcs: List[NodeBase]):
        super().__init__()
        self.funcs = funcs
        self.target = target

    def forward(self, data_dict: Dict):
        for func in self.funcs:
            data_dict = func(data_dict)
        return data_dict[self.funcs[-1].key]


def post_traverse(cur_node, nodes):
    # traverse into sub-nodes
    if isinstance(cur_node, sympy.core.function.UndefinedFunction):
        nodes.append(cur_node)
    elif isinstance(cur_node, sympy.Function):
        for arg in cur_node.args:
            nodes = post_traverse(arg, nodes)
        nodes.append(cur_node)
    elif isinstance(cur_node, sympy.Derivative):
        nodes = post_traverse(cur_node.args[0], nodes)
        nodes.append(cur_node)
    elif isinstance(cur_node, sympy.Symbol):
        return nodes
    elif isinstance(cur_node, sympy.Number):
        nodes.append(cur_node)
    else:
        for arg in cur_node.args:
            nodes = post_traverse(arg, nodes)
        nodes.append(cur_node)
    return nodes


def sympy_to_function(target: str, expr: sympy.Expr, models: nn.Layer):
    """
    Convert a sympy expression to a ComposedFunc.

    Args:
        expr (sympy.Expr): the sympy expression

    Returns:
        A ComposedFunc that can execute the formula represented by the sympy expression.

    Examples:
    """
    sympy_nodes = []
    sympy_nodes = post_traverse(expr, sympy_nodes)
    sympy_nodes = [node for node in sympy_nodes if not node.is_Symbol]
    sympy_nodes = list(
        dict.fromkeys(sympy_nodes)
    )  # remove duplicates with topo-order kept

    callable_nodes = []
    for i, node in enumerate(sympy_nodes):
        logger.debug(f"tree node [{i + 1}/{len(sympy_nodes)}]: {node}")
        if isinstance(node.func, sympy.core.function.UndefinedFunction):
            match = False
            for model in models:
                if str(node.func.name) in model.output_keys:
                    callable_nodes.append(LayerNode(node, model))
                    if match:
                        raise ValueError(
                            f"function {node} can match at least 2 output key of models, which is forbidden."
                        )
                    match = True
        elif (
            isinstance(node, tuple(FUNC_MAP.keys()))
            or node.is_Add
            or node.is_Mul
            or node.is_Derivative
            or node.is_Pow
        ):
            callable_nodes.append(OperatorNode(node))
        elif node.is_Number or node.is_NumberSymbol:
            callable_nodes.append(ConstantNode(node))
        else:
            raise NotImplementedError(
                f"The node {node} is not supported in sympy_to_function."
            )
    return ComposedFunc(target, callable_nodes)


class ZeroEquation:
    """
    Zero Equation Turbulence model

    Parameters
    ==========
    nu : float
        The kinematic viscosity of the fluid.
    max_distance : float
        The maximum wall distance in the flow field.
    rho : float, Sympy Symbol/Expr, str
        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 1.
    dim : int
        Dimension of the Zero Equation Turbulence model (2 or 3).
        Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Example
    """

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
        x, y, z = sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")

        # time
        t = sympy.Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = sympy.Function("u")(*input_variables)
        v = sympy.Function("v")(*input_variables)
        if self.dim == 3:
            w = sympy.Function("w")(*input_variables)
        else:
            w = sympy.Number(0)

        # density
        if type(rho) is str:
            rho = sympy.Function(rho)(*input_variables)
        elif type(rho) in [float, int]:
            rho = sympy.Number(rho)

        # wall distance
        normal_distance = sympy.Function("sdf")(*input_variables)

        # mixing length
        mixing_length = sympy.Min(
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
        self.equations["nu"] = nu + rho * mixing_length**2 * sympy.sqrt(G)


class NavierStokes_sympy:
    """
    Compressible Navier Stokes equations

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    """

    name = "NavierStokes"

    def __init__(self, nu, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")

        # time
        t = sympy.Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = sympy.Function("u")(*input_variables)
        v = sympy.Function("v")(*input_variables)
        if self.dim == 3:
            w = sympy.Function("w")(*input_variables)
        else:
            w = sympy.Number(0)

        # pressure
        p = sympy.Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = sympy.Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = sympy.Number(nu)

        # density
        if isinstance(rho, str):
            rho = sympy.Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = sympy.Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        if not self.mixed_form:
            curl = (
                sympy.Number(0)
                if rho.diff(x) == 0
                else u.diff(x) + v.diff(y) + w.diff(z)
            )
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

        elif self.mixed_form:
            u_x = sympy.Function("u_x")(*input_variables)
            u_y = sympy.Function("u_y")(*input_variables)
            u_z = sympy.Function("u_z")(*input_variables)
            v_x = sympy.Function("v_x")(*input_variables)
            v_y = sympy.Function("v_y")(*input_variables)
            v_z = sympy.Function("v_z")(*input_variables)

            if self.dim == 3:
                w_x = sympy.Function("w_x")(*input_variables)
                w_y = sympy.Function("w_y")(*input_variables)
                w_z = sympy.Function("w_z")(*input_variables)
            else:
                w_x = sympy.Number(0)
                w_y = sympy.Number(0)
                w_z = sympy.Number(0)
                u_z = sympy.Number(0)
                v_z = sympy.Number(0)

            curl = sympy.Number(0) if rho.diff(x) == 0 else u_x + v_y + w_z
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u.diff(x)))
                    + v * ((rho * u.diff(y)))
                    + w * ((rho * u.diff(z)))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u_x).diff(x)
                - (mu * u_y).diff(y)
                - (mu * u_z).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v.diff(x)))
                    + v * ((rho * v.diff(y)))
                    + w * ((rho * v.diff(z)))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v_x).diff(x)
                - (mu * v_y).diff(y)
                - (mu * v_z).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w.diff(x)))
                    + v * ((rho * w.diff(y)))
                    + w * ((rho * w.diff(z)))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w_x).diff(x)
                - (mu * w_y).diff(y)
                - (mu * w_z).diff(z)
                - (mu * (curl).diff(z))
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_v_x"] = v.diff(x) - v_x
            self.equations["compatibility_v_y"] = v.diff(y) - v_y
            self.equations["compatibility_v_z"] = v.diff(z) - v_z
            self.equations["compatibility_w_x"] = w.diff(x) - w_x
            self.equations["compatibility_w_y"] = w.diff(y) - w_y
            self.equations["compatibility_w_z"] = w.diff(z) - w_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            self.equations["compatibility_v_xy"] = v_x.diff(y) - v_y.diff(x)
            self.equations["compatibility_v_xz"] = v_x.diff(z) - v_z.diff(x)
            self.equations["compatibility_v_yz"] = v_y.diff(z) - v_z.diff(y)
            self.equations["compatibility_w_xy"] = w_x.diff(y) - w_y.diff(x)
            self.equations["compatibility_w_xz"] = w_x.diff(z) - w_z.diff(x)
            self.equations["compatibility_w_yz"] = w_y.diff(z) - w_z.diff(y)

            if self.dim == 2:
                self.equations.pop("momentum_z")
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_v_z")
                self.equations.pop("compatibility_w_x")
                self.equations.pop("compatibility_w_y")
                self.equations.pop("compatibility_w_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")
                self.equations.pop("compatibility_v_xz")
                self.equations.pop("compatibility_v_yz")
                self.equations.pop("compatibility_w_xy")
                self.equations.pop("compatibility_w_xz")
                self.equations.pop("compatibility_w_yz")


if __name__ == "__main__":
    logger.init_logger(log_level="debug")
    # ze = ZeroEquation(nu=1, rho=1.0, dim=2, max_distance=4, time=False)
    ns = NavierStokes_sympy(nu=2.0, rho=1.0, dim=2, time=False)
    target = "momentum_x"
    test_expr = ns.equations[target]

    x = paddle.randn([4, 1])
    y = paddle.randn([4, 1])
    z = paddle.randn([4, 1])
    sdf = paddle.randn([4, 1])
    sdf__x = paddle.randn([4, 1])
    sdf__y = paddle.randn([4, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    z.stop_gradient = False
    sdf.stop_gradient = False
    sdf__x.stop_gradient = False
    sdf__y.stop_gradient = False

    input_dict = {
        "x": x,
        "y": y,
        "z": z,
        "sdf": sdf,
        "sdf__x": sdf__x,
        "sdf__y": sdf__y,
    }

    model1 = ppsci.arch.MLP(("x", "y", "z"), ("u", "v"), 2, 10)
    model2 = ppsci.arch.MLP(("x", "y", "z"), ("w", "p"), 2, 10)

    cvt_expr = sympy_to_function(target, test_expr, [model1, model2])

    output = cvt_expr(input_dict)
    print(output.shape)
