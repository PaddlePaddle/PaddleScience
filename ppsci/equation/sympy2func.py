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
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import sympy

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

func_map = {
    sympy.sin: paddle.sin,
    sympy.cos: paddle.cos,
    sympy.exp: paddle.exp,
    sympy.Pow: paddle.pow,
    sympy.sqrt: paddle.sqrt,
    sympy.log: paddle.log,
    sympy.tan: paddle.tan,
    sympy.Mul: paddle.multiply,
}

constant_map = {
    sympy.pi: np.pi,
    sympy.E: np.e,
}


class FuncNode(nn.Layer):
    """
    A node representing a function in the computational graph

    Args:
        fun (paddle.nn.Layer): the function
        args (List[paddle.nn.Layer]): the arguments of the function

    Returns:
        the result of the function
    """

    def __init__(self, fun, *args):
        super().__init__()
        self.fun = fun
        self.args = args

    def forward(self, x):
        return self.fun(*[arg(x) for arg in self.args])


class AddNode(nn.Layer):
    """
    A node representing a sum in the computational graph

    Args:
        args (List[paddle.nn.Layer]): the arguments of the sum

    Returns:
        the result of the sum
    """

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return paddle.add_n([arg(x) for arg in self.args])


class SymbolNode(nn.Layer):
    """
    A node representing a symbol in the computational graph

    Args:
        sym (sympy.Symbol): the symbol

    Returns:
        the value of the symbol
    """

    def __init__(self, sym: sympy.Symbol):
        super().__init__()
        self.sym = sym

    def forward(self, x: Dict):
        if self.sym in x.keys():
            return x[self.sym]
        else:
            raise KeyError(f"Symbol {self.sym} not in {x.keys()}")


class NumberNode(nn.Layer):
    """
    A node representing a number in the computational graph

    Args:
        num (sympy.Number): the number

    Returns:
        the value of the number
    """

    def __init__(self, num):
        super().__init__()
        assert isinstance(num, sympy.Number)
        if num in constant_map.keys():
            num = constant_map[num]
        else:
            num = float(num)
        self.num = num

    def forward(self, x):
        return paddle.to_tensor(self.num, dtype="float32")


class DerivativeNode(nn.Layer):
    """
    A node representing a derivative in the computational graph

    Args:
        expr (sympy.Expr): the expression to be derived
        syms (List[Tuple[sympy.Symbol, int]]): the symbols to be derived and their orders

    Returns:
        the value of the derivative
    """

    def __init__(self, expr: sympy.Expr, syms: List[Tuple[sympy.Symbol, int]]):
        super().__init__()
        self.expr = expr
        self.syms = syms

    def forward(self, x):
        x_value = self.expr(x)
        for sym, order in self.syms:
            sym_value = sym(x)
            if order == 1:
                x_value = jacobian(x_value, sym_value)
            elif order == 2:
                x_value = hessian(x_value, sym_value)
            else:
                raise NotImplementedError(
                    f"Higher order derivatives are not implemented yet, got {order}"
                )
        return x_value


class ExtraFuncNode(nn.Layer):
    """
    A node representing a extra function in the computational graph

    Args:
        fun (sympy.Function): the function
        args (List[paddle.nn.Layer]): the arguments of the function

    Returns:
        the result of the function

    Note:
        This is used to handle the case where the function is a neural network

    Examples:
        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> fun = sympy.Derivative(u, x, y)
        >>> fun = sympy_to_function(fun)
        >>> fun({u: model, x: paddle.to_tensor(0.5), y: paddle.to_tensor(0.5)})

        Other cases:

        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> v = sympy.Function("v")(x, y)
        >>> fun = sympy.Derivative(u, x, y) + sympy.Derivative(v, x, y)
        >>> fun = sympy_to_function(fun)
        >>> fun({u: (model, 0), v: (model, 1), x: paddle.to_tensor(0.5), y: paddle.to_tensor(0.5)})
    """

    def __init__(self, fun: sympy.Function, *args):
        super().__init__()
        assert isinstance(fun, sympy.Function)
        self.fun = fun
        self.args = args

    def forward(self, x: Dict):
        model = x[self.fun]
        if isinstance(model, tuple):
            model, pos = model
            return model(*[arg(x) for arg in self.args])[
                pos
            ]  # TODO(PuQing): lazy computing for model, avoid multiple computing
        return model(*[arg(x) for arg in self.args])


def process_sympy_expression(expr: sympy.Expr):
    if expr.is_Symbol:
        return SymbolNode(expr)
    elif expr.is_Function or expr.is_Pow or expr.is_Mul:
        args = [process_sympy_expression(arg) for arg in expr.args]
        try:
            paddle_func = func_map[expr.func]
            return FuncNode(paddle_func, *args)
        except KeyError:
            logger.warning(
                f"Note that you appear to be using a non-built-in function {expr}, please pass in that when you call the function"
            )
    elif expr.is_Add:
        args = [process_sympy_expression(arg) for arg in expr.args]
        return AddNode(*args)
    elif expr.is_Number:
        return NumberNode(expr)
    elif expr.is_Derivative:
        expr = process_sympy_expression(expr.args[0])
        syms = [(process_sympy_expression(sym), order) for sym, order in expr.args[1:]]
        return DerivativeNode(expr, syms)
    else:
        raise NotImplementedError(f"Unknown type {expr}")


def sympy_to_function(expr: sympy.Expr):
    """
    Convert a sympy expression to a function that can be used in paddle

    Args:
        expr (sympy.Expr): the sympy expression

    Returns:
        a function that can be used in paddle

    Examples:
        >>> x = sympy.Symbol("x")
        >>> expr = sympy.sin(x)
        >>> func = sympy_to_function(expr)
        >>> func({"x": paddle.to_tensor(0.5)})
        Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
                [0.47942555])
    """
    return process_sympy_expression(sympy.expand(expr))
