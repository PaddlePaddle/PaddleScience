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

import paddle
import paddle.nn as nn
import sympy

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

FUNC_MAP = {
    sympy.sin: paddle.sin,
    sympy.cos: paddle.cos,
    sympy.exp: paddle.exp,
    sympy.Pow: paddle.pow,
    sympy.sqrt: paddle.sqrt,
    sympy.log: paddle.log,
    sympy.tan: paddle.tan,
    sympy.Mul: paddle.multiply,
    sympy.Add: paddle.add_n,
}


class FuncNode(nn.Layer):
    """
    A node representing a paddle function in the computational graph.

    Args:
        func (nn.Layer): The function to be applied.
        *args (nn.Layer): The arguments of the function.

    Returns:
        The result of applying the function to the arguments.

    Examples:
        >>> x = sympy.Symbol("x")
        >>> node = FuncNode(paddle.sin, SymbolNode(x))
        >>> node({x: paddle.to_tensor(0.5)})
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
         0.47942555)
    """

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def forward(self, x):
        if self.func == paddle.add_n:
            return self.func([arg(x) for arg in self.args])
        return self.func(*[arg(x) for arg in self.args])


class SymbolNode(nn.Layer):
    """
    A node retrieves the value of a symbol from the provided dictionary.

    Args:
        symbol (sympy.Symbol): The symbol to be represent in the graph

    Returns:
        The value of the symbol

    Examples:
        >>> x = sympy.Symbol("x")
        >>> node = SymbolNode(x)
        >>> node({x: paddle.to_tensor(0.5)})
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
         0.50000000)

       Or you can use the name of the symbol

        >>> x = sympy.Symbol("x")
        >>> node = SymbolNode(x)
        >>> node({"x": paddle.to_tensor(0.5)})
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
         0.50000000)
    """

    def __init__(self, symbol: sympy.Symbol):
        super().__init__()
        self.symbol = symbol

    def forward(self, x: Dict):
        value = x.get(self.symbol, x.get(self.symbol.name))
        if value is None:
            raise ValueError(
                f"Symbol {self.symbol} not in provided dictionary {list(x.keys())}!"
            )
        return value


class NumberNode(nn.Layer):
    """
    A node representing a number in the computational graph

    Args:
        number (sympy.Number): the number

    Returns:
        the value of the number

    Examples:
        >>> node = NumberNode(sympy.pi)
        >>> node({})
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            3.1415927)
    """

    def __init__(self, number: sympy.Number):
        super().__init__()
        self.number = float(number)

    def forward(self, x):
        return paddle.to_tensor(self.number, dtype=paddle.get_default_dtype())


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


class LayerNode(nn.Layer):
    """
    A node representing a neural network in the computational graph

    Args:
        func (sympy.Function): the neural network represented by a sympy function
        *args (SymbolNode): the arguments of the function

    Returns:
        the output of the neural network

    Note:
        For a multi-output model, only one symbol can be provided in the input dictionary,

    Examples:
        Single output case:
        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> func = sympy.Derivative(u, x, y)
        >>> func = sympy_to_function(func)
        >>> func({u: model, x: paddle.to_tensor(0.5), y: paddle.to_tensor(0.5)})

        Multi-output case:
        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> v = sympy.Function("v")(x, y)
        >>> func = sympy.Derivative(u, x, y) + sympy.Derivative(v, x, y)
        >>> func = sympy_to_function(func)
        >>> func({u: model, x: paddle.to_tensor(0.5), y: paddle.to_tensor(0.5)}) # The model should have output_keys = ["u", "v"]
    """

    _MODEL_OUTPUT_CACHE: Dict[str, paddle.Tensor] = {}

    def __init__(self, func: sympy.Function, *args: SymbolNode):
        super().__init__()
        assert isinstance(func, sympy.Function)
        self.func = func
        self.args = args

    def forward(self, x: Dict):
        # check if the model output is in the cache
        model_output = self._MODEL_OUTPUT_CACHE.get(self.func.name)

        if model_output is None:
            # Find which model provides the symbol value
            for model in x.values():
                if hasattr(model, "output_keys"):
                    output_keys: Dict = model.output_keys
                    if self.func.name in output_keys:
                        model_output_dict: Dict = model(
                            {arg.symbol.name: arg(x) for arg in self.args}
                        )
                        for key in output_keys:
                            self._MODEL_OUTPUT_CACHE[key] = model_output_dict[key]
                        break
            else:  # when no model provides the symbol value
                raise ValueError(
                    f"Model {self.func.name} not in provided dictionary {list(x.keys())}!"
                )

        output = self._MODEL_OUTPUT_CACHE[self.func.name]
        self._MODEL_OUTPUT_CACHE[self.func.name] = None
        return output


def process_sympy_expression(expr: sympy.Expr):
    if expr.is_Symbol:
        return SymbolNode(expr)
    elif expr.is_Function or expr.is_Pow or expr.is_Mul or expr.is_Add:
        args = [process_sympy_expression(arg) for arg in expr.args]
        try:
            paddle_func = FUNC_MAP[expr.func]
            return FuncNode(paddle_func, *args)
        except KeyError:
            logger.warning(
                f"Note that you appear to be using a non-built-in function {expr}, please pass in that when you call the function"
            )
            return LayerNode(expr, *args)
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
