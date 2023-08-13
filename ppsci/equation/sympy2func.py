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
    sympy.Max: paddle.maximum,
    sympy.Min: paddle.minimum,
    sympy.Abs: paddle.abs,
}


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

    def forward(self, inputs_dict: Dict, models_dict: Dict[str, nn.Layer] = None):
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

    Examples:
        >>> x = sympy.Symbol("x")
        >>> node = OperatorNode(sympy.sin(x))
        >>> node({"x": paddle.to_tensor(np.random.randn(1, 1))})
        {'x': Tensor(shape=[1, 1], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                [[-0.49221350]]),
        'sin(x)': Tensor(shape=[1, 1], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                [[-0.47257778]])}
    """

    def __init__(
        self, expr: sympy.Function or sympy.Add or sympy.Mul or sympy.Derivative
    ):
        super().__init__(expr)

    def forward(self, inputs_dict: Dict, models_dict: Dict[str, nn.Layer] = None):
        expr_str = str(self.expr)
        if self.expr.func == sympy.Add:
            inputs_dict[expr_str] = paddle.add_n(
                [inputs_dict[str(arg)] for arg in self.expr.args]
            )
        elif self.expr.func == sympy.Mul:
            inputs_dict[expr_str] = paddle.to_tensor(
                1.0, dtype=paddle.get_default_dtype()
            )
            for arg in self.expr.args:
                inputs_dict[expr_str] *= inputs_dict[str(arg)]
        elif self.expr.func == sympy.Derivative:
            inputs_dict[expr_str] = inputs_dict[
                str(self.expr.args[0])
            ]  # initialize the derivative
            symbols = self.expr.args[1:]
            for symbol, order in symbols:
                expr_tensor = inputs_dict[expr_str]
                symbol_tensor = inputs_dict[str(symbol)]
                if order == 1:
                    inputs_dict[expr_str] = jacobian(expr_tensor, symbol_tensor)
                elif order == 2:
                    inputs_dict[expr_str] = hessian(expr_tensor, symbol_tensor)
                else:
                    logger.warning(
                        f"The order {order} of the derivative is not supported, the order should be 1 or 2."
                    )
        else:
            try:
                inputs_dict[expr_str] = FUNC_MAP[self.expr.func](
                    *[inputs_dict[str(arg)] for arg in self.expr.args]
                )
            except KeyError:
                logger.warning(
                    f"The operator {self.expr.func} is not supported, please add it to FUNC_MAP."
                )
        return inputs_dict


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
        Single output case:
        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> func = u.diff(x) + u.diff(y)
        >>> node = LayerNode(func)
        >>> x = paddle.to_tensor(np.random.randn(1, 1), stop_gradient=False, dtype="float32")
        >>> y = paddle.to_tensor(np.random.randn(1, 1), stop_gradient=False, dtype="float32")
        >>> class MyLayer(nn.Layer):
        >>>    def __init__(self):
        >>>        super(MyLayer, self).__init__()
        >>>        self.output_keys = ["u"]
        >>>    def forward(self, x):
        >>>        x, y = x["x"], x["y"]
        >>>        u = paddle.cos(y * x)
        >>>        return {"u": u}
        >>> node(inputs_dict={"x": x, "y": y}, model_dict={f"u": MyLayer()})
        {'x': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
        [[0.20314099]]),
         'y': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.95114714]]),
         'u(x, y)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.98139161]]),
         'Derivative(u(x, y), x)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[-0.18263641]])}

        Multi-output case:
        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> v = sympy.Function("v")(x, y)
        >>> func = u.diff(x) + u.diff(y)
        >>> node = LayerNode(func)
        >>> class MyLayer(nn.Layer):
        >>>    def __init__(self):
        >>>        super(MyLayer, self).__init__()
        >>>        self.output_keys = ["u", "v"]
        >>>    def forward(self, x):
        >>>        x, y = x["x"], x["y"]
        >>>        u = paddle.cos(y * x)
        >>>        v = paddle.tanh(x**2)
        >>>        return {"u": u, "v": v}
        >>> x = paddle.to_tensor(np.random.randn(1, 1), stop_gradient=False, dtype="float32")
        >>> y = paddle.to_tensor(np.random.randn(1, 1), stop_gradient=False, dtype="float32")
        >>> node(inputs_dict={"x": x, "y": y}, model_dict={"u": MyLayer()})
        {'x': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.65654278]]),
        'y': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.07239681]]),
        'u(x, y)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.99887061]]),
        'v(x, y)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.40619713]]),
        'Derivative(u(x, y), y)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[-0.03119478]]),
        'Derivative(u(x, y), x)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[-0.00343984]]),
        'Derivative(u(x, y), x) + Derivative(u(x, y), y)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[-0.03463462]])}
    """

    def __init__(self, expr: sympy.core.function.UndefinedFunction):
        super().__init__(expr)

    def forward(self, inputs_dict: Dict, models_dict: Dict[str, nn.Layer]):
        if str(self.expr) in inputs_dict:
            return inputs_dict
        for model in models_dict.values():
            if model.output_keys is None:
                raise ValueError(
                    "The output_keys of the model should be provided in the __init__ function."
                )
            model_output_keys = model.output_keys
            if str(self.expr.func) in model_output_keys:  # u(x, y) to u
                model_output = model(
                    {str(arg): inputs_dict[str(arg)] for arg in self.expr.args}
                )
                for key in model_output_keys:
                    # u to u(x, y)
                    expr_key = (
                        f"{key}({', '.join([str(arg) for arg in self.expr.args])})"
                    )
                    inputs_dict[expr_key] = model_output[key]
                break
        else:  # no break
            raise ValueError(
                f"The model with output_keys = {str(self.expr.func)} is not found."
            )
        return inputs_dict


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

    def forward(self, inputs_dict: Dict, models_dict: Dict[str, nn.Layer]):
        inputs_dict[str(self.expr)] = paddle.to_tensor(
            float(self.expr), dtype=paddle.get_default_dtype()
        )
        return inputs_dict


class ComposedFunc(nn.Layer):
    """
    Compose multiple functions into one function.

    Args:
        inputs_dict (Dict): The input tensor dictionary.
        model_dict (Dict[str, nn.Layer]): The dictionary of the models.

    Returns:
        The dictionary of the outputs of the all calculated nodes.

    Examples:
        >>> x = sympy.Symbol("x")
        >>> expr = sympy.sin(x)
        >>> func = sympy_to_function(expr)
        >>> func({x: paddle.to_tensor(0.5)})
        {'sin(x)': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            0.47942555)}
    """

    def __init__(self, nodes: List[NodeBase]):
        super().__init__()
        self.nodes = nodes

    def _derivative_to_str(self, expr: sympy.Expr) -> str:
        """
        Convert the derivative expression to string.

        Args:
            expr (sympy.Expr): The derivative expression.

        Returns:
            The string of the derivative expression.
        """
        key = str(expr.args[0].func)
        for symbol, order in expr.args[1:]:
            key += f"__{symbol}" * order
        return key

    def forward(self, inputs_dict: Dict, model_dict: Dict[str, nn.Layer] = None):
        for node in self.nodes:
            inputs_dict = node(inputs_dict, model_dict)

        last_expr = self.nodes[-1].expr
        inputs_dict["output"] = inputs_dict.pop(
            str(last_expr)
        )  # rename the last node key to output

        layer_key_maps = {}

        for key in list(inputs_dict.keys()):
            expr = sympy.sympify(key)
            if key.startswith(
                "Derivative("
            ):  # rename the derivative key Derivative(u(x, y), x) to u__x
                inputs_dict[self._derivative_to_str(expr)] = inputs_dict.pop(key)
            if key.startswith("-Derivative("):  # remove the negative derivative
                inputs_dict[
                    "-" + self._derivative_to_str(expr.args[1])
                ] = inputs_dict.pop(key)
            if isinstance(expr.func, sympy.core.function.UndefinedFunction):
                layer_key_maps[key] = str(expr.func)

        for (
            key,
            value,
        ) in layer_key_maps.items():  # rename the layer key e.g. u(x, y) to u
            for inputs_key in list(inputs_dict.keys()):
                if key in inputs_key:
                    inputs_dict[inputs_key.replace(key, value)] = inputs_dict.pop(
                        inputs_key
                    )

        return inputs_dict


def get_expression_nodes(expr: sympy.Expr) -> List[sympy.Expr]:
    """
    Convert a sympy expression to a list of sympy expressions using post-order traversal.

    Args:
        expr (sympy.Expr): the sympy expression to be converted

    Returns:
        A list of sympy expressions.

    Examples:
        >>> x = sympy.Symbol("x")
        >>> expr = sympy.sin(x) + x
        >>> nodes = get_expression_nodes(expr)
        >>> nodes
        [x, sin(x), x, x + sin(x)]

    Notes:
        This function performs a post-order traversal of the expression tree rooted at `expr`.
        The resulting list contains the sub-expressions of `expr` in the order in which they would be evaluated.
    """
    nodes = []

    def traverse_expression(expr, nodes):
        nodes.insert(0, expr)
        if expr.func == sympy.Derivative:
            nodes.insert(0, expr.args[0])
            return nodes
        for arg in expr.args:
            nodes = traverse_expression(arg, nodes)
        return nodes

    nodes = traverse_expression(expr, nodes)
    return nodes


def sympy_to_function(expr: sympy.Expr):
    """
    Convert a sympy expression to a ComposedFunc.

    Args:
        expr (sympy.Expr): the sympy expression

    Returns:
        A ComposedFunc that can execute the formula represented by the sympy expression.

    Examples:
        >>> x, y = sympy.symbols("x y")
        >>> u = sympy.Function("u")(x, y)
        >>> v = sympy.Function("v")(x, y)
        >>> expr = u.diff(x) - v.diff(x, 2) + u * v + sympy.sin(u) * sympy.cos(v)
        >>> function = sympy_to_function(expr)

        >>> class MyLayer(nn.Layer):
        >>>    def __init__(self):
        >>>        super(MyLayer, self).__init__()
        >>>        self.output_keys = ["u", "v"]
        >>>    def forward(self, x):
        >>>        x, y = x["x"], x["y"]
        >>>        u = paddle.cos(y * x)
        >>>        v = paddle.tanh(x**2)
        >>>        return {"u": u, "v": v}

        >>> x = paddle.to_tensor(np.random.randn(1, 1), stop_gradient=False, dtype="float32")
        >>> y = paddle.to_tensor(np.random.randn(1, 1), stop_gradient=False, dtype="float32")
        >>> function(inputs_dict={"x": x, "y": y}, model_dict={"u": MyLayer()})
        {'x': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
        [[-0.21531263]]),
         'y': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[-0.20731021]]),
         '-1': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                 -1.),
         'output': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[-1.08300245]]),
         'u__x': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[0.00925053]]),
         'v__x__x': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[1.97856331]]),
         '-v__x__x': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[-1.97856331]]),
         'u': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[0.99900395]]),
         'sin(u)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[0.84093243]]),
         'v': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[0.04632635]]),
         'cos(v)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[0.99892712]]),
         'u*v': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                 [[0.04628020]]),
         'sin(u)*cos(v)': Tensor(shape=[1, 1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[0.84003019]])}
    """
    expression_nodes = get_expression_nodes(expr)
    expression_nodes = [
        node for node in expression_nodes if not node.is_Symbol
    ]  # remove symbol.Symbols
    expression_nodes = list(dict.fromkeys(expression_nodes))  # remove duplicates
    nodes = []
    for node in expression_nodes:
        if isinstance(node.func, sympy.core.function.UndefinedFunction):
            nodes.append(LayerNode(node))
        elif (
            node.is_Function
            or node.is_Add
            or node.is_Mul
            or node.is_Pow
            or node.is_Derivative
        ):
            nodes.append(OperatorNode(node))
        elif node.is_Number or node.is_NumberSymbol:
            nodes.append(ConstantNode(node))
        else:
            raise NotImplementedError(
                f"The node {node} is not supported in sympy_to_function."
            )
    return ComposedFunc(nodes)
