"""
Sympy to python function conversion module
"""

import functools
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import sympy as sp
from paddle import nn
from typing_extensions import TypeAlias

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

if TYPE_CHECKING:
    from ppsci import arch


__all__ = [
    "sympy_to_function",
]


PADDLE_FUNC_MAP = {
    sp.sin: paddle.sin,
    sp.cos: paddle.cos,
    sp.exp: paddle.exp,
    sp.Pow: paddle.pow,
    sp.log: paddle.log,
    sp.tan: paddle.tan,
    sp.Max: paddle.maximum,
    sp.Min: paddle.minimum,
    sp.Abs: paddle.abs,
    sp.Heaviside: functools.partial(paddle.heaviside, y=paddle.zeros([])),
}

SYMPY_BUILTIN_FUNC: TypeAlias = Union[
    sp.sin,
    sp.cos,
    sp.exp,
    sp.Pow,
    sp.log,
    sp.tan,
    sp.Max,
    sp.Min,
    sp.Abs,
    sp.Heaviside,
]


def _cvt_to_key(expr: sp.Basic) -> str:
    """Convert sympy expression to a string key, mainly as retrieval key in dict.

    Args:
        expr (sp.Basic): Sympy expression.

    Returns:
        str: Converted string key.
    """
    if isinstance(expr, (sp.Symbol, sp.core.function.UndefinedFunction, sp.Function)):
        if hasattr(expr, "name"):
            # use name of custom function instead of itself.
            return expr.name
        else:
            str(expr)
    elif isinstance(expr, sp.Derivative):
        # convert Derivative(u(x,y),(x,2),(y,2)) to "u__x__x__y__y"
        expr_str = expr.args[0].name
        for symbol, order in expr.args[1:]:
            expr_str += f"__{symbol}" * order
        return expr_str
    else:
        return str(expr)


def _compute_single_derivate(
    dvar: paddle.Tensor, invar: paddle.Tensor, order: int
) -> paddle.Tensor:
    """Compute derivative for a single dependent variable to a single independent variable.

    Args:
        dvar (paddle.Tensor): Dependent variable.
        invar (paddle.Tensor): Independent variable.
        order (int): Order of derivative

    Returns:
        paddle.Tensor: Result of derivative d^{order}{dvar} / d{invar}^{order}.
    """
    order_left = order
    while order_left > 0:
        if order_left & 1:
            dvar = jacobian(dvar, invar)
            order_left -= 1
        if order_left >= 2:
            dvar = hessian(dvar, invar)
            order_left -= 2
    return dvar


class Node(nn.Layer):
    """The base class of the node in expression tree.

    Args:
        expr (sp.Basic): Sympy expression.
    """

    def __init__(self, expr: sp.Basic):
        super().__init__()
        self.expr = expr
        self.key = _cvt_to_key(self.expr)

    def forward(self, **kwargs):
        raise NotImplementedError("Node.forward is not implemented")

    def __str__(self):
        return f"{self.__class__.__name__}(expr: {self.expr}, expr_type: {type(self.expr)})"

    def __repr__(self):
        return f"{self.__class__.__name__}(expr: {self.expr})"


class OperatorNode(Node):
    """Class for operator node in converted expression tree.

    Args:
        expr (SYMPY_BUILTIN_FUNC): Sympy expression.
    """

    def __init__(self, expr: SYMPY_BUILTIN_FUNC):
        super().__init__(expr)
        # preprocess childs' key instead of processing at run-time
        # which can reduce considerable overhead of time for calling "_cvt_to_key"
        if self.expr.func == sp.Derivative:
            self.childs = [_cvt_to_key(self.expr.args[0])] + [
                (_cvt_to_key(arg), order) for (arg, order) in self.expr.args[1:]
            ]
        else:
            self.childs = [_cvt_to_key(arg) for arg in self.expr.args]

        if self.expr.func == sp.Add:
            self.func = self._add_operator_func
        elif self.expr.func == sp.Mul:
            self.func = self._mul_operator_func
        elif self.expr.func == sp.Derivative:
            self.func = self._derivate_operator_func
        else:
            if self.expr.func == sp.Heaviside:
                self.func = self._heaviside_operator_func
            else:
                self.func = self._vanilla_operator_func

    def forward(self, data_dict: Dict):
        # use cache
        if self.key in data_dict:
            return data_dict

        return self.func(data_dict)

    def _add_operator_func(self, data_dict):
        data_dict[self.key] = sum([data_dict[child] for child in self.childs])
        return data_dict

    def _mul_operator_func(self, data_dict):
        data_dict[self.key] = data_dict[self.childs[0]]
        for child in self.childs[1:]:
            data_dict[self.key] *= data_dict[child]
        return data_dict

    def _derivate_operator_func(self, data_dict):
        data_dict[self.key] = data_dict[self.childs[0]]
        for child, order in self.childs[1:]:
            data_dict[self.key] = _compute_single_derivate(
                data_dict[self.key],
                data_dict[child],
                order,
            )
        return data_dict

    def _heaviside_operator_func(self, data_dict):
        data_dict[self.key] = PADDLE_FUNC_MAP[sp.Heaviside](data_dict[self.childs[0]])
        return data_dict

    def _vanilla_operator_func(self, data_dict):
        data_dict[self.key] = PADDLE_FUNC_MAP[self.expr.func](
            *[data_dict[child] for child in self.childs]
        )
        return data_dict


class LayerNode(Node):
    """Class for layer node in converted expression tree.

    Args:
        expr (sp.core.function.UndefinedFunction): Sympy expression.
        model (nn.Layer): NN model for computing forward result in this node.
    """

    def __init__(self, expr: sp.core.function.UndefinedFunction, model: nn.Layer):
        super().__init__(expr)
        self.model = model

    def forward(self, data_dict: Dict):
        # use cache
        if self.key in data_dict:
            return data_dict

        output_dict = self.model(data_dict)
        data_dict.update(output_dict)

        return data_dict


class ConstantNode(Node):
    """ "Class for constant variable node in converted expression tree.

    Args:
        expr (Union[sp.Number, sp.NumberSymbol]): Number expression.
    """

    def __init__(self, expr: Union[sp.Number, sp.NumberSymbol]):
        super().__init__(expr)
        if (
            self.expr.is_Float
            or self.expr.is_Integer
            or self.expr.is_Boolean
            or self.expr.is_Rational
        ):
            self.expr = float(self.expr)
        else:
            raise TypeError(
                f"expr({expr}) should be float/int/bool, but got {type(self.expr)}"
            )
        self.expr = paddle.to_tensor(self.expr)

    def forward(self, data_dict: Dict):
        # use cache
        if self.key in data_dict:
            return data_dict

        data_dict[self.key] = self.expr
        return data_dict


class ComposedNode(nn.Layer):
    """
    Compose list of several callable objects together.
    """

    def __init__(self, target: str, funcs: List[Node]):
        super().__init__()
        self.funcs = funcs
        self.target = target

    def forward(self, data_dict: Dict):
        # call all funcs in order
        for func in self.funcs:
            data_dict = func(data_dict)

        # return result of last node(root node) for target
        return data_dict[self.funcs[-1].key]


def _post_traverse(cur_node: sp.Basic, nodes: List[sp.Basic]) -> List[sp.Basic]:
    """Traverse sympy expression tree in postorder.

    Args:
        cur_node (sp.Basic): Sympy expression of current node.
        nodes (List[sp.Basic]): Node list storing all tree nodes in postorder.

    Returns:
        List[sp.Basic]: Node list storing all tree nodes in postorder.
    """
    # traverse into sub-nodes
    if isinstance(cur_node, sp.core.function.UndefinedFunction):
        nodes.append(cur_node)
    elif isinstance(cur_node, sp.Function):
        for arg in cur_node.args:
            nodes = _post_traverse(arg, nodes)
        nodes.append(cur_node)
    elif isinstance(cur_node, sp.Derivative):
        nodes = _post_traverse(cur_node.args[0], nodes)
        nodes.append(cur_node)
    elif isinstance(cur_node, sp.Symbol):
        return nodes
    elif isinstance(cur_node, sp.Number):
        nodes.append(cur_node)
    else:
        for arg in cur_node.args:
            nodes = _post_traverse(arg, nodes)
        nodes.append(cur_node)
    return nodes


def sympy_to_function(
    target: str,
    expr: sp.Expr,
    models: Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]] = None,
) -> ComposedNode:
    """Convert sympy expression to callable function.

    Args:
        target (str): Alias of `expr`, such as "z" for expression: "z = a + b * c".
        expr (sp.Expr): Sympy expression to be converted.
        models (Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]]): Model(s) for computing forward result in `LayerNode`.

    Returns:
        ComposedNode: Callable object for computing expr with necessary input(s) data in dict given.
    """

    # simplify expression to reduce nodes in tree
    expr = sp.nsimplify(expr)
    expr = sp.expand(expr)
    expr = sp.simplify(expr)

    # convert sympy expression tree to list of nodes in postorder
    sympy_nodes = []
    sympy_nodes = _post_traverse(expr, sympy_nodes)

    # remove unnecessary symbol node for already in input dict
    sympy_nodes = [node for node in sympy_nodes if not node.is_Symbol]

    # remove duplicates with topo-order kept
    sympy_nodes = list(dict.fromkeys(sympy_nodes))

    # convert sympy node to callable node
    callable_nodes = []
    for i, node in enumerate(sympy_nodes):
        logger.debug(f"tree node [{i + 1}/{len(sympy_nodes)}]: {node}")
        if isinstance(node.func, sp.core.function.UndefinedFunction):
            match = False
            for model in models:
                if str(node.func.name) in model.output_keys:
                    callable_nodes.append(LayerNode(node, model))
                    if match:
                        raise ValueError(
                            f"Function {node} can match at least 2 output key of models, which is forbidden."
                        )
                    match = True
        elif (
            isinstance(node, tuple(PADDLE_FUNC_MAP.keys()))
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

    # Compose callable nodes into one callable object
    return ComposedNode(target, callable_nodes)
