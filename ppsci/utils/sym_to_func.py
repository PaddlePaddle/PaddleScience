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

"""
Sympy to python function conversion module
"""

from __future__ import annotations

import functools
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import paddle
import sympy as sp
from paddle import nn
from typing_extensions import TypeAlias

from ppsci import arch
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian

__all__ = [
    "sympy_to_function",
]


DATA_DICT: TypeAlias = Dict[str, paddle.Tensor]

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
            return str(expr)
    elif isinstance(expr, sp.Derivative):
        # convert Derivative(u(x,y),(x,2),(y,2)) to "u__x__x__y__y"
        expr_str = expr.args[0].name
        for symbol, order in expr.args[1:]:
            expr_str += f"__{symbol}" * order
        return expr_str
    else:
        return str(expr)


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


class DetachNode(nn.Layer):
    """Class for detach node in converted expression tree.

    Args:
        expr (sp.Basic): Sympy expression.
    """

    def __init__(self, expr: sp.Basic):
        super().__init__()
        self.expr = expr
        self.key = _cvt_to_key(self.expr)
        self.key_detach = self.key + "_detach"

    def forward(self, data_dict: DATA_DICT):
        if self.key_detach in data_dict:
            return data_dict

        data_dict[self.key_detach] = data_dict[self.key].detach()
        return data_dict


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
            self._operator_func = self._add_operator_func
        elif self.expr.func == sp.Mul:
            self._operator_func = self._mul_operator_func
        elif self.expr.func == sp.Derivative:
            self._operator_func = self._derivate_operator_func
        else:
            if self.expr.func == sp.Heaviside:
                self._operator_func = self._heaviside_operator_func
                self._compute_func = PADDLE_FUNC_MAP[sp.Heaviside]
            else:
                self._operator_func = self._vanilla_operator_func
                self._compute_func = PADDLE_FUNC_MAP[self.expr.func]

    def forward(self, data_dict: DATA_DICT):
        # use cache
        if self.key in data_dict:
            return data_dict

        return self._operator_func(data_dict)

    def _add_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = sum([data_dict[child] for child in self.childs])
        return data_dict

    def _mul_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = data_dict[self.childs[0]]
        for child in self.childs[1:]:
            data_dict[self.key] *= data_dict[child]
        return data_dict

    def _derivate_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = data_dict[self.childs[0]]
        for child, order in self.childs[1:]:
            if order & 1:
                data_dict[self.key] = jacobian(data_dict[self.key], data_dict[child])
                order -= 1
            for _ in range(0, order, 2):
                data_dict[self.key] = hessian(data_dict[self.key], data_dict[child])
                order -= 2
        return data_dict

    def _heaviside_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = self._compute_func(data_dict[self.childs[0]])
        return data_dict

    def _vanilla_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = self._compute_func(
            *tuple(data_dict[child] for child in self.childs)
        )
        return data_dict


class LayerNode(Node):
    """Class for layer node in converted expression tree.

    Args:
        expr (sp.core.function.UndefinedFunction): Sympy expression.
        model (arch.Arch): NN model for computing forward result in this node.
    """

    def __init__(
        self,
        expr: sp.core.function.UndefinedFunction,
        model: arch.Arch,
    ):
        super().__init__(expr)
        self.model = model

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        # use cache
        if self.key in data_dict:
            return data_dict

        output_dict = self.model(data_dict)
        data_dict.update(output_dict)

        return data_dict


class ConstantNode(Node):
    """Class for constant variable node in converted expression tree.

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
                f"expr({expr}) should be Float/Integer/Boolean/Rational, but got {type(self.expr)}"
            )
        self.expr = paddle.to_tensor(self.expr)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        # use cache
        if self.key in data_dict:
            return data_dict

        data_dict[self.key] = self.expr
        return data_dict


class ParameterNode(Node):
    """Class for constant variable node in converted expression tree.

    Args:
        expr (sp.Symbol): Parameter expression.
        parameter (paddle.framework.io.EagerParamBase): Parameter tensor.
    """

    def __init__(self, expr: sp.Symbol, parameter: paddle.framework.io.EagerParamBase):
        super().__init__(expr)
        self.parameter = parameter

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = self.parameter
        return data_dict


class ComposedNode(nn.Layer):
    """
    Compose list of several callable objects together.
    """

    def __init__(self, funcs: List[Node]):
        super().__init__()
        self.funcs = funcs

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
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
    if isinstance(cur_node, sp.Function):
        for arg in cur_node.args:
            nodes = _post_traverse(arg, nodes)
        nodes.append(cur_node)
    elif isinstance(cur_node, sp.Derivative):
        nodes = _post_traverse(cur_node.args[0], nodes)
        nodes.append(cur_node)
    elif isinstance(cur_node, sp.Symbol):
        nodes.append(cur_node)
        return nodes
    elif isinstance(cur_node, sp.Number):
        nodes.append(cur_node)
    else:
        for arg in cur_node.args:
            nodes = _post_traverse(arg, nodes)
        nodes.append(cur_node)
    return nodes


def sympy_to_function(
    expr: sp.Expr,
    models: Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]] = None,
    extra_parameters: Optional[Sequence[paddle.Tensor]] = None,
) -> ComposedNode:
    """Convert sympy expression to callable function.

    Args:
        expr (sp.Expr): Sympy expression to be converted.
        models (Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]]): Model(s) for computing forward result in `LayerNode`.
        # detach_keys (Optional[Tuple[str, ...]], optional): Keys which will be detached in computation. Defaults to None.
        extra_parameters (Optional[nn.ParameterList], optional): Extra learnable parameters. Defaults to None.

    Returns:
        ComposedNode: Callable object for computing expr with necessary input(s) data in dict given.

    Examples:
        >>> import paddle
        >>> import sympy as sp
        >>> from ppsci import arch
        >>> from ppsci.utils import sym_to_func

        >>> a, b, c, x, y = sp.symbols("a b c x y")
        >>> u = sp.Function("u")(x, y)
        >>> v = sp.Function("v")(x, y)
        >>> z = -a + b * (c ** 2) + u * v + 2.3

        >>> model = arch.MLP(("x", "y"), ("u", "v"), 4, 16)

        >>> batch_size = 13
        >>> a_tensor = paddle.randn([batch_size, 1])
        >>> b_tensor = paddle.randn([batch_size, 1])
        >>> c_tensor = paddle.randn([batch_size, 1])
        >>> x_tensor = paddle.randn([batch_size, 1])
        >>> y_tensor = paddle.randn([batch_size, 1])

        >>> model_output_dict = model({"x": x_tensor, "y": y_tensor})
        >>> u_tensor, v_tensor = model_output_dict["u"], model_output_dict["v"]

        >>> z_tensor_manually = (
        ...     -a_tensor + b_tensor * (c_tensor ** 2)
        ...     + u_tensor * v_tensor + 2.3
        ... )
        >>> z_tensor_sympy = sym_to_func.sympy_to_function(z, model)(
        ...     {
        ...         "a": a_tensor,
        ...         "b": b_tensor,
        ...         "c": c_tensor,
        ...         "x": x_tensor,
        ...         "y": y_tensor,
        ...     }
        ... )

        >>> paddle.allclose(z_tensor_manually, z_tensor_sympy).item()
        True
    """

    # NOTE: Those simplify methods seem complicate given expr instead, so not use them here
    # simplify expression to reduce nodes in tree
    # expr = sp.nsimplify(expr)
    # expr = sp.expand(expr)
    # expr = sp.simplify(expr)

    # convert sympy expression tree to list of nodes in postorder
    sympy_nodes = []
    sympy_nodes = _post_traverse(expr, sympy_nodes)

    # remove unnecessary symbol node for already in input dict(except for paramter symbol)
    _parameter_names = tuple(param.name for param in extra_parameters)
    sympy_nodes = [
        node
        for node in sympy_nodes
        if (not node.is_Symbol) or (_cvt_to_key(node) in _parameter_names)
    ]

    # remove duplicates with topo-order kept
    sympy_nodes = list(dict.fromkeys(sympy_nodes))

    if isinstance(models, arch.ModelList):
        models = tuple(models.model_list[i] for i in range(len(models.model_list)))
    if not isinstance(models, (tuple, list)):
        models = (models,)

    # convert sympy node to callable node
    callable_nodes = []
    for i, node in enumerate(sympy_nodes):
        if (
            isinstance(node, tuple(PADDLE_FUNC_MAP.keys()))
            or node.is_Add
            or node.is_Mul
            or node.is_Derivative
            or node.is_Pow
        ):
            callable_nodes.append(OperatorNode(node))
        elif isinstance(node, sp.Function):
            if node.name == "detach":
                callable_nodes.append(DetachNode(node))
            else:
                match_index = None
                for j, model in enumerate(models):
                    if str(node.func.name) in model.output_keys:
                        callable_nodes.append(
                            LayerNode(
                                node,
                                model,
                            )
                        )
                        if match_index is not None:
                            raise ValueError(
                                f"Name of function({node}) should be unique along given models,"
                                f" but got same output_key({node.func.name}) in models[{match_index}]"
                                f" and models[{j}]."
                            )
                        match_index = j
        elif node.is_Number or node.is_NumberSymbol:
            callable_nodes.append(ConstantNode(node))
        elif isinstance(node, sp.Symbol):
            callable_nodes.append(
                ParameterNode(
                    node,
                    *[param for param in extra_parameters if param.name == node.name],
                )
            )
        else:
            raise NotImplementedError(
                f"The node {node} is not supported in sympy_to_function."
            )

    # Compose callable nodes into one callable object
    return ComposedNode(callable_nodes)
