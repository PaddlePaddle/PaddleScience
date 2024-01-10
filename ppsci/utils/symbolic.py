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
import os
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
from ppsci import equation
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

__all__ = [
    "lambdify",
]


DATA_DICT: TypeAlias = Dict[str, paddle.Tensor]

SYMPY_BUILTIN_FUNC: TypeAlias = Union[
    sp.sin,
    sp.sinh,
    sp.asin,
    sp.cos,
    sp.acos,
    sp.cosh,
    sp.tan,
    sp.atan,
    sp.atan2,
    sp.acosh,
    sp.asinh,
    sp.tanh,
    sp.atanh,
    sp.erf,
    sp.loggamma,
    sp.exp,
    sp.Pow,
    sp.log,
    sp.Max,
    sp.Min,
    sp.Abs,
    sp.Heaviside,
    sp.sign,
    sp.ceiling,
    sp.floor,
    sp.Add,
    sp.Mul,
]

SYMPY_TO_PADDLE = {
    sp.sin: paddle.sin,
    sp.sinh: paddle.sinh,
    sp.asin: paddle.asin,
    sp.cos: paddle.cos,
    sp.acos: paddle.acos,
    sp.cosh: paddle.cosh,
    sp.tan: paddle.tan,
    sp.atan: paddle.atan,
    sp.atan2: paddle.atan2,
    sp.acosh: paddle.acosh,
    sp.asinh: paddle.asinh,
    sp.tanh: paddle.tanh,
    sp.atanh: paddle.atanh,
    sp.erf: paddle.erf,
    sp.loggamma: paddle.lgamma,
    sp.exp: paddle.exp,
    sp.Pow: paddle.pow,
    sp.log: paddle.log,
    sp.Max: paddle.maximum,
    sp.Min: paddle.minimum,
    sp.Abs: paddle.abs,
    sp.Heaviside: functools.partial(paddle.heaviside, y=paddle.zeros([])),
    sp.sign: paddle.sign,
    sp.ceiling: paddle.ceil,
    sp.floor: paddle.floor,
    # NOTE: sp.Add and sp.Mul is not included here for un-alignment with sympy
    # and are implemented manually in 'OperatorNode._add_operator_func' and
    # 'OperatorNode._mul_operator_func'
}


def _numerator_of_derivative(expr: sp.Basic) -> sp.Basic:
    if not isinstance(expr, sp.Derivative):
        raise TypeError(
            f"expr({expr}) should be of type sp.Derivative, but got {type(expr)}"
        )
    if len(expr.args) <= 2:
        if expr.args[1][1] == 1:
            return expr.args[0]
        return sp.Derivative(expr.args[0], (expr.args[1][0], expr.args[1][1] - 1))
    else:
        return sp.Derivative(*expr.args[:-1])


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
        return (
            f"{self.__class__.__name__}(expr: {self.expr}, "
            f"expr_type: {type(self.expr)})"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(expr: {self.expr})"


class DetachNode(nn.Layer):
    """Class for detach operation in converted expression tree.

    Args:
        expr (sp.Basic): Sympy expression.
    """

    def __init__(self, expr: sp.Basic):
        super().__init__()
        self.expr = expr
        self.key = _cvt_to_key(self.expr)
        self.child = _cvt_to_key(self.expr.args[0])

    def forward(self, data_dict: DATA_DICT):
        if self.key in data_dict:
            return data_dict

        data_dict[self.key] = data_dict[self.child].detach()
        return data_dict


class OperatorNode(Node):
    """Class for operator node in converted expression tree.

    Args:
        expr (SYMPY_BUILTIN_FUNC): Sympy expression.
    """

    def __init__(
        self,
        expr: sp.Basic,
    ):
        super().__init__(expr)
        # preprocess children's key instead of processing at run-time in forward
        # which can reduce considerable overhead of time for calling "_cvt_to_key"
        self.childs = [_cvt_to_key(arg) for arg in self.expr.args]

        if self.expr.func == sp.Add:
            self._apply_func = self._add_operator_func
        elif self.expr.func == sp.Mul:
            self._apply_func = self._mul_operator_func
        elif self.expr.func == sp.Heaviside:
            self._apply_func = self._heaviside_operator_func
            self._auxiliary_func = SYMPY_TO_PADDLE[sp.Heaviside]
        elif self.expr.func == sp.Min:
            self._apply_func = self._minimum_operator_func
        elif self.expr.func == sp.Max:
            self._apply_func = self._maximum_operator_func
        else:
            self._apply_func = self._vanilla_operator_func
            self._auxiliary_func = SYMPY_TO_PADDLE[self.expr.func]

    def forward(self, data_dict: DATA_DICT):
        # use cache
        if self.key in data_dict:
            return data_dict

        return self._apply_func(data_dict)

    def _add_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = data_dict[self.childs[0]]
        for p in self.childs[1:]:
            data_dict[self.key] += data_dict[p]
        return data_dict

    def _mul_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = data_dict[self.childs[0]]
        for child in self.childs[1:]:
            data_dict[self.key] *= data_dict[child]
        return data_dict

    def _heaviside_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = self._auxiliary_func(data_dict[self.childs[0]])
        return data_dict

    def _minimum_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = paddle.minimum(
            data_dict[self.childs[0]], data_dict[self.childs[1]]
        )
        for i in range(2, len(self.childs)):
            data_dict[self.key] = paddle.minimum(
                data_dict[self.key],
                data_dict[self.childs[i]],
            )
        return data_dict

    def _maximum_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = paddle.maximum(
            data_dict[self.childs[0]], data_dict[self.childs[1]]
        )
        for i in range(2, len(self.childs)):
            data_dict[self.key] = paddle.maximum(
                data_dict[self.key],
                data_dict[self.childs[i]],
            )
        return data_dict

    def _vanilla_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.key] = self._auxiliary_func(
            *tuple(data_dict[child] for child in self.childs)
        )
        return data_dict


class DerivativeNode(Node):
    """Class for operator node in converted expression tree.

    Args:
        expr (SYMPY_BUILTIN_FUNC): Sympy expression.
        create_graph (bool, optional): Whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Defaults to True.
        retain_graph (Optional[bool]): Whether to retain the forward graph which
            is used to calculate the gradient. When it is True, the graph would
            be retained, in which way users can calculate backward twice for the
            same graph. When it is False, the graph would be freed. Defaults to None,
            which means it is equal to `create_graph`.
    """

    def __init__(
        self,
        expr: sp.Basic,
        create_graph: bool = True,
        retain_graph: Optional[bool] = None,
    ):
        super().__init__(expr)
        # preprocess children's key instead of processing at run-time in forward
        # which can reduce considerable overhead of time for calling "_cvt_to_key"
        self.childs = [_cvt_to_key(self.expr.args[0])] + [
            (_cvt_to_key(arg), int(order)) for (arg, order) in self.expr.args[1:]
        ]
        self.create_graph = create_graph
        self.retain_graph = retain_graph
        self._apply_func = self._derivate_operator_func

    def forward(self, data_dict: DATA_DICT):
        # use cache
        if self.key in data_dict:
            return data_dict

        return self._apply_func(data_dict)

    def _derivate_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        # NOTE: Derivative of 'sdf' function will not be executed here, which is already
        # generated in 'data_dict' during points sampling using discrete difference
        # method(see also: ppsci/geometry/geometry.py: Geometry.sdf_derivatives),
        # such as 'sdf__x', 'sdf__y'.
        data_dict[self.key] = data_dict[self.childs[0]]
        for child, order in self.childs[1:]:
            if order & 1:
                data_dict[self.key] = jacobian(
                    data_dict[self.key],
                    data_dict[child],
                    create_graph=self.create_graph,
                    retain_graph=self.retain_graph,
                )
                order -= 1
            for _ in range(0, order, 2):
                data_dict[self.key] = hessian(
                    data_dict[self.key],
                    data_dict[child],
                    create_graph=self.create_graph,
                    retain_graph=self.retain_graph,
                )
                order -= 2
        return data_dict


class FusedDerivativeNode(nn.Layer):
    """Class for operator node in converted expression tree.

    Args:
        expr (SYMPY_BUILTIN_FUNC): Sympy expression.
        create_graph (bool, optional): Whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Defaults to True.
        retain_graph (Optional[bool]): Whether to retain the forward graph which
            is used to calculate the gradient. When it is True, the graph would
            be retained, in which way users can calculate backward twice for the
            same graph. When it is False, the graph would be freed. Defaults to None,
            which means it is equal to `create_graph`.
    """

    def __init__(
        self,
        f_x_tuples: List[Tuple[Union[sp.Function, sp.Derivative], sp.Symbol]],
        create_graph: bool = True,
        retain_graph: Optional[bool] = None,
    ):
        super().__init__()
        self.expr: List[sp.Derivative] = [f.diff(x) for f, x in f_x_tuples]
        self.key: List[str] = [_cvt_to_key(expr) for expr in self.expr]
        self.create_graph = create_graph
        self.retain_graph = retain_graph

        # preprocess children's key instead of processing at run-time in forward
        # which can reduce considerable overhead of time for calling "_cvt_to_key"
        self.y_key: str = _cvt_to_key(f_x_tuples[0][0])
        self.childs: List[str] = [_cvt_to_key(x) for _, x in f_x_tuples]
        self._apply_func = self._parallel_derivate_operator_func

    def forward(self, data_dict: DATA_DICT):
        # use cache
        if all([key in data_dict for key in self.key]):
            return data_dict

        return self._apply_func(data_dict)

    def _parallel_derivate_operator_func(self, data_dict: DATA_DICT) -> DATA_DICT:
        # NOTE: Derivative of 'sdf' function will not be executed here, which is already
        # generated in 'data_dict' during points sampling using discrete difference
        # method(see also: ppsci/geometry/geometry.py: Geometry.sdf_derivatives),
        # such as 'sdf__x', 'sdf__y'.
        y_data = data_dict[self.y_key]
        xs_data = [data_dict[x_key] for x_key in self.childs]
        y_wrt_xs_grad: List[paddle.Tensor] = jacobian(
            y_data,
            xs_data,
            create_graph=self.create_graph,
            retain_graph=self.retain_graph,
        )
        for i, key in enumerate(self.key):
            data_dict[key] = y_wrt_xs_grad[i]
        return data_dict

    def __str__(self):
        return (
            f"{self.__class__.__name__}(expr: {self.expr}, "
            f"expr_type: {type(self.expr)})"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(expr: {self.expr})"


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
                "expr({expr}) should be Float/Integer/Boolean/Rational, "
                f"but got {type(self.expr)}"
            )
        self.expr = paddle.to_tensor(self.expr)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        # use cache
        if self.key in data_dict:
            return data_dict

        data_dict[self.key] = self.expr
        return data_dict

    def __str__(self):
        return (
            f"{self.__class__.__name__}(expr: {float(self.expr)}, "
            f"expr_type: {type(self.expr)})"
        )


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

    def __init__(self, callable_nodes: List[Node]):
        super().__init__()
        assert len(callable_nodes)
        self.callable_nodes = callable_nodes

    def forward(self, data_dict: DATA_DICT) -> paddle.Tensor:
        # call all callable_nodes in order
        for i, func in enumerate(self.callable_nodes):
            data_dict = func(data_dict)

        # return result of last node(root node) for target
        return data_dict[self.callable_nodes[-1].key]


def _post_traverse(cur_node: sp.Basic, nodes: List[sp.Basic]) -> List[sp.Basic]:
    """Traverse sympy expression tree in post-order.

    Args:
        cur_node (sp.Basic): Sympy expression of current node.
        nodes (List[sp.Basic]): Node list storing all tree nodes in post-order.

    Returns:
        List[sp.Basic]: Node list storing all tree nodes in post-order.
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


def _visualize_graph(nodes: List[sp.Basic], graph_filename: str):
    try:
        import pygraphviz
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install pygraphviz by steps below:\n"
            "1. apt-get install graphviz graphviz-dev\n"
            "2. python -m pip install pygraphviz"
        )

    SYMPY_BUILTIN_NAME = {
        sp.sin: "sin",
        sp.sinh: "sinh",
        sp.asin: "asin",
        sp.cos: "cos",
        sp.acos: "acos",
        sp.cosh: "cosh",
        sp.tan: "tan",
        sp.atan: "atan",
        sp.atan2: "atan2",
        sp.acosh: "acosh",
        sp.asinh: "asinh",
        sp.tanh: "tanh",
        sp.atanh: "atanh",
        sp.erf: "erf",
        sp.loggamma: "loggamma",
        sp.exp: "exp",
        sp.Pow: "Pow",
        sp.log: "log",
        sp.Max: "Max",
        sp.Min: "Min",
        sp.Abs: "Abs",
        sp.Heaviside: "Heaviside",
        sp.sign: "sign",
        sp.ceiling: "ceiling",
        sp.floor: "floor",
        sp.Add: "Add",
        sp.Mul: "Mul",
    }
    naming_counter = {k: 0 for k in SYMPY_BUILTIN_NAME}

    def get_operator_name(node):
        ret = f"{SYMPY_BUILTIN_NAME[node.func]}_{naming_counter[node.func]}"
        naming_counter[node.func] += 1
        return ret

    graph = pygraphviz.AGraph(directed=True, rankdir="TB")
    C_FUNC = "#9196f1"  # purple color function node
    C_DATA = "#feb64d"  # orange color for data node
    C_EDGE = "#000000"  # black color for edge

    def add_edge(u: str, v: str, u_color: str = C_DATA, v_color: str = C_DATA):
        """Add an edge from `u` to `v`.

        Args:
            u (str): Name of begin node u.
            v (str): Name of end node v.
            u_color (str, optional): _description_. Defaults to C_DATA.
            v_color (str, optional): _description_. Defaults to C_DATA.
        """
        graph.add_node(u, style="filled", shape="ellipse", color=u_color)
        graph.add_node(v, style="filled", shape="ellipse", color=v_color)
        graph.add_edge(u, v, color=C_EDGE, style="solid", penwidth=0.5, arrowsize=0.5)

    for node in nodes:
        if isinstance(node, tuple(SYMPY_BUILTIN_NAME.keys())):
            operator_str = get_operator_name(node)
            for arg in node.args:
                add_edge(_cvt_to_key(arg), operator_str, v_color=C_FUNC)
            add_edge(operator_str, _cvt_to_key(node), u_color=C_FUNC)
        elif isinstance(node, sp.Function):
            for arg in node.args:
                add_edge(_cvt_to_key(arg), str(node), v_color=C_FUNC)
            add_edge(str(node), _cvt_to_key(node), u_color=C_FUNC)
        elif isinstance(node, sp.Derivative):
            add_edge(str(node), _cvt_to_key(node), u_color=C_FUNC)
            add_edge(_cvt_to_key(node.args[0]), str(node), v_color=C_FUNC)
            for arg in node.args[1:]:
                add_edge(_cvt_to_key(arg[0]), str(node), v_color=C_FUNC)

    # export graph to image
    graph.layout()
    image_path = f"{graph_filename}.png"
    dot_path = f"{graph_filename}.dot"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    graph.draw(image_path, prog="dot")
    graph.write(dot_path)
    logger.message(
        f"Computational graph has been written to: {image_path} and {dot_path}, "
        "which can be visualized at: https://dreampuf.github.io/GraphvizOnline/"
    )


def derivetive_tree(
    derivative_exprs: List[sp.Derivative],
) -> List[FusedDerivativeNode]:
    class DerivativeTrie:
        def __init__(self, expr: sp.Basic):
            self.expr: sp.Basic = expr
            self.next: Dict["sp.Symbol", "DerivativeTrie"] = {}

    # build derivative trie
    trie_root = DerivativeTrie(derivative_exprs[0].args[0])
    for derivative_expr in derivative_exprs:
        cur_node = trie_root
        for (child, order) in derivative_expr.args[1:]:
            for _ in range(order):
                if child not in cur_node.next:
                    cur_node.next[child] = DerivativeTrie(cur_node.expr.diff(child))
                cur_node = cur_node.next[child]

    # walk on derivative trie
    def dfs_trie(
        node: DerivativeTrie, derivative_nodes_seq: List[FusedDerivativeNode]
    ) -> None:
        node.vis = True
        if node.next:
            derivative_nodes_seq.append(
                FusedDerivativeNode(
                    [(node.expr, name) for name in node.next],
                )
            )
        for child in node.next:
            dfs_trie(node.next[child], derivative_nodes_seq)

    derivative_nodes_seq: List[FusedDerivativeNode] = []
    dfs_trie(trie_root, derivative_nodes_seq)

    return derivative_nodes_seq


def lambdify(
    expr: Union[sp.Basic, List[sp.Basic]],
    models: Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]] = None,
    extra_parameters: Optional[Sequence[paddle.Tensor]] = None,
    graph_filename: Optional[str] = None,
    create_graph: bool = True,
    retain_graph: Optional[bool] = None,
    fuse_derivative: bool = False,
) -> Union[ComposedNode, List[ComposedNode]]:
    """Convert sympy expression to callable function.

    Args:
        expr (Union[sp.Basic, List[sp.Basic]]): Sympy expression(s) to be converted.
            will return callable functions in list if multiple expressions are given.
            else will return one single callable function.
        models (Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]]): Model(s) for
            computing forward result in `LayerNode`.
        extra_parameters (Optional[nn.ParameterList]): Extra learnable parameters.
            Defaults to None.
        graph_filename (Optional[str]): Save computational graph to `graph_filename.png`
            for given `expr`, if `graph_filename` is not None and a valid string,
            such as 'momentum_x'. Defaults to None.
        create_graph (bool, optional): Whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Defaults to True.
        retain_graph (Optional[bool]): Whether to retain the forward graph which
            is used to calculate the gradient. When it is True, the graph would
            be retained, in which way users can calculate backward twice for the
            same graph. When it is False, the graph would be freed. Defaults to None,
            which means it is equal to `create_graph`.
        fuse_derivative (bool, optional): Whether to fuse the derivative nodes.
            for example, if `expr` is 'Derivative(u, x) + Derivative(u, y)'
            It will compute grad(u, x) + grad(u, y) if fuse_derivative=False,
            else will compute sum(grad(u, [x, y])) if fuse_derivative=True as is more
            efficient in backward-graph. Defaults to False.

    Returns:
        ComposedNode: Callable object for computing expr with necessary input(s) data
            in dict given.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> import sympy as sp

        >>> a, b, c, x, y = sp.symbols("a b c x y")
        >>> u = sp.Function("u")(x, y)
        >>> v = sp.Function("v")(x, y)
        >>> z = -a + b * (c ** 2) + u * v + 2.3

        >>> model = ppsci.arch.MLP(("x", "y"), ("u", "v"), 4, 16)

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
        >>> z_tensor_sympy = ppsci.lambdify(z, model)(
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
    if not extra_parameters:
        extra_parameters = ()

    if isinstance(models, arch.ModelList):
        models = tuple(models.model_list[i] for i in range(len(models.model_list)))
    if not isinstance(models, (tuple, list)):
        models = (models,)

    def _expr_to_nodes_seq(
        single_expr: sp.Basic, graph_filename_: Optional[str] = None
    ) -> List[Node]:
        """Convert sympy expression to a sequence of nodes in topologic order.

        Args:
            single_expr (sp.Basic): Single sympy expression, such as "a+b*c".
            graph_filename_ (Optional[str]): Save computational graph to
            `/path/to/graph_filename.png` for given `expr`, if `graph_filename` is not
            None and a valid string, such as 'momentum_x'. Defaults to None.

        Returns:
            List[Node]: Sequence of callable nodes.
        """
        # NOTE: Those simplify methods may complicate given expr instead, so not use here
        # simplify expression to reduce nodes in tree
        # expr = sp.nsimplify(expr)
        # expr = sp.expand(expr)
        # expr = sp.simplify(expr)

        # remove 1.0 from sympy expression tree
        single_expr = single_expr.subs(1.0, 1)

        # convert sympy expression tree to list of nodes in post-order
        sympy_nodes: List[sp.Basic] = []
        sympy_nodes = _post_traverse(single_expr, sympy_nodes)

        # remove unnecessary symbol nodes already in input dict(except for parameter symbol)
        _parameter_names = tuple(param.name for param in extra_parameters)
        sympy_nodes = [
            node
            for node in sympy_nodes
            if (not node.is_Symbol) or (_cvt_to_key(node) in _parameter_names)
        ]

        # remove duplicated node(s) with topological order kept
        sympy_nodes = list(dict.fromkeys(sympy_nodes))

        # convert sympy node to callable node
        callable_nodes = []
        for i, node in enumerate(sympy_nodes):
            if isinstance(
                node, tuple(SYMPY_TO_PADDLE.keys()) + (sp.Add, sp.Mul, sp.Derivative)
            ):
                if isinstance(node, sp.Derivative):
                    callable_nodes.append(
                        DerivativeNode(node, create_graph, retain_graph)
                    )
                else:
                    callable_nodes.append(OperatorNode(node))
            elif isinstance(node, sp.Function):
                if node.name == equation.DETACH_FUNC_NAME:
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
                                    f"Name of function: '{node}' should be unique along given"
                                    f" models, but got same output_key: '{node.func.name}' "
                                    f"in given models[{match_index}] and models[{j}]."
                                )
                            match_index = j
                    # NOTE: Skip 'sdf' function, which should be already generated in
                    # given data_dict
                    if match_index is None and node.name != "sdf":
                        raise ValueError(
                            f"Node {node} can not match any model in given model(s)."
                        )
            elif node.is_Number or node.is_NumberSymbol:
                callable_nodes.append(ConstantNode(node))
            elif isinstance(node, sp.Symbol):
                callable_nodes.append(
                    ParameterNode(
                        node,
                        *[
                            param
                            for param in extra_parameters
                            if param.name == node.name
                        ],
                    )
                )
            else:
                raise NotImplementedError(
                    f"The node {node} is not supported in lambdify."
                )

        # NOTE: visualize computational graph using 'pygraphviz'
        if isinstance(graph_filename, str):
            _visualize_graph(sympy_nodes, os.path.join(graph_filename, graph_filename_))

        return callable_nodes

    if isinstance(expr, sp.Basic):
        callable_nodes_group = [_expr_to_nodes_seq(expr, "expr")]
    else:
        callable_nodes_group = [
            _expr_to_nodes_seq(expr_i, f"expr_{i}") for i, expr_i in enumerate(expr)
        ]

    # Fused derivatives nodes that with same function to be differentiated
    while fuse_derivative:
        candidate_derivative_nodes_pos = []

        # use 4-nested for-loop to find all potential mergable node list
        for i in range(len(callable_nodes_group)):
            for j in range(len(callable_nodes_group[i])):
                # skip node that not of derivative
                if not isinstance(
                    callable_nodes_group[i][j], DerivativeNode
                ) or not isinstance(callable_nodes_group[i][j].expr, sp.Derivative):
                    continue
                # skip sdf function for it is already computed in data_dict
                if callable_nodes_group[i][j].expr.args[0].name == "sdf":
                    continue

                candidate_derivative_nodes_pos = [[i, j]]
                for ii in range(len(callable_nodes_group)):
                    for jj in range(len(callable_nodes_group[ii])):
                        # skip node that not of Derivative
                        if not isinstance(
                            callable_nodes_group[ii][jj], DerivativeNode
                        ) or not isinstance(
                            callable_nodes_group[ii][jj].expr, sp.Derivative
                        ):
                            continue

                        # skip same node
                        if i == ii and j == jj:
                            continue

                        # has same function item
                        if (
                            callable_nodes_group[i][j].expr.args[0]
                            == callable_nodes_group[ii][jj].expr.args[0]
                        ):
                            candidate_derivative_nodes_pos.append([ii, jj])

                if len(candidate_derivative_nodes_pos) > 1:
                    break
            if len(candidate_derivative_nodes_pos) > 1:
                break

        # merge all candidate nodes into one or more FusedDerivativeNode node
        if len(candidate_derivative_nodes_pos) > 1:
            fused_seq_node = derivetive_tree(
                [
                    callable_nodes_group[gi][ni].expr
                    for gi, ni in candidate_derivative_nodes_pos
                ]
            )
            assert isinstance(
                fused_seq_node, list
            ), "'fused_seq_node' should be list of 'FusedDerivativeNode'"
            gi0, ni0 = candidate_derivative_nodes_pos[0]
            logger.debug(
                f"Fused {len(candidate_derivative_nodes_pos)} derivatives nodes: {[callable_nodes_group[ii][jj].expr for ii, jj in candidate_derivative_nodes_pos]} into"
                f" node sequence: {fused_seq_node} at position: ([{gi0}][{ni0}])"
            )
            callable_nodes_group[gi0][ni0] = fused_seq_node

            # remove merged node(except [gi0, ni0])
            for gi, ni in candidate_derivative_nodes_pos[1:]:
                # keep the last node to avoid empty callable node sequence
                # this will not effect performance for cache strategy in Node.forward
                if ni != len(callable_nodes_group[gi]) - 1:
                    callable_nodes_group[gi][ni] = None
        else:
            break

        for i in range(len(callable_nodes_group)):
            tmp = []
            for j in range(len(callable_nodes_group[i])):
                if isinstance(callable_nodes_group[i][j], Node):
                    tmp.append(callable_nodes_group[i][j])
                elif isinstance(callable_nodes_group[i][j], list) and isinstance(
                    callable_nodes_group[i][j][0], FusedDerivativeNode
                ):
                    for fuse_node in callable_nodes_group[i][j]:
                        tmp.append(fuse_node)
                elif isinstance(callable_nodes_group[i][j], FusedDerivativeNode):
                    tmp.append(callable_nodes_group[i][j])
                else:
                    assert (
                        callable_nodes_group[i][j] is None
                    ), f"Unexpected element: {callable_nodes_group[i][j]}"
            callable_nodes_group[i] = tmp

    # Compose callable nodes into one callable object
    if isinstance(expr, sp.Basic):
        return ComposedNode(callable_nodes_group[0])
    else:
        return [ComposedNode(callable_nodes) for callable_nodes in callable_nodes_group]
