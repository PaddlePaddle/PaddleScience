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
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import sympy as sp
from paddle import jit
from paddle import nn
from typing_extensions import TypeAlias

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

if TYPE_CHECKING:
    from ppsci import constraint
    from ppsci import validate

from ppsci.autodiff import clear


class ExpressionSolver(nn.Layer):
    """Expression computing helper, which compute named result according to corresponding
    function and related inputs.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.MLP(("x", "y"), ("u", "v"), 5, 128)
        >>> expr_solver = ExpressionSolver()
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use train_forward/eval_forward/visu_forward instead of forward."
        )

    @jit.to_static
    def train_forward(
        self,
        expr_dicts: Tuple[Dict[str, Callable], ...],
        input_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
        model: nn.Layer,
        constraint: Dict[str, "constraint.Constraint"],
        label_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
        weight_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
    ) -> Tuple["paddle.Tensor", ...]:
        """Forward computation for training, including model forward and equation
        forward.

        Args:
            expr_dicts (Tuple[Dict[str, Callable], ...]): Tuple of expression dicts.
            input_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of input dicts.
            model (nn.Layer): NN model.
            constraint (Dict[str, "constraint.Constraint"]): Constraint dict.
            label_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of label dicts.
            weight_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of weight dicts.

        Returns:
            Tuple[paddle.Tensor, ...]: Tuple of losses for each constraint.
        """
        output_dicts = []
        for i, expr_dict in enumerate(expr_dicts):
            # model forward
            if callable(next(iter(expr_dict.values()))):
                output_dict = model(input_dicts[i])

            # equation forward
            for name, expr in expr_dict.items():
                if name not in label_dicts[i]:
                    continue
                if callable(expr):
                    output_dict[name] = expr({**output_dict, **input_dicts[i]})
                else:
                    raise TypeError(f"expr type({type(expr)}) is invalid")

            # put field 'area' into output_dict
            if "area" in input_dicts[i]:
                output_dict["area"] = input_dicts[i]["area"]

            output_dicts.append(output_dict)

            # clear differentiation cache
            clear()

        # compute loss for each constraint according to its' own output, label and weight
        constraint_losses = []
        for i, _constraint in enumerate(constraint.values()):
            constraint_loss = _constraint.loss(
                output_dicts[i],
                label_dicts[i],
                weight_dicts[i],
            )
            constraint_losses.append(constraint_loss)
        return constraint_losses

    @jit.to_static
    def eval_forward(
        self,
        expr_dict: Dict[str, Callable],
        input_dict: Dict[str, "paddle.Tensor"],
        model: nn.Layer,
        validator: "validate.Validator",
        label_dict: Dict[str, "paddle.Tensor"],
        weight_dict: Dict[str, "paddle.Tensor"],
    ) -> Tuple[Dict[str, "paddle.Tensor"], "paddle.Tensor"]:
        """Forward computation for evaluation, including model forward and equation
        forward.

        Args:
            expr_dict (Dict[str, Callable]): Expression dict.
            input_dict (Dict[str, paddle.Tensor]): Input dict.
            model (nn.Layer): NN model.
            validator (validate.Validator): Validator.
            label_dict (Dict[str, paddle.Tensor]): Label dict.
            weight_dict (Dict[str, paddle.Tensor]): Weight dict.

        Returns:
            Tuple[Dict[str, paddle.Tensor], paddle.Tensor]: Result dict and loss for
                given validator.
        """
        # model forward
        if callable(next(iter(expr_dict.values()))):
            output_dict = model(input_dict)

        # equation forward
        for name, expr in expr_dict.items():
            if name not in label_dict:
                continue
            if callable(expr):
                output_dict[name] = expr({**output_dict, **input_dict})
            else:
                raise TypeError(f"expr type({type(expr)}) is invalid")

        # put field 'area' into output_dict
        if "area" in input_dict:
            output_dict["area"] = input_dict["area"]

        # clear differentiation cache
        clear()

        # compute loss for each validator according to its' own output, label and weight
        validator_loss = validator.loss(
            output_dict,
            label_dict,
            weight_dict,
        )
        return output_dict, validator_loss

    def visu_forward(
        self,
        expr_dict: Optional[Dict[str, Callable]],
        input_dict: Dict[str, "paddle.Tensor"],
        model: nn.Layer,
    ) -> Dict[str, "paddle.Tensor"]:
        """Forward computation for visualization, including model forward and equation
        forward.

        Args:
            expr_dict (Optional[Dict[str, Callable]]): Expression dict.
            input_dict (Dict[str, paddle.Tensor]): Input dict.
            model (nn.Layer): NN model.

        Returns:
            Dict[str, paddle.Tensor]: Result dict for given expression dict.
        """
        # model forward
        output_dict = model(input_dict)

        if isinstance(expr_dict, dict):
            # equation forward
            for name, expr in expr_dict.items():
                if callable(expr):
                    output_dict[name] = expr({**output_dict, **input_dict})
                else:
                    raise TypeError(f"expr type({type(expr)}) is invalid")

            # clear differentiation cache
            clear()

        return output_dict


FUNC_MAP = {
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


def _single_derivate_func(dvar: paddle.Tensor, invar: paddle.Tensor, order: int):
    order_left = order
    while order_left > 0:
        if order_left >= 2:
            dvar = hessian(dvar, invar)
            order_left -= 2
        else:
            dvar = jacobian(dvar, invar)
            order_left -= 1
    return dvar


def _cvt_to_key(sympy_node: sp.Basic):
    if isinstance(
        sympy_node, (sp.Symbol, sp.core.function.UndefinedFunction, sp.Function)
    ):
        if hasattr(sympy_node, "name"):
            # custom function
            return sympy_node.name
        else:
            str(sympy_node)
    elif isinstance(sympy_node, sp.Derivative):
        # convert Derivative(u(x,y),(x,2),(y,2)) to "u__x__x__y__y"
        expr_str = sympy_node.args[0].name
        for symbol, order in sympy_node.args[1:]:
            expr_str += f"__{symbol}" * order
        return expr_str
    else:
        return str(sympy_node)


class Node(nn.Layer):
    """The base class of the node in expression tree."""

    def __init__(self, expr: sp.Basic):
        super().__init__()
        self.expr = expr
        self.key = _cvt_to_key(self.expr)

    def forward(self, **kwargs):
        raise NotImplementedError("Node.forward is not implemented")

    def __str__(self):
        return (
            self.__class__.__name__ + f"(expr: {self.expr}), type: {type(self.expr)})"
        )


class OperatorNode(Node):
    """
    A node representing a sp operator in the computational graph.
    """

    def __init__(self, expr: SYMPY_BUILTIN_FUNC):
        super().__init__(expr)

    def forward(self, data_dict: Dict):
        if self.expr.func == sp.Add:
            data_dict[self.key] = sum(
                [data_dict[_cvt_to_key(arg)] for arg in self.expr.args]
            )
        elif self.expr.func == sp.Mul:
            data_dict[self.key] = data_dict[_cvt_to_key(self.expr.args[0])]
            for arg in self.expr.args[1:]:
                data_dict[self.key] = data_dict[self.key] * data_dict[_cvt_to_key(arg)]
        elif self.expr.func == sp.Derivative:
            if self.key in data_dict:
                return data_dict
            data_dict[self.key] = data_dict[_cvt_to_key(self.expr.args[0])]
            for symbol, order in self.expr.args[1:]:
                data_dict[self.key] = _single_derivate_func(
                    data_dict[self.key],
                    data_dict[_cvt_to_key(symbol)],
                    order,
                )
        else:
            try:
                func = FUNC_MAP[self.expr.func]
            except KeyError:
                raise NotImplementedError(
                    f"'{self.expr.func}' operator is not supported now."
                )
            if self.expr.func == sp.Heaviside:
                data_dict[self.key] = func(data_dict[_cvt_to_key(self.expr.args[0])])
            else:
                data_dict[self.key] = func(
                    *[data_dict[_cvt_to_key(arg)] for arg in self.expr.args]
                )
        return data_dict


class LayerNode(Node):
    """
    A node representing a neural network in the computational graph
    """

    def __init__(self, expr: sp.core.function.UndefinedFunction, model: nn.Layer):
        super().__init__(expr)
        self.model = model

    def forward(self, data_dict: Dict):
        if self.key in data_dict:
            return data_dict

        output_dict = self.model(data_dict)
        print("call model forward")
        for key, value in output_dict.items():
            data_dict[key] = value

        return data_dict


class ConstantNode(Node):
    """
    A node representing a constant in the computational graph.
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
        data_dict[self.key] = self.expr
        return data_dict


class ComposedFunc(nn.Layer):
    """
    Compose multiple functions into one function.
    """

    def __init__(self, target: str, funcs: List[Node]):
        super().__init__()
        self.funcs = funcs
        self.target = target

    def forward(self, data_dict: Dict):
        for func in self.funcs:
            data_dict = func(data_dict)
        return data_dict[self.funcs[-1].key]  # return the computed result of root node


def _post_traverse(cur_node: sp.Basic, nodes: List[sp.Basic]) -> List[sp.Basic]:
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


def sympy_to_function(target: str, expr: sp.Expr, models: nn.Layer) -> ComposedFunc:
    """
    Convert a sp expression to a ComposedFunc.
    """
    sympy_nodes = []
    sympy_nodes = _post_traverse(expr, sympy_nodes)
    sympy_nodes = [node for node in sympy_nodes if not node.is_Symbol]
    sympy_nodes = list(
        dict.fromkeys(sympy_nodes)
    )  # remove duplicates with topo-order kept

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
