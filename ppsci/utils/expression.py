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

import types
from typing import Union

import paddle
import sympy

from ppsci.autodiff import clear
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian


class ExpressionSolver(paddle.nn.Layer):
    """Expression Solver

    Args:
        input_keys (Dict[str]): List of string for input keys.
        output_keys (Dict[str]): List of string for output keys.
        model (nn.Layer): Model to get output variables from input variables.
    """

    def __init__(self, input_keys, output_keys, model):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.model = model
        self.expr_dict = {}
        self.output_dict = {}

    def solve_expr(self, expr: sympy.Basic) -> Union[float, paddle.Tensor]:
        """Evaluates the value of the expression recursively in the expression tree
         by post-order traversal.

        Args:
            expr (sympy.Basic): Expression.

        Returns:
            Union[float, paddle.Tensor]: Value of current expression `expr`.
        """
        # already computed in output_dict(including input data)
        if getattr(expr, "name", None) in self.output_dict:
            return self.output_dict[expr.name]

        # compute output from model
        if isinstance(expr, sympy.Symbol):
            if expr.name in self.model.output_keys:
                out_dict = self.model(self.output_dict)
                self.output_dict.update(out_dict)
                return self.output_dict[expr.name]
            else:
                raise ValueError(f"varname {expr.name} not exist!")

        # compute output from model
        elif isinstance(expr, sympy.Function):
            out_dict = self.model(self.output_dict)
            self.output_dict.update(out_dict)
            return self.output_dict[expr.name]

        # compute derivative
        elif isinstance(expr, sympy.Derivative):
            ys = self.solve_expr(expr.args[0])
            ys_name = expr.args[0].name
            if ys_name not in self.output_dict:
                self.output_dict[ys_name] = ys
            xs = self.solve_expr(expr.args[1][0])
            xs_name = expr.args[1][0].name
            if xs_name not in self.output_dict:
                self.output_dict[xs_name] = xs
            order = expr.args[1][1]
            if order == 1:
                der = jacobian(self.output_dict[ys_name], self.output_dict[xs_name])
                der_name = f"{ys_name}__{xs_name}"
            elif order == 2:
                der = hessian(self.output_dict[ys_name], self.output_dict[xs_name])
                der_name = f"{ys_name}__{xs_name}__{xs_name}"
            else:
                raise NotImplementedError(
                    f"Expression {expr} has derivative order({order}) >=3, "
                    f"which is not implemented yet"
                )
            if der_name not in self.output_dict:
                self.output_dict[der_name] = der
            return der

        # return single python number directly for leaf node
        elif isinstance(expr, sympy.Number):
            return float(expr)

        # compute sub-nodes value and merge by addition
        elif isinstance(expr, sympy.Add):
            results = [self.solve_expr(arg) for arg in expr.args]
            out = results[0]
            for i in range(1, len(results)):
                out = out + results[i]
            return out

        # compute sub-nodes value and merge by multiplication
        elif isinstance(expr, sympy.Mul):
            results = [self.solve_expr(arg) for arg in expr.args]
            out = results[0]
            for i in range(1, len(results)):
                out = out * results[i]
            return out

        # compute sub-nodes value and merge by power
        elif isinstance(expr, sympy.Pow):
            results = [self.solve_expr(arg) for arg in expr.args]
            return results[0] ** results[1]
        else:
            raise ValueError(
                f"Expression {expr} of type({type(expr)}) can't be solved yet."
            )

    def forward(self, input_dict):
        self.output_dict = input_dict
        if isinstance(next(iter(self.expr_dict.values())), types.FunctionType):
            model_output_dict = self.model(input_dict)
            self.output_dict.update(model_output_dict)

        for name, expr in self.expr_dict.items():
            if isinstance(expr, sympy.Basic):
                self.output_dict[name] = self.solve_expr(expr)
            elif isinstance(expr, types.FunctionType):
                self.output_dict[name] = expr(self.output_dict)
            else:
                raise TypeError(f"expr type({type(expr)}) is invalid")

        # clear differentiation cache
        clear()

        return {k: self.output_dict[k] for k in self.output_keys}

    def add_target_expr(self, expr, expr_name):
        self.expr_dict[expr_name] = expr

    def __str__(self):
        return f"input: {self.input_keys}, output: {self.output_keys}\n" + "\n".join(
            [f"{name} = {expr}" for name, expr in self.expr_dict.items()]
        )
