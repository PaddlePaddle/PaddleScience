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

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Tuple

import paddle
from paddle import jit
from paddle import nn

if TYPE_CHECKING:
    from ppsci import constraint
    from ppsci import validate

from ppsci.autodiff import clear


class ExpressionSolver(nn.Layer):
    """Expression Solver

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.MLP(("x", "y"), ("u", "v"), 5, 128)
        >>> expr_solver = ExpressionSolver()
    """

    def __init__(self):
        super().__init__()

    @jit.to_static
    def train_forward(
        self,
        expr_dicts: Tuple[Dict[str, Callable], ...],
        input_dicts: Tuple[Dict[str, paddle.Tensor], ...],
        model: nn.Layer,
        constraint: Dict[str, "constraint.Constraint"],
        label_dicts: Tuple[Dict[str, paddle.Tensor], ...],
        weight_dicts: Tuple[Dict[str, paddle.Tensor], ...],
    ):
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
        input_dict: Dict[str, paddle.Tensor],
        model: nn.Layer,
        validator: "validate.Validator",
        label_dict: Dict[str, paddle.Tensor],
        weight_dict: Dict[str, paddle.Tensor],
    ):
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
        expr_dict: Dict[str, Callable],
        input_dict: Dict[str, paddle.Tensor],
        model: nn.Layer,
    ):
        # model forward
        if callable(next(iter(expr_dict.values()))):
            output_dict = model(input_dict)

        # equation forward
        for name, expr in expr_dict.items():
            if callable(expr):
                output_dict[name] = expr({**output_dict, **input_dict})
            else:
                raise TypeError(f"expr type({type(expr)}) is invalid")

        # clear differentiation cache
        clear()

        # compute loss for each validator according to its' own output, label and weight
        return output_dict
