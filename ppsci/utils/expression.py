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

from paddle import jit
from paddle import nn

from ppsci.autodiff import clear


class ExpressionSolver(nn.Layer):
    """Expression Solver

    Args:
        input_keys (Dict[str]):Names of input keys.
        output_keys (Dict[str]):Names of output keys.
        model (nn.Layer): Model to get output variables from input variables.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.MLP(("x", "y"), ("u", "v"), 5, 128)
        >>> expr_solver = ExpressionSolver(("x", "y"), ("u", "v"), model)
    """

    def __init__(self):
        super().__init__()

    @jit.to_static
    def forward(
        self,
        expr_dict_list,
        input_dict_list,
        model,
        constraint,
        label_dict_list,
        weight_dict_list,
    ):
        output_dict_list = []
        for i, expr_dict in enumerate(expr_dict_list):
            # model forward
            if callable(next(iter(expr_dict.values()))):
                output_dict = model(input_dict_list[i])

            # equation forward
            for name, expr in expr_dict.items():
                if callable(expr):
                    output_dict[name] = expr({**output_dict, **input_dict_list[i]})
                else:
                    raise TypeError(f"expr type({type(expr)}) is invalid")

            output_dict_list.append(output_dict)

            # clear differentiation cache
            clear()

        # compute loss for each constraint according to its' own output, label and weight
        constraint_losses = []
        for i, (_, _constraint) in enumerate(constraint.items()):
            constraint_loss = _constraint.loss(
                output_dict_list[i],
                label_dict_list[i],
                weight_dict_list[i],
            )
            constraint_losses.append(constraint_loss)
        return constraint_losses
