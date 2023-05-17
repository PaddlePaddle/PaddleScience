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
    def forward(self, expr_dict, input_dict, model):
        output_dict = {k: v for k, v in input_dict.items()}

        # model forward
        if callable(next(iter(expr_dict.values()))):
            model_output_dict = model(input_dict)
            output_dict.update(model_output_dict)

        # equation forward
        for name, expr in expr_dict.items():
            if callable(expr):
                output_dict[name] = expr(output_dict)
            else:
                raise TypeError(f"expr type({type(expr)}) is invalid")

        # clear differentiation cache
        clear()

        return output_dict
