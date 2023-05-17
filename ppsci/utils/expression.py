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

import paddle
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
    def forward(self, expr_dict_list, input_dict_list, model):
        output_dict_list = []
        intput_dict_total = {}
        beg_end = [
            0,
        ]
        for i, input_dict in enumerate(input_dict_list):
            for k, v in input_dict.items():
                if k not in intput_dict_total:
                    intput_dict_total[k] = v
                else:
                    intput_dict_total[k] = paddle.concat(
                        (intput_dict_total[k], v), axis=0
                    )
            beg_end.append(beg_end[-1] + (next(iter(input_dict.values())).shape[0]))

        output_dict_total = model(intput_dict_total)

        for i, expr_dict in enumerate(expr_dict_list):
            # model forward
            if callable(next(iter(expr_dict.values()))):
                beg, end = beg_end[i], beg_end[i + 1]
                output_dict = {k: v[beg:end] for k, v in output_dict_total.items()}

            # equation forward
            for name, expr in expr_dict.items():
                if callable(expr):
                    output_tensor = expr({**output_dict, **input_dict_list[i]})
                    output_dict[name] = output_tensor
                else:
                    raise TypeError(f"expr type({type(expr)}) is invalid")

            output_dict_list.append(output_dict)

            # clear differentiation cache
            clear()

        return output_dict_list
