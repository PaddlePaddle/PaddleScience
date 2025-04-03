# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://github.com/omron-sinicx/transformer4sr
"""


import numpy as np
import paddle
from utils import is_tree_complete
from utils import simplify_output


class VisualizeFuncs:
    """Visualizing results as equations."""

    def __init__(self, model):
        self.model = model
        self.complete_func = is_tree_complete

    def visualize_valid_data(self, data_targets, data_values, num_valid):
        """Visulizing for given data and target."""
        for idx in range(min(num_valid, data_values.shape[0])):
            target_seq = paddle.to_tensor(
                data_targets[idx : idx + 1, :-1], dtype=paddle.get_default_dtype()
            )
            sympy_target = simplify_output(target_seq[0], "sympy")

            test_input = paddle.to_tensor(
                data_values[idx : idx + 1], dtype=paddle.get_default_dtype()
            )
            res = self.model.decode_process(test_input, self.complete_func)
            sympy_pred = simplify_output(res[0], "sympy")
            print("target", sympy_target, "pred:", sympy_pred)

    def visualize_demo(self):
        """Visulizing for a demo of equation '25*x1 + x2*log(x1)'."""
        import sympy

        C, y, x1, x2, x3, x4, x5, x6 = sympy.symbols(
            "C, y, x1, x2, x3, x4, x5, x6", real=True, positive=True
        )
        y = 25 * x1 + x2 * sympy.log(x1)
        print("The ground truth is:", y)

        x1_values = np.power(10.0, np.random.uniform(-1.0, 1.0, size=50))
        x2_values = np.power(10.0, np.random.uniform(-1.0, 1.0, size=50))
        f = sympy.lambdify([x1, x2], y)
        y_values = f(x1_values, x2_values)
        dataset = np.zeros((50, 7))
        dataset[:, 0] = y_values
        dataset[:, 1] = x1_values
        dataset[:, 2] = x2_values
        encoder_input = (
            paddle.to_tensor(data=dataset, dtype=paddle.get_default_dtype())
            .unsqueeze(axis=0)
            .unsqueeze(axis=-1)
        )
        res = self.model.decode_process(encoder_input, self.complete_func)
        sympy_pred = simplify_output(res[0], "sympy")
        print("The prediction is:", sympy_pred)
