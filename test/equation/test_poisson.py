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
import pytest
import sympy as sp

from ppsci import arch
from ppsci import equation
from ppsci.utils import sym_to_func

__all__ = []


@pytest.mark.parametrize("dim", (2, 3))
def test_poisson(dim):
    """Test for only mean."""
    batch_size = 13
    input_dims = ("x", "y", "z")[:dim]
    output_dims = ("p",)

    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    input_data = paddle.concat([x, y], axis=1)
    if dim == 3:
        z = paddle.randn([batch_size, 1])
        z.stop_gradient = False
        input_data = paddle.concat([x, y, z], axis=1)

    # build NN model
    model = arch.MLP(input_dims, output_dims, 2, 16)

    # manually generate output
    p = model.forward_tensor(input_data)

    def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return jacobian(jacobian(y, x), x)

    # compute expected result
    expected_result = hessian(p, x) + hessian(p, y)
    if dim == 3:
        expected_result += hessian(p, z)

    # compute result using built-in Laplace module
    poisson_equation = equation.Poisson(dim=dim)
    for name, expr in poisson_equation.equations.items():
        if isinstance(expr, sp.Basic):
            poisson_equation.equations[name] = sym_to_func.sympy_to_function(
                expr,
                model,
            )

    data_dict = {
        "x": x,
        "y": y,
        "p": p,
    }
    if dim == 3:
        data_dict["z"] = z
    test_result = poisson_equation.equations["poisson"](data_dict)
    # check result whether is equal
    assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
