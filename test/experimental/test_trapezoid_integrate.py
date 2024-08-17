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

from typing import Callable

import numpy as np
import paddle
import pytest
from typing_extensions import Literal

from ppsci.experimental import trapezoid_integrate

paddle.seed(1024)


def trapezoid_sum_test(y, x, dx):
    dx = 1 if not dx else dx
    y = y.numpy()
    res = []
    for i in range(len(y) - 1):
        res.append((y[i] + y[i + 1]) * dx / 2)
    return np.sum(np.array(res))


def trapezoid_cum_test(y, x, dx):
    dx = 1 if not dx else dx
    y = y.numpy()
    res = []
    for i in range(len(y) - 1):
        res.append((y[i] + y[i + 1]) * dx / 2)
    return np.cumsum(np.array(res))


def trapezoid_x_test(y, x, dx):
    dx = 1 if not dx else dx
    y = y.numpy()
    res = []
    for yi in y:
        res_i = []
        for i in range(len(yi) - 1):
            res_i.append((yi[i] + yi[i + 1]) * (x[i + 1] - x[i]) / 2)
        res.append(res_i)
    return np.sum(np.array(res), axis=1)


@pytest.mark.parametrize(
    "y,x,dx,axis,mode,antideriv_func",
    [
        (
            paddle.to_tensor([0, 1, 2, 3, 4, 5], dtype="float32"),
            None,
            None,
            -1,
            "sum",
            trapezoid_sum_test,
        ),
        (
            paddle.to_tensor([0, 1, 2, 3, 4, 5], dtype="float32"),
            None,
            2,
            -1,
            "cumsum",
            trapezoid_cum_test,
        ),
        (
            paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="float32"),
            paddle.to_tensor([0, 1, 2], dtype="float32"),
            None,
            1,
            "sum",
            trapezoid_x_test,
        ),
    ],
)
def test_trapezoid_integrate(
    y: paddle.Tensor,
    x: paddle.Tensor,
    dx: float,
    axis: int,
    mode: Literal["sum", "cumsum"],
    antideriv_func: Callable,
):
    integrate_result = trapezoid_integrate(y, x, dx, axis, mode)
    reference_result = antideriv_func(y, x, dx)
    assert np.allclose(integrate_result.numpy(), reference_result)


if __name__ == "__main__":
    pytest.main()
