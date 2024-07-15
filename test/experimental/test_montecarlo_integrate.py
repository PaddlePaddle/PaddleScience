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
from typing import List

import numpy as np
import paddle
import pytest

import ppsci

paddle.seed(1024)


@pytest.mark.parametrize(
    "fn, dim, N, integration_domains, expected",
    [
        (
            lambda x: paddle.sin(x[:, 0]) + paddle.exp(x[:, 1]),
            2,
            10000,
            [[0, 1], [-1, 1]],
            3.25152588,
        )
    ],
)
def test_montecarlo_integrate(
    fn: Callable,
    dim: int,
    N: int,
    integration_domains: List[List[float]],
    expected: float,
):
    assert np.allclose(
        ppsci.experimental.montecarlo_integrate(
            fn, dim, N, integration_domains
        ).numpy(),
        expected,
    )


if __name__ == "__main__":
    pytest.main()
