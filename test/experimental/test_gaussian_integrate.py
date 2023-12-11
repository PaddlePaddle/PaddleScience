from typing import Callable
from typing import List

import numpy as np
import paddle
import pytest

from ppsci.experimental import gaussian_integrate

paddle.seed(1024)


@pytest.mark.parametrize(
    "fn,dim,integration_domains,antideriv_func",
    [
        (lambda x: paddle.exp(x), 1, [[-np.pi, np.pi * 2]], lambda x: np.exp(x)),
        (
            lambda x: paddle.sin(x[:, 0]) + paddle.cos(x[:, 1]),
            2,
            [[-np.pi, np.pi * 2], [-10, -np.pi * 3]],
            lambda x, y: -y * np.cos(x) + x * np.sin(y),
        ),
    ],
)
@pytest.mark.parametrize("N", [int(1e2 + 1), int(1e3 + 1), int(1e4 + 1)])
def test_gaussian_integrate(
    fn: Callable,
    dim: int,
    N: int,
    integration_domains: List[List[float]],
    antideriv_func: Callable,
):
    integrate_result = gaussian_integrate(fn, dim, N, integration_domains)
    if dim == 1:
        a, b = integration_domains[0][0], integration_domains[0][1]
        reference_result = antideriv_func(b) - antideriv_func(a)
    elif dim == 2:
        a, b, c, d = (
            integration_domains[0][0],
            integration_domains[0][1],
            integration_domains[1][0],
            integration_domains[1][1],
        )
        reference_result = (
            antideriv_func(b, d)
            - antideriv_func(a, d)
            - antideriv_func(b, c)
            + antideriv_func(a, c)
        )
    else:
        raise NotImplementedError
    assert np.allclose(integrate_result.numpy(), reference_result)


if __name__ == "__main__":
    pytest.main()
