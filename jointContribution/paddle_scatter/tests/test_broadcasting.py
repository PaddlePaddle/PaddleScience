from itertools import product

import paddle
import pytest
from paddle_scatter import scatter
from paddle_scatter.tests.testing import places
from paddle_scatter.tests.testing import reductions


@pytest.mark.parametrize("reduce,place", product(reductions, places))
def test_broadcasting(reduce, place):
    paddle.set_device(place)

    B, C, H, W = (4, 3, 8, 8)

    src = paddle.randn((B, C, H, W))
    index = paddle.randint(0, H, (H,)).astype(paddle.int64)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.shape == [B, C, H, W]

    src = paddle.randn((B, C, H, W))
    index = paddle.randint(0, H, (B, 1, H, W)).astype(paddle.int64)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.shape == [B, C, H, W]

    src = paddle.randn((B, C, H, W))
    index = paddle.randint(0, H, (H,)).astype(paddle.int64)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.shape == [B, C, H, W]
