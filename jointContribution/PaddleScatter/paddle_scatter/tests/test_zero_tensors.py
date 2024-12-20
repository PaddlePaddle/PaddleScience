from itertools import product

import paddle
import pytest
from paddle_scatter import gather_coo
from paddle_scatter import gather_csr
from paddle_scatter import scatter
from paddle_scatter import segment_coo
from paddle_scatter import segment_csr
from paddle_scatter.tests.testing import grad_dtypes
from paddle_scatter.tests.testing import ind_dtypes
from paddle_scatter.tests.testing import places
from paddle_scatter.tests.testing import reductions
from paddle_scatter.tests.testing import tensor


@pytest.mark.parametrize(
    "reduce,dtype,ind_dtype,place", product(reductions, grad_dtypes, ind_dtypes, places)
)
def test_zero_elements(reduce, dtype, ind_dtype, place):
    paddle.set_device(place)
    x = paddle.randn([0, 0, 0, 16], dtype=dtype)
    x.stop_gradient = False
    index = tensor([], ind_dtype)
    indptr = tensor([], ind_dtype)

    out = scatter(x, index, dim=0, dim_size=0, reduce=reduce)
    out.backward(paddle.randn(out.shape, out.dtype))
    assert out.shape == [0, 0, 0, 16]

    out = segment_coo(x, index, dim_size=0, reduce=reduce)
    out.backward(paddle.randn(out.shape, out.dtype))
    assert out.shape == [0, 0, 0, 16]

    out = gather_coo(x, index)
    out.backward(paddle.randn(out.shape, out.dtype))
    assert out.shape == [0, 0, 0, 16]

    out = segment_csr(x, indptr, reduce=reduce)
    out.backward(paddle.randn(out.shape, out.dtype))
    assert out.shape == [0, 0, 0, 16]

    out = gather_csr(x, indptr)
    out.backward(paddle.randn(out.shape, out.dtype))
    assert out.shape == [0, 0, 0, 16]
