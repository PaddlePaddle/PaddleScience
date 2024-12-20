from itertools import product

import paddle
import paddle_scatter
import pytest
from paddle_scatter.tests.testing import dtypes
from paddle_scatter.tests.testing import reductions
from paddle_scatter.tests.testing import tensor

tests = [
    {
        "src": [1, 2, 3, 4, 5, 6],
        "index": [0, 0, 1, 1, 1, 3],
        "indptr": [0, 2, 5, 5, 6],
        "dim": 0,
        "sum": [3, 12, 0, 6],
        "add": [3, 12, 0, 6],
        "mean": [1.5, 4, 0, 6],
        "min": [1, 3, 0, 6],
        "max": [2, 5, 0, 6],
    },
]


@pytest.mark.skipif(paddle.device.cuda.device_count() == 0, reason="CUDA not available")
@pytest.mark.skipif(paddle.device.cuda.device_count() < 2, reason="No multiple GPUS")
@pytest.mark.parametrize("test,reduce,dtype", product(tests, reductions, dtypes))
def test_forward(test, reduce, dtype):
    paddle.set_device("gpu:1")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], paddle.int64)
    indptr = tensor(test["indptr"], paddle.int64)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    out = paddle_scatter.scatter(src, index, dim, reduce=reduce)
    assert paddle.all(out == expected)

    out = paddle_scatter.segment_coo(src, index, reduce=reduce)
    assert paddle.all(out == expected)

    out = paddle_scatter.segment_csr(src, indptr, reduce=reduce)
    assert paddle.all(out == expected)
