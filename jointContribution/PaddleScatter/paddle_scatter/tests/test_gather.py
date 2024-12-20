from itertools import product

import paddle
import pytest
from paddle_scatter import gather_coo
from paddle_scatter import gather_csr
from paddle_scatter.tests.testing import device
from paddle_scatter.tests.testing import dtypes
from paddle_scatter.tests.testing import dtypes_half
from paddle_scatter.tests.testing import ind_dtypes
from paddle_scatter.tests.testing import places
from paddle_scatter.tests.testing import tensor

tests = [
    {
        "src": [1, 2, 3, 4],
        "index": [0, 0, 1, 1, 1, 3],
        "indptr": [0, 2, 5, 5, 6],
        "expected": [1, 1, 2, 2, 2, 4],
        "expected_grad": [2, 3, 0, 1],
    },
    {
        "src": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "index": [0, 0, 1, 1, 1, 3],
        "indptr": [0, 2, 5, 5, 6],
        "expected": [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [7, 8]],
        "expected_grad": [[2, 2], [3, 3], [0, 0], [1, 1]],
    },
    {
        "src": [[1, 3, 5, 7], [2, 4, 6, 8]],
        "index": [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
        "indptr": [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
        "expected": [[1, 1, 3, 3, 3, 7], [2, 2, 2, 4, 4, 6]],
        "expected_grad": [[2, 3, 0, 1], [3, 2, 1, 0]],
    },
    {
        "src": [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
        "index": [[0, 0, 1], [0, 2, 2]],
        "indptr": [[0, 2, 3, 3], [0, 1, 1, 3]],
        "expected": [[[1, 2], [1, 2], [3, 4]], [[7, 9], [12, 13], [12, 13]]],
        "expected_grad": [[[2, 2], [1, 1], [0, 0]], [[1, 1], [0, 0], [2, 2]]],
    },
    {
        "src": [[1], [2]],
        "index": [[0, 0], [0, 0]],
        "indptr": [[0, 2], [0, 2]],
        "expected": [[1, 1], [2, 2]],
        "expected_grad": [[2], [2]],
    },
    {
        "src": [[[1, 1]], [[2, 2]]],
        "index": [[0, 0], [0, 0]],
        "indptr": [[0, 2], [0, 2]],
        "expected": [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
        "expected_grad": [[[2, 2]], [[2, 2]]],
    },
]


@pytest.mark.parametrize(
    "test,dtype,ind_dtype,place", product(tests, dtypes, ind_dtypes, places)
)
def test_forward(test, dtype, ind_dtype, place):
    paddle.set_device(place)
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    indptr = tensor(test["indptr"], ind_dtype)
    expected = tensor(test["expected"], dtype)

    out = gather_csr(src, indptr)
    assert paddle.all(out == expected)

    out = gather_coo(src, index)
    assert paddle.all(out == expected)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize(
    "test,dtype,ind_dtype", product(tests, dtypes_half, ind_dtypes)
)
def test_forward_half(test, dtype, ind_dtype):
    paddle.set_device("gpu")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    indptr = tensor(test["indptr"], ind_dtype)
    expected = tensor(test["expected"], dtype)

    out = gather_csr(src, indptr)
    assert paddle.all(out == expected)

    out = gather_coo(src, index)
    assert paddle.all(out == expected)


@pytest.mark.parametrize("test,place", product(tests, places))
def test_backward(test, place):
    paddle.set_device(place)
    index = tensor(test["index"], paddle.int64)
    indptr = tensor(test["indptr"], paddle.int64)
    exp_grad = tensor(test["expected_grad"], paddle.float64)

    src = tensor(test["src"], paddle.float64)
    src.stop_gradient = False
    out = gather_csr(src, indptr)
    out.backward()
    assert paddle.all(src.grad == exp_grad)

    src = tensor(test["src"], paddle.float64)
    src.stop_gradient = False
    out = gather_coo(src, index)
    out.backward()
    assert paddle.all(src.grad == exp_grad)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize("test", tests)
def test_backward_half(test):
    paddle.set_device("gpu")
    index = tensor(test["index"], paddle.int64)
    indptr = tensor(test["indptr"], paddle.int64)
    exp_grad = tensor(test["expected_grad"], paddle.float16)

    src = tensor(test["src"], paddle.float16)
    src.stop_gradient = False
    out = gather_csr(src, indptr)
    out.backward()
    assert paddle.all(src.grad == exp_grad)

    src = tensor(test["src"], paddle.float16)
    src.stop_gradient = False
    out = gather_coo(src, index)
    out.backward()
    assert paddle.all(src.grad == exp_grad)


@pytest.mark.parametrize(
    "test,dtype,ind_dtype,place", product(tests, dtypes, ind_dtypes, places)
)
def test_out(test, dtype, ind_dtype, place):
    paddle.set_device(place)
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    indptr = tensor(test["indptr"], ind_dtype)
    expected = tensor(test["expected"], dtype)

    size = src.shape
    size[index.dim() - 1] = index.shape[-1]
    out = paddle.full(size, -2).astype(dtype)

    gather_csr(src, indptr, out)
    assert paddle.all(out == expected)

    out.fill_(-2)

    gather_coo(src, index, out)
    assert paddle.all(out == expected)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize(
    "test,dtype,ind_dtype", product(tests, dtypes_half, ind_dtypes)
)
def test_out_half(test, dtype, ind_dtype):
    paddle.set_device("gpu")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    indptr = tensor(test["indptr"], ind_dtype)
    expected = tensor(test["expected"], dtype)

    size = src.shape
    size[index.dim() - 1] = index.shape[-1]
    out = paddle.full(size, -2).astype(dtype)

    gather_csr(src, indptr, out)
    assert paddle.all(out == expected)

    out.fill_(-2)

    gather_coo(src, index, out)
    assert paddle.all(out == expected)


@pytest.mark.parametrize(
    "test,dtype,ind_dtype,place", product(tests, dtypes, ind_dtypes, places)
)
def test_non_contiguous(test, dtype, ind_dtype, place):
    paddle.set_device(place)
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    indptr = tensor(test["indptr"], ind_dtype)
    expected = tensor(test["expected"], dtype)

    if src.dim() > 1:
        shape = list(range(src.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        src = src.transpose(shape).contiguous().transpose(shape)
    if index.dim() > 1:
        shape = list(range(index.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        index = index.transpose(shape).contiguous().transpose(shape)
    if indptr.dim() > 1:
        shape = list(range(indptr.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        indptr = indptr.transpose(shape).contiguous().transpose(shape)

    out = gather_csr(src, indptr)
    assert paddle.all(out == expected)

    out = gather_coo(src, index)
    assert paddle.all(out == expected)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize(
    "test,dtype,ind_dtype", product(tests, dtypes_half, ind_dtypes)
)
def test_non_contiguous_half(test, dtype, ind_dtype):
    paddle.set_device("gpu")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    indptr = tensor(test["indptr"], ind_dtype)
    expected = tensor(test["expected"], dtype)

    if src.dim() > 1:
        shape = list(range(src.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        src = src.transpose(shape).contiguous().transpose(shape)
    if index.dim() > 1:
        shape = list(range(index.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        index = index.transpose(shape).contiguous().transpose(shape)
    if indptr.dim() > 1:
        shape = list(range(indptr.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        indptr = indptr.transpose(shape).contiguous().transpose(shape)

    out = gather_csr(src, indptr)
    assert paddle.all(out == expected)

    out = gather_coo(src, index)
    assert paddle.all(out == expected)
