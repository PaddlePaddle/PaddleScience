from itertools import product

import numpy as np
import paddle
import paddle_scatter
import pytest
from paddle_scatter.tests.testing import device
from paddle_scatter.tests.testing import dtypes
from paddle_scatter.tests.testing import dtypes_half
from paddle_scatter.tests.testing import ind_dtypes
from paddle_scatter.tests.testing import places
from paddle_scatter.tests.testing import reductions
from paddle_scatter.tests.testing import tensor

reductions = reductions + ["mul"]

tests = [
    {
        "src": [1, 3, 2, 4, 5, 6],
        "index": [0, 1, 0, 1, 1, 3],
        "dim": -1,
        "sum": [3, 12, 0, 6],
        "add": [3, 12, 0, 6],
        "mul": [2, 60, 1, 6],
        "mean": [1.5, 4, 0, 6],
        "min": [1, 3, 0, 6],
        "arg_min": [0, 1, 6, 5],
        "max": [2, 5, 0, 6],
        "arg_max": [2, 4, 6, 5],
        "sum_grad": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "add_grad": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "mean_grad": [0.5000, 0.33333333, 0.5000, 0.33333333, 0.33333333, 1.0000],
        "min_grad": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "max_grad": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "mul_grad": [2.0, 20.0, 1.0, 15.0, 12.0, 1.0],
    },
    {
        "src": [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],
        "index": [0, 1, 0, 1, 1, 3],
        "dim": 0,
        "sum": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "add": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "mul": [[1 * 3, 2 * 4], [5 * 7 * 9, 6 * 8 * 10], [1, 1], [11, 12]],
        "mean": [[2, 3], [7, 8], [0, 0], [11, 12]],
        "min": [[1, 2], [5, 6], [0, 0], [11, 12]],
        "arg_min": [[0, 0], [1, 1], [6, 6], [5, 5]],
        "max": [[3, 4], [9, 10], [0, 0], [11, 12]],
        "arg_max": [[2, 2], [4, 4], [6, 6], [5, 5]],
        "sum_grad": [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        "add_grad": [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        "mean_grad": [
            [0.5000, 0.5000],
            [0.33333333, 0.33333333],
            [0.5000, 0.5000],
            [0.33333333, 0.33333333],
            [0.33333333, 0.33333333],
            [1.0000, 1.0000],
        ],
        "min_grad": [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        "max_grad": [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        "mul_grad": [
            [3.0, 4.0],
            [63.0, 80.0],
            [1.0, 2.0],
            [45.0, 60.0],
            [35.0, 48.0],
            [1.0, 1.0],
        ],
    },
    {
        "src": [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]],
        "index": [[0, 1, 0, 1, 1, 3], [0, 0, 1, 0, 1, 2]],
        "dim": 1,
        "sum": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "add": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "mul": [[1 * 3, 5 * 7 * 9, 1, 11], [2 * 4 * 6, 8 * 10, 12, 1]],
        "mean": [[2, 7, 0, 11], [4, 9, 12, 0]],
        "min": [[1, 5, 0, 11], [2, 8, 12, 0]],
        "arg_min": [[0, 1, 6, 5], [0, 2, 5, 6]],
        "max": [[3, 9, 0, 11], [6, 10, 12, 0]],
        "arg_max": [[2, 4, 6, 5], [3, 4, 5, 6]],
        "sum_grad": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "add_grad": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "mean_grad": [
            [0.5000, 0.33333333, 0.5000, 0.33333333, 0.33333333, 1.0000],
            [0.33333333, 0.33333333, 0.5000, 0.33333333, 0.5000, 1.0000],
        ],
        "min_grad": [[1.0, 1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]],
        "max_grad": [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        "mul_grad": [
            [3.0, 63.0, 1.0, 45.0, 35.0, 1.0],
            [24.0, 12.0, 10.0, 8.0, 8.0, 1.0],
        ],
    },
    {
        "src": [[[1, 2], [5, 6], [3, 4]], [[10, 11], [7, 9], [12, 13]]],
        "index": [[0, 1, 0], [2, 0, 2]],
        "dim": 1,
        "sum": [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        "add": [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        "mul": [[[3, 8], [5, 6], [1, 1]], [[7, 9], [1, 1], [120, 11 * 13]]],
        "mean": [[[2, 3], [5, 6], [0, 0]], [[7, 9], [0, 0], [11, 12]]],
        "min": [[[1, 2], [5, 6], [0, 0]], [[7, 9], [0, 0], [10, 11]]],
        "arg_min": [[[0, 0], [1, 1], [3, 3]], [[1, 1], [3, 3], [0, 0]]],
        "max": [[[3, 4], [5, 6], [0, 0]], [[7, 9], [0, 0], [12, 13]]],
        "arg_max": [[[2, 2], [1, 1], [3, 3]], [[1, 1], [3, 3], [2, 2]]],
        "sum_grad": [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ],
        "add_grad": [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ],
        "mean_grad": [
            [[0.5000, 0.5000], [1.0000, 1.0000], [0.5000, 0.5000]],
            [[0.5000, 0.5000], [1.0000, 1.0000], [0.5000, 0.5000]],
        ],
        "min_grad": [
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        ],
        "max_grad": [
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        ],
        "mul_grad": [
            [[3.0, 4.0], [1.0, 1.0], [1.0, 2.0]],
            [[12.0, 13.0], [1.0, 1.0], [10.0, 11.0]],
        ],
    },
    {
        "src": [[1, 3], [2, 4]],
        "index": [[0, 0], [0, 0]],
        "dim": 1,
        "sum": [[4], [6]],
        "add": [[4], [6]],
        "mul": [[3], [8]],
        "mean": [[2], [3]],
        "min": [[1], [2]],
        "arg_min": [[0], [0]],
        "max": [[3], [4]],
        "arg_max": [[1], [1]],
        "sum_grad": [[1.0, 1.0], [1.0, 1.0]],
        "add_grad": [[1.0, 1.0], [1.0, 1.0]],
        "mean_grad": [[0.5000, 0.5000], [0.5000, 0.5000]],
        "min_grad": [[1.0, 0.0], [1.0, 0.0]],
        "max_grad": [[0.0, 1.0], [0.0, 1.0]],
        "mul_grad": [[3.0, 1.0], [4.0, 2.0]],
    },
    {
        "src": [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
        "index": [[0, 0], [0, 0]],
        "dim": 1,
        "sum": [[[4, 4]], [[6, 6]]],
        "add": [[[4, 4]], [[6, 6]]],
        "mul": [[[3, 3]], [[8, 8]]],
        "mean": [[[2, 2]], [[3, 3]]],
        "min": [[[1, 1]], [[2, 2]]],
        "arg_min": [[[0, 0]], [[0, 0]]],
        "max": [[[3, 3]], [[4, 4]]],
        "arg_max": [[[1, 1]], [[1, 1]]],
        "sum_grad": [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        "add_grad": [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        "mean_grad": [
            [[0.5000, 0.5000], [0.5000, 0.5000]],
            [[0.5000, 0.5000], [0.5000, 0.5000]],
        ],
        "min_grad": [[[1.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]],
        "max_grad": [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]],
        "mul_grad": [[[3.0, 3.0], [1.0, 1.0]], [[4.0, 4.0], [2.0, 2.0]]],
    },
]


@pytest.mark.parametrize(
    "test,reduce,dtype,ind_dtype,place",
    product(tests, reductions, dtypes, ind_dtypes, places),
)
def test_forward(test, reduce, dtype, ind_dtype, place):
    paddle.set_device(place)
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    fn = getattr(paddle_scatter, "scatter_" + reduce)
    out = fn(src, index, dim)
    if reduce == "min" or reduce == "max":
        out, arg_out = out
        arg_expected = tensor(test["arg_" + reduce], ind_dtype)
        print(arg_out)
        print(arg_expected)
        assert paddle.all(arg_out == arg_expected)
    assert paddle.all(out == expected)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize(
    "test,reduce,dtype,ind_dtype", product(tests, reductions, dtypes_half, ind_dtypes)
)
def test_forward_half(test, reduce, dtype, ind_dtype):
    paddle.set_device("gpu")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    fn = getattr(paddle_scatter, "scatter_" + reduce)
    out = fn(src, index, dim)
    if reduce == "min" or reduce == "max":
        out, arg_out = out
        arg_expected = tensor(test["arg_" + reduce], ind_dtype)
        print(arg_out)
        print(arg_expected)
        assert paddle.all(arg_out == arg_expected)
    assert paddle.all(out == expected)


@pytest.mark.parametrize("test,reduce,place", product(tests, reductions, places))
def test_backward(test, reduce, place):
    paddle.set_device(place)
    index = tensor(test["index"], paddle.int64)
    dim = test["dim"]
    exp_grad = tensor(test[f"{reduce}_grad"], paddle.float64)

    src = tensor(test["src"], paddle.float64)
    src.stop_gradient = False
    out = paddle_scatter.scatter(src, index, dim, None, None, reduce)
    out.backward()
    np.testing.assert_allclose(
        src.grad.numpy(), exp_grad.numpy(), rtol=1e-05, atol=1e-06
    )


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize("test,reduce", product(tests, reductions))
def test_backward_half(test, reduce):
    paddle.set_device("gpu")
    index = tensor(test["index"], paddle.int64)
    dim = test["dim"]
    exp_grad = tensor(test[f"{reduce}_grad"], paddle.float16)

    src = tensor(test["src"], paddle.float16)
    src.stop_gradient = False
    out = paddle_scatter.scatter(src, index, dim, None, None, reduce)
    out.backward()
    np.testing.assert_allclose(
        src.grad.numpy(), exp_grad.numpy(), rtol=1e-05, atol=1e-06
    )


@pytest.mark.parametrize(
    "test,reduce,dtype,ind_dtype,place",
    product(tests, reductions, dtypes, ind_dtypes, places),
)
def test_out(test, reduce, dtype, ind_dtype, place):
    paddle.set_device(place)
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    out = paddle.full_like(expected, -2)

    getattr(paddle_scatter, "scatter_" + reduce)(src, index, dim, out)

    if reduce == "sum" or reduce == "add":
        expected = expected - 2
    elif reduce == "mul":
        expected = out  # We can not really test this here.
    elif reduce == "mean":
        expected = out  # We can not really test this here.
    elif reduce == "min":
        expected = expected.fill_(-2)
    elif reduce == "max":
        expected[expected == 0] = -2
    else:
        raise ValueError

    np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-05, atol=1e-06)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize(
    "test,reduce,dtype,ind_dtype", product(tests, reductions, dtypes_half, ind_dtypes)
)
def test_out_half(test, reduce, dtype, ind_dtype):
    paddle.set_device("gpu")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    out = paddle.full_like(expected, -2)

    getattr(paddle_scatter, "scatter_" + reduce)(src, index, dim, out)

    if reduce == "sum" or reduce == "add":
        expected = expected - 2
    elif reduce == "mul":
        expected = out  # We can not really test this here.
    elif reduce == "mean":
        expected = out  # We can not really test this here.
    elif reduce == "min":
        print(expected)
        expected = expected.fill_(-2)
    elif reduce == "max":
        expected[expected == 0] = -2
    else:
        raise ValueError

    np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-05, atol=1e-06)


@pytest.mark.skipif(
    not paddle.core.is_compiled_with_cuda()
    or not paddle.core.is_bfloat16_supported(device)
    or not paddle.core.is_float16_supported(device),
    reason="half dtype not available",
)
@pytest.mark.parametrize(
    "test,reduce,dtype,ind_dtype", product(tests, reductions, dtypes_half, ind_dtypes)
)
def test_non_contiguous(test, reduce, dtype, ind_dtype):
    paddle.set_device("gpu")
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    if src.dim() > 1:
        shape = list(range(src.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        src = src.transpose(shape).contiguous().transpose(shape)
    if index.dim() > 1:
        shape = list(range(index.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        index = index.transpose(shape).contiguous().transpose(shape)

    out = getattr(paddle_scatter, "scatter_" + reduce)(src, index, dim)
    if reduce == "min" or reduce == "max":
        out, arg_out = out
        arg_expected = tensor(test["arg_" + reduce], ind_dtype)
        assert paddle.all(arg_out == arg_expected)
    assert paddle.all(out == expected)


@pytest.mark.parametrize(
    "test,reduce,dtype,ind_dtype,place",
    product(tests, reductions, dtypes, ind_dtypes, places),
)
def test_non_contiguous_half(test, reduce, dtype, ind_dtype, place):
    paddle.set_device(place)
    src = tensor(test["src"], dtype)
    index = tensor(test["index"], ind_dtype)
    dim = test["dim"]
    expected = tensor(test[reduce], dtype)

    if src.dim() > 1:
        shape = list(range(src.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        src = src.transpose(shape).contiguous().transpose(shape)
    if index.dim() > 1:
        shape = list(range(index.dim()))
        shape[0], shape[1] = shape[1], shape[0]
        index = index.transpose(shape).contiguous().transpose(shape)

    out = getattr(paddle_scatter, "scatter_" + reduce)(src, index, dim)
    if reduce == "min" or reduce == "max":
        out, arg_out = out
        arg_expected = tensor(test["arg_" + reduce], ind_dtype)
        assert paddle.all(arg_out == arg_expected)
    assert paddle.all(out == expected)
