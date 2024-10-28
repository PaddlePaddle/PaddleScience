import deepali.utils.paddle_aux  # noqa
import paddle
import pytest
from deepali.core.kernels import gaussian1d
from deepali.core.random import _multinomial


@pytest.fixture
def generator():
    return paddle.framework.core.default_cpu_generator().manual_seed(123456789)


def test_multinomial_with_replacement(generator) -> None:
    r"""Test weighted sampling with replacement using inverse transform sampling."""

    input = paddle.arange(end=1000)

    # Input checks
    with pytest.raises(ValueError):
        _multinomial(paddle.to_tensor(data=0), 1, replacement=True)
    with pytest.raises(ValueError):
        _multinomial(paddle.ones(shape=(1, 2, 3)), 1, replacement=True)

    index = _multinomial(paddle.ones(shape=[10]), 11, replacement=True)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (11,)

    # Output type and shape
    index = _multinomial(input, 10, replacement=True)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (10,)

    index = _multinomial(input.unsqueeze(axis=0), 10, replacement=True)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (1, 10)

    index = _multinomial(input.unsqueeze(axis=0).tile(repeat_times=[3, 1]), 10, replacement=True)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (3, 10)

    # Only samples with non-zero weight
    subset = input.clone()
    subset[subset.less_than(y=paddle.to_tensor(100))] = 0
    subset[subset.greater_than(y=paddle.to_tensor(200))] = 0
    for _ in range(10):
        index = _multinomial(subset, 10, replacement=True, generator=generator)
        assert isinstance(index, paddle.Tensor)
        assert index.dtype == paddle.int64
        assert tuple(index.shape) == (10,)
        assert (
            subset[index]
            .greater_equal(y=paddle.to_tensor(100))
            .less_equal(y=paddle.to_tensor(200))
            .astype("bool")
            .all()
        )
    # Sample from a spatial Gaussian distribution
    num_samples = 1000000
    input = gaussian1d(5)
    index = _multinomial(input, num_samples, replacement=True, generator=generator)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (num_samples,)
    freq = index.bincount().div(num_samples)
    assert freq.allclose(y=input, rtol=0.1).item()


def test_multinomial_without_replacement(generator=None) -> None:
    r"""Test weighted sampling without replacement using Gumbel-max trick."""
    input = paddle.arange(end=1000)

    # Input checks
    with pytest.raises(ValueError):
        _multinomial(paddle.to_tensor(data=0), 1, replacement=False)
    with pytest.raises(ValueError):
        _multinomial(paddle.ones(shape=(1, 2, 3)), 1, replacement=False)

    index = _multinomial(paddle.ones(shape=[10]), 10, replacement=False)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (10,)
    assert index.sort().equal(paddle.arange(end=10)).astype("bool").all()

    with pytest.raises(ValueError):
        _multinomial(paddle.ones(shape=[10]), 11, replacement=False)

    # Output type and shape
    index = _multinomial(input, 10, replacement=False)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (10,)

    index = _multinomial(input.unsqueeze(axis=0), 10, replacement=False)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (1, 10)

    index = _multinomial(input.unsqueeze(axis=0).tile(repeat_times=[3, 1]), 10, replacement=False)
    assert isinstance(index, paddle.Tensor)
    assert index.dtype == paddle.int64
    assert tuple(index.shape) == (3, 10)

    # No duplicates
    subset = input.clone()
    subset[subset.less_than(y=paddle.to_tensor(100))] = 0
    subset[subset.greater_than(y=paddle.to_tensor(200))] = 0
    subset[subset.greater_than(y=paddle.to_tensor(0))] = 1

    num_samples = 10
    num_repeat = 100

    for _ in range(num_repeat):
        index = _multinomial(subset, num_samples, replacement=False, generator=generator)
        assert isinstance(index, paddle.Tensor)
        assert index.dtype == paddle.int64
        assert tuple(index.shape) == (num_samples,)
        assert tuple(index.unique().shape) == (num_samples,)
        assert subset[index].equal(y=1).astype("bool").all()
