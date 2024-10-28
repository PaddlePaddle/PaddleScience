import numpy as np
import paddle
from deepali.core import tensor as U


def test_move_dim():
    # Same dimension
    for dim in range(5):
        shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), dim, dim).shape
        assert tuple(shape) == (1, 5, 7, 9, 3)
    # Move channels to last dimension
    shape = U.move_dim(paddle.empty(shape=[1, 3, 5, 7, 9]), 1, -1).shape
    assert tuple(shape) == (1, 5, 7, 9, 3)
    shape = U.move_dim(paddle.empty(shape=[1, 3, 5, 7, 9]), 1, 4).shape
    assert tuple(shape) == (1, 5, 7, 9, 3)
    # Move last dimension to channels dimension
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), -1, 1).shape
    assert tuple(shape) == (1, 3, 5, 7, 9)
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 4, 1).shape
    assert tuple(shape) == (1, 3, 5, 7, 9)
    # Move arbitrary dimensions
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 0, 1).shape
    assert tuple(shape) == (5, 1, 7, 9, 3)
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 1, 0).shape
    assert tuple(shape) == (5, 1, 7, 9, 3)
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 0, 2).shape
    assert tuple(shape) == (5, 7, 1, 9, 3)
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 2, 0).shape
    assert tuple(shape) == (7, 1, 5, 9, 3)
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 2, 4).shape
    assert tuple(shape) == (1, 5, 9, 3, 7)
    shape = U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 4, 2).shape
    assert tuple(shape) == (1, 5, 3, 7, 9)
    # Reversibility
    shape = U.move_dim(U.move_dim(paddle.empty(shape=[1, 5, 7, 9, 3]), 0, 2), 2, 0).shape
    assert tuple(shape) == (1, 5, 7, 9, 3)


def test_unravel_coords():
    indices = paddle.to_tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype="int64")
    size = 3, 3
    shape = tuple(reversed(size))
    coords = U.unravel_coords(indices, size)
    assert isinstance(coords, paddle.Tensor)
    assert coords.dtype == paddle.int64

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    expected = np.flip(expected, axis=-1)
    assert np.all(coords.numpy() == expected)

    indices = paddle.to_tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int32")
    size = 2, 5
    shape = tuple(reversed(size))
    coords = U.unravel_coords(indices, size)
    assert isinstance(coords, paddle.Tensor)
    assert coords.dtype == paddle.int32

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    expected = np.flip(expected, axis=-1)
    assert np.all(coords.numpy() == expected)


def test_unravel_index():
    indices = paddle.to_tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype="int64")
    shape = 3, 3
    result = U.unravel_index(indices, shape)
    assert isinstance(result, paddle.Tensor)
    assert result.dtype == paddle.int64

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    assert np.all(result.numpy() == expected)

    indices = paddle.to_tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int32")
    shape = 2, 5
    result = U.unravel_index(indices, shape)
    assert isinstance(result, paddle.Tensor)
    assert result.dtype == paddle.int32

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    assert np.all(result.numpy() == expected)
