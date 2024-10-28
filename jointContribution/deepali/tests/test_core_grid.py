import deepali.utils.paddle_aux  # noqa
import numpy as np
import paddle
import pytest
from deepali.core import affine as A
from deepali.core.grid import Grid


@pytest.fixture
def default_angle() -> paddle.Tensor:
    return paddle.deg2rad(x=paddle.to_tensor(data=33.0))


@pytest.fixture
def default_grid(default_angle: paddle.Tensor) -> Grid:
    direction = A.euler_rotation_matrix(default_angle)
    grid = Grid(size=(34, 42), spacing=(0.7, 1.2), center=(7, 4), direction=direction)
    return grid


def test_grid_to_from_numpy(default_grid: Grid) -> None:
    r"""Test converting a Grid to a 1-dimensional NumPy array and constructing a new one from such array."""
    dim = default_grid.ndim
    arr = default_grid.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == (np.float32 if default_grid.dtype == paddle.float32 else np.float64)
    assert arr.shape == ((dim + 3) * dim,)
    grid = Grid.from_numpy(arr)
    assert grid == default_grid


def test_grid_eq(default_grid: Grid, default_angle: paddle.Tensor) -> None:
    r"""Test comparison of different Grid instances for equality."""
    # Same instance
    assert default_grid == default_grid

    # Different instance, same attributes
    other_grid = Grid(
        size=default_grid.size(),
        spacing=default_grid.spacing(),
        center=default_grid.center(),
        direction=default_grid.direction(),
    )
    assert default_grid == other_grid

    # Different size
    other_grid = Grid(
        size=default_grid.size_tensor().add(1),
        spacing=default_grid.spacing(),
        center=default_grid.center(),
        direction=default_grid.direction(),
    )
    assert default_grid != other_grid

    # Different spacing
    other_grid = Grid(
        size=default_grid.size(),
        spacing=default_grid.spacing().add(0.001),
        center=default_grid.center(),
        direction=default_grid.direction(),
    )
    assert default_grid != other_grid

    # Different center
    other_grid = Grid(
        size=default_grid.size(),
        spacing=default_grid.spacing(),
        center=default_grid.center().add(0.001),
        direction=default_grid.direction(),
    )
    assert default_grid != other_grid

    # Different direction
    other_direction = A.euler_rotation_matrix(default_angle.add(0.001))

    other_grid = Grid(
        size=default_grid.size(),
        spacing=default_grid.spacing(),
        center=default_grid.center(),
        direction=other_direction,
    )
    assert default_grid != other_grid


def test_grid_crop():
    grid = Grid((10, 7, 5))

    new_grid = grid.crop(2, -1)
    assert isinstance(new_grid, Grid)
    assert new_grid is not grid
    assert new_grid.size() == (6, 9, 5)
    assert paddle.allclose(x=new_grid.center(), y=grid.center()).item()
    assert paddle.allclose(x=new_grid.index_to_world((-2, 1, 0)), y=grid.origin()).item()
    new_grid_2 = grid.crop((2, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.crop(margin=(2, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.crop(num=(2, 2, -1, -1))
    assert new_grid_2 == new_grid


def test_grid_pad():
    grid = Grid((10, 7, 5))

    new_grid = grid.pad(4, -1)
    assert isinstance(new_grid, Grid)
    assert new_grid is not grid
    assert new_grid.size() == (18, 5, 5)
    assert paddle.allclose(x=new_grid.center(), y=grid.center()).item()
    assert paddle.allclose(x=new_grid.index_to_world((4, -1, 0)), y=grid.origin()).item()
    new_grid_2 = grid.pad((4, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.pad(margin=(4, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.pad(num=(4, 4, -1, -1))
    assert new_grid_2 == new_grid
