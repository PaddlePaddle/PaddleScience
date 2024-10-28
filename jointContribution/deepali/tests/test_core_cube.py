import numpy as np
import paddle
import pytest
from deepali.core import Cube
from deepali.core import Grid
from deepali.core import affine as A
from deepali.utils import paddle_aux  # noqa


@pytest.fixture
def default_angle() -> paddle.Tensor:
    return paddle.deg2rad(x=paddle.to_tensor(data=33.0))


@pytest.fixture
def default_cube(default_angle: paddle.Tensor) -> Cube:
    direction = A.euler_rotation_matrix(default_angle)
    cube = Cube(extent=(34, 42), center=(7, 4), direction=direction)
    return cube


def test_cube_is_grid_with_three_points(default_cube: Cube) -> None:
    cube = default_cube
    grid = cube.grid(size=3, align_corners=True)
    assert type(grid) is Grid
    assert grid.ndim == cube.ndim
    assert paddle_aux.is_eq_place(grid.device, cube.device)
    assert tuple(grid.shape) == (3,) * grid.ndim
    assert grid.align_corners() is True
    assert paddle.allclose(x=grid.cube_extent(), y=cube.extent()).item()
    assert paddle.allclose(x=grid.affine(), y=cube.affine()).item()
    assert paddle.allclose(x=grid.inverse_affine(), y=cube.inverse_affine()).item()
    assert paddle.allclose(x=grid.spacing(), y=cube.spacing()).item()
    assert paddle.allclose(
        x=grid.transform("cube_corners", "world"), y=cube.transform("cube", "world")
    ).item()
    assert paddle.allclose(
        x=grid.transform("world", "cube_corners"), y=cube.transform("world", "cube")
    ).item()
    assert paddle.allclose(x=grid.transform(), y=cube.transform()).item()
    assert paddle.allclose(x=grid.inverse_transform(), y=cube.inverse_transform()).item()


def test_cube_to_from_numpy(default_cube: Cube) -> None:
    r"""Test converting a Cube to a 1-dimensional NumPy array and constructing a new one from such array."""
    dim = default_cube.ndim
    arr = default_cube.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert str(arr.dtype) == ("float32" if default_cube.dtype == paddle.float32 else "float64")
    assert tuple(arr.shape) == ((dim + 2) * dim,)
    cube = Cube.from_numpy(arr)
    assert cube == default_cube


def test_cube_eq(default_cube: Cube, default_angle: paddle.Tensor) -> None:
    r"""Test comparison of different Cube instances for equality."""
    # Same instance
    assert default_cube == default_cube

    # Different instance, same atributes
    other_cube = Cube(
        extent=default_cube.extent(),
        center=default_cube.center(),
        direction=default_cube.direction(),
    )
    assert default_cube == other_cube

    # Different extent
    other_cube = Cube(
        extent=default_cube.extent().add(1),
        center=default_cube.center(),
        direction=default_cube.direction(),
    )
    assert default_cube != other_cube

    # Different center
    other_cube = Cube(
        extent=default_cube.extent(),
        center=default_cube.center().add(0.001),
        direction=default_cube.direction(),
    )
    assert default_cube != other_cube

    # Different direction
    other_direction = A.euler_rotation_matrix(default_angle.add(0.001))
    other_cube = Cube(
        extent=default_cube.extent(), center=default_cube.center(), direction=other_direction
    )
    assert default_cube != other_cube
