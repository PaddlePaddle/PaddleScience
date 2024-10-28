import paddle
import pytest
from deepali.core import bspline as B


def test_cubic_bspline_interpolation_weights() -> None:
    kernel = B.cubic_bspline_interpolation_weights(5)
    assert isinstance(kernel, paddle.Tensor)
    assert tuple(kernel.shape) == (5, 4)

    assert paddle.allclose(
        x=kernel, y=B.cubic_bspline_interpolation_weights(5, derivative=0)
    ).item()
    kernels = B.cubic_bspline_interpolation_weights(5, derivative=[0, 1])
    assert isinstance(kernels, tuple)
    assert len(kernels) == 2
    assert paddle.allclose(
        x=kernels[0], y=B.cubic_bspline_interpolation_weights(5, derivative=0)
    ).item()
    assert paddle.allclose(
        x=kernels[1], y=B.cubic_bspline_interpolation_weights(5, derivative=1)
    ).item()
    with pytest.raises(ValueError):
        B.cubic_bspline_interpolation_weights([5], derivative=[0, 1])
    with pytest.raises(ValueError):
        B.cubic_bspline_interpolation_weights([5, 5], derivative=[0])


def test_cubic_bspline_interpolation_at_control_points() -> None:
    r"""Test evaluation of cubic B-spline at control point grid locations only."""
    kernel_all = B.cubic_bspline_interpolation_weights(5)
    kernel_cps = B.cubic_bspline_interpolation_weights(1)
    assert tuple(kernel_all.shape) == (5, 4)
    assert tuple(kernel_cps.shape) == (1, 4)
    assert paddle.allclose(x=kernel_cps[0], y=kernel_all[0]).item()

    # Single non-zero control point coefficient
    data = paddle.zeros(shape=(1, 1, 4))
    data[0, 0, 1] = 1

    values_all = B.evaluate_cubic_bspline(data, stride=5, kernel=kernel_all)
    values_cps = B.evaluate_cubic_bspline(data, stride=1, kernel=kernel_cps)
    assert paddle.allclose(x=B.evaluate_cubic_bspline(data, stride=1), y=values_cps).item()

    assert tuple(values_all.shape) == (1, 1, 5)
    assert tuple(values_cps.shape) == (1, 1, 1)

    assert paddle.allclose(x=values_cps, y=values_all[:, :, ::5]).item()
    assert paddle.allclose(x=values_cps[0, 0][0], y=kernel_cps[0, 1]).item()

    # Random uniformly distributed control point coefficients
    data = paddle.rand(shape=(1, 1, 35))

    values_all = B.evaluate_cubic_bspline(data, stride=5, kernel=kernel_all)
    values_cps = B.evaluate_cubic_bspline(data, stride=1, kernel=kernel_cps)
    assert paddle.allclose(x=B.evaluate_cubic_bspline(data, stride=1), y=values_cps).item()

    assert tuple(values_all.shape) == (1, 1, 32 * 5)
    assert tuple(values_cps.shape) == (1, 1, 32)
    assert paddle.allclose(x=values_cps, y=values_all[:, :, ::5]).item()
