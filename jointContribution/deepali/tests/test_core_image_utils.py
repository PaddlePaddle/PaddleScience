import os

import numpy as np
import paddle
import pytest
from deepali.core import Grid
from deepali.core import functional as U
from deepali.utils import paddle_aux  # noqa
from paddle import Tensor


def test_fill_border():
    image = paddle.zeros(shape=(2, 1, 7, 5), dtype="float32")

    result = U.fill_border(image, margin=1, value=1)
    assert isinstance(result, paddle.Tensor)
    assert tuple(result.shape) == tuple(image.shape)
    assert result[:, :, :1, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, :1].equal(y=1).astype("bool").all()
    assert result[:, :, 6:, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, 4:].equal(y=1).astype("bool").all()
    assert result[:, :, 1:6, 1:4].equal(y=0).astype("bool").all()
    assert result.sum() == 40

    result = U.fill_border(image, margin=(1, 2), value=1)
    assert isinstance(result, paddle.Tensor)
    assert tuple(result.shape) == tuple(image.shape)
    assert result[:, :, :2, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, :1].equal(y=1).astype("bool").all()
    assert result[:, :, 5:, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, 4:].equal(y=1).astype("bool").all()
    assert result[:, :, 2:5, 1:4].equal(y=0).astype("bool").all()
    assert result.sum() == 52

    image = paddle.zeros(shape=(2, 1, 7, 5, 11), dtype="float32")
    result = U.fill_border(image, margin=(3, 1, 2), value=1)
    assert isinstance(result, paddle.Tensor)
    assert tuple(result.shape) == tuple(image.shape)
    assert result[:, :, :2, :, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, :1, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, :, :3].equal(y=1).astype("bool").all()
    assert result[:, :, 5:, :, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, 4:, :].equal(y=1).astype("bool").all()
    assert result[:, :, :, :, 8:].equal(y=1).astype("bool").all()
    assert result[:, :, 2:5, 1:4, 3:8].equal(y=0).astype("bool").all()
    assert result.sum() == 680


def test_finite_differences() -> None:
    image = paddle.randn(shape=(3, 2, 8, 16, 24))

    # Forward difference scheme
    image.stop_gradient = not True
    deriv = U.finite_differences(image, "x", mode="forward")
    assert isinstance(deriv, paddle.Tensor)
    assert (not deriv.stop_gradient) is True
    deriv = deriv.detach()
    assert (not deriv.stop_gradient) is False
    assert deriv.is_floating_point()
    assert tuple(deriv.shape) == tuple(image.shape)

    expected = paddle.nn.functional.pad(
        x=image, pad=(0, 1, 0, 0, 0, 0), mode="replicate", pad_from_left_axis=False
    )
    expected = expected[..., 1:].sub(expected[..., :-1])
    assert paddle.allclose(x=deriv, y=expected).item()

    # Backward difference scheme
    image.stop_gradient = not False
    deriv = U.finite_differences(image, "y", mode="backward")
    assert isinstance(deriv, paddle.Tensor)
    assert (not deriv.stop_gradient) is False
    assert deriv.is_floating_point()
    assert tuple(deriv.shape) == tuple(image.shape)

    expected = paddle.nn.functional.pad(
        x=image, pad=(0, 0, 1, 0, 0, 0), mode="replicate", pad_from_left_axis=False
    )
    expected = expected[..., 1:, :].sub(expected[..., :-1, :])
    assert paddle.allclose(x=deriv, y=expected).item()

    # Central difference scheme
    image.stop_gradient = not False
    deriv = U.finite_differences(image, "z", mode="central")
    assert isinstance(deriv, paddle.Tensor)
    assert (not deriv.stop_gradient) is False
    assert deriv.is_floating_point()
    assert tuple(deriv.shape) == tuple(image.shape)

    expected = paddle.nn.functional.pad(
        x=image, pad=(0, 0, 0, 0, 1, 1), mode="replicate", pad_from_left_axis=False
    )
    expected = expected[:, :, 2:, :, :].sub(expected[:, :, :-2, :, :]).div(2)
    assert paddle.allclose(x=deriv, y=expected).item()

    # Forward-central-backward difference scheme
    deriv = U.finite_differences(image, "z", mode="forward_central_backward")
    assert isinstance(deriv, paddle.Tensor)
    assert (not deriv.stop_gradient) is False
    assert deriv.is_floating_point()
    assert tuple(deriv.shape) == tuple(image.shape)
    expected = paddle.concat(
        x=[
            image[:, :, 1:2, :, :].sub(image[:, :, :1, :, :]),
            image[:, :, 2:, :, :].sub(image[:, :, :-2, :, :]).div(2),
            image[:, :, -1:, :, :].sub(image[:, :, -2:-1, :, :]),
        ],
        axis=2,
    )
    assert paddle.allclose(x=deriv, y=expected).item()

    # Zero-th order
    assert U.finite_differences(image, "x", mode="central", order=0) is image


@pytest.mark.skipif(not paddle.device.cuda.device_count() >= 1, reason="CUDA not available")
def test_rand_sample_cuda() -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = paddle.CUDAPlace(int("cuda:0".replace("cuda:", "")))
    generator = paddle.framework.core.default_cpu_generator().manual_seed(123456789)

    start = paddle.device.cuda.Event(enable_timing=True)
    end = paddle.device.cuda.Event(enable_timing=True)

    shape = tuple((5, 1, 32, 64, 128))
    data = (
        paddle.arange(end=np.prod(shape[2:]))
        .reshape((1, 1) + shape[2:])
        .expand(shape=shape)
        .to(device)
    )
    num_samples = 10

    # Draw unweighted samples with and without replacement
    t_elapsed_1 = 0
    for i in range(5):
        start.record()
        values = U.rand_sample(data, num_samples, mask=None, replacement=False, generator=generator)
        end.record()
        paddle.device.synchronize()
        if i > 0:
            t_elapsed_1 += start.elapsed_time(end)
    t_elapsed_1 /= 4

    assert not paddle_aux.allclose_int(x=values[0], y=values[1]).item()

    t_elapsed_2 = 0
    for i in range(5):
        start.record()
        values = U.rand_sample(data, num_samples, mask=None, replacement=True, generator=generator)
        end.record()
        paddle.device.synchronize()
        if i > 0:
            t_elapsed_2 += start.elapsed_time(end)
    t_elapsed_2 /= 4

    assert not paddle_aux.allclose_int(x=values[0], y=values[1]).item()

    # Compare with using multinomial with an all-one mask
    t_elapsed_3 = 0
    for i in range(5):
        start.record()
        mask = paddle.ones(shape=(1, 1) + tuple(data.shape)[2:])
        values = U.rand_sample(data, num_samples, mask=mask, replacement=False, generator=generator)
        end.record()
        paddle.device.synchronize()
        if i > 0:
            t_elapsed_3 += start.elapsed_time(end)
    t_elapsed_3 /= 4

    assert not paddle_aux.allclose_int(x=values[0], y=values[1]).item()

    t_elapsed_4 = 0
    for i in range(5):
        start.record()
        mask = paddle.ones(shape=(1, 1) + tuple(data.shape)[2:])
        values = U.rand_sample(data, num_samples, mask=mask, replacement=True, generator=generator)
        end.record()
        paddle.device.synchronize()
        if i > 0:
            t_elapsed_4 += start.elapsed_time(end)
    t_elapsed_4 /= 4

    assert not paddle_aux.allclose_int(x=values[0], y=values[1]).item()


def test_sample_image() -> None:
    shape = tuple((5, 2, 32, 64, 63))
    image: Tensor = paddle.arange(end=np.prod(shape))
    image = image.reshape(shape)
    grid = Grid(shape=shape[2:])

    indices = paddle.arange(start=0, end=np.prod(grid.size()), step=10)
    voxels = U.unravel_coords(indices.unsqueeze(axis=0), tuple(grid.size()))
    coords = grid.index_to_cube(voxels)
    assert coords.dtype == grid.dtype
    assert coords.is_floating_point()
    assert tuple(coords.shape) == (1, len(indices), 3)
    assert coords.min().greater_equal(y=paddle.to_tensor(-1))
    assert coords.max().less_equal(y=paddle.to_tensor(1))

    result = U.sample_image(image, coords, mode="nearest")
    expected = image.flatten(start_axis=2).index_select(axis=2, index=indices)
    assert result.equal(y=expected).astype("bool").all()
