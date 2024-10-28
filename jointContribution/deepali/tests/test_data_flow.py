from copy import deepcopy
from typing import Tuple

import deepali.utils.paddle_aux  # noqa
import numpy as np
import paddle
import pytest
from deepali.core import Axes
from deepali.core import Grid
from deepali.data import FlowField
from deepali.data import FlowFields
from deepali.data import Image
from deepali.data import ImageBatch
from paddle import Tensor


def image_size(sdim: int) -> Tuple[int, ...]:
    if sdim == 2:
        return 64, 57
    if sdim == 3:
        return 64, 57, 31
    raise ValueError("image_size() 'sdim' must be 2 or 3")


def image_shape(sdim: int) -> Tuple[int, ...]:
    return tuple(reversed(image_size(sdim)))


@pytest.fixture(scope="function")
def grid(request) -> paddle.Tensor:
    size = image_size(request.param)
    spacing = (0.25, 0.2, 0.5)[: len(size)]
    return Grid(size=size, spacing=spacing)


@pytest.fixture(scope="function")
def data(request) -> paddle.Tensor:
    shape = image_shape(request.param)
    return paddle.randn(shape=(len(shape),) + shape).multiply_(y=paddle.to_tensor(100))


@pytest.fixture(scope="function")
def zeros(request) -> paddle.Tensor:
    shape = image_shape(request.param)
    return paddle.zeros(shape=(len(shape),) + shape)


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def _test_flowfield_torch_function(zeros: paddle.Tensor, grid: Grid) -> None:
    data = zeros
    axes = Axes.WORLD  # different from default to check if attribute is preserved

    image = FlowField(data, grid, axes)
    assert type(image) is FlowField
    assert hasattr(image, "_axes")
    assert image.axes() is axes
    assert hasattr(image, "_grid")
    assert image.grid() is grid

    result = image.astype(image.dtype)
    assert result is image

    result = image.astype("int16")
    assert result is not image
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is axes
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype == "int16"
    assert result.data_ptr() != image.data_ptr()

    result = image.equal(y=0)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is axes
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype == "bool"
    assert result.data_ptr() != image.data_ptr()

    result = result.astype("bool").all()
    assert type(result) is Tensor
    assert not hasattr(result, "_axes")
    assert not hasattr(result, "_grid")
    assert result.ndim == 0
    assert result.item() is True

    result = paddle.add(x=image, y=paddle.to_tensor(2))
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.equal(y=2).astype("bool").all()

    result = paddle.add(x=4, y=paddle.to_tensor(image))
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.equal(y=4).astype("bool").all()

    result = image.add(1)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.equal(y=1).astype("bool").all()

    result = image.clone()
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is not image.grid()
    assert result.grid() == image.grid()
    assert result.data_ptr() != image.data_ptr()

    result = image.to("cpu", "int16")
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.place == paddle.CPUPlace()
    assert result.dtype == "int16"

    if paddle.device.cuda.device_count() >= 1:
        result = image.cuda(blocking=True)
        assert type(result) is FlowField
        assert hasattr(result, "_axes")
        assert result.axes() is image.axes()
        assert hasattr(result, "_grid")
        assert result.grid() is image.grid()
        assert result.place.is_gpu_place()

    result = paddle.concat(x=[image, image], axis=0)
    assert isinstance(result, Image)
    assert type(result) == Image
    assert not hasattr(result, "_axes")
    assert hasattr(result, "_grid")
    assert tuple(result.shape)[0] == tuple(image.shape)[0] * 2
    assert tuple(result.shape)[1:] == tuple(image.shape)[1:]
    assert result.grid() is image.grid()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def _test_flowfields_torch_function(zeros: paddle.Tensor, grid: Grid) -> None:
    data = zeros

    batch = FlowFields(data.unsqueeze(axis=0), grid, Axes.WORLD)

    result = batch.detach()
    assert isinstance(result, FlowFields)
    assert tuple(result.shape) == tuple(batch.shape)
    assert result.axes() == batch.axes()
    assert result.grids() == batch.grids()

    result = paddle.concat(x=[batch, batch], axis=0)
    assert isinstance(result, FlowFields)
    assert tuple(result.shape)[0] == tuple(batch.shape)[0] + tuple(batch.shape)[0]
    assert tuple(result.shape)[1:] == tuple(batch.shape)[1:]
    assert result.axes() == batch.axes()
    assert result.grids() == batch.grids() * 2

    result = paddle.concat(x=[batch, batch], axis=1)
    assert type(result) == ImageBatch
    assert tuple(result.shape)[0] == tuple(batch.shape)[0]
    assert tuple(result.shape)[1] == tuple(batch.shape)[1] * 2
    assert tuple(result.shape)[2:] == tuple(batch.shape)[2:]
    assert result.grids() == batch.grids()

    with pytest.raises(ValueError):
        # Cannot batch together flow fields with mismatching Axes
        paddle.concat(x=[batch, batch.axes(Axes.CUBE_CORNERS)], axis=0)


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_flowfields_getitem(zeros: paddle.Tensor, grid: Grid) -> None:
    grids = [deepcopy(grid) for _ in range(5)]  # make grids distinguishable
    batch = FlowFields(
        zeros.unsqueeze(axis=0).expand(shape=(5,) + tuple(zeros.shape)), grids, axes=Axes.GRID
    )
    for i in range(len(batch)):
        batch.tensor()[i] = i

    for i in range(len(batch)):
        item = batch[i]
        assert type(item) is FlowField
        assert item.axes() is batch.axes()
        assert item.grid() is batch.grid(i)
        assert paddle.allclose(x=item, y=batch.tensor()[i]).item()

    item = batch[1:-1:2]
    assert type(item) is FlowFields
    assert item.axes() is batch.axes()
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[1:-1:2]).item()

    item = batch[...]
    assert type(item) is FlowFields
    assert item.axes() is batch.axes()
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[...]).item()

    item = batch[:, ...]
    assert type(item) is FlowFields
    assert item.axes() is batch.axes()
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[:, ...]).item()

    item = batch[[4]]
    assert type(item) is FlowFields
    assert len(item) == 1
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(4)
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[[4]]).item()

    index = [3, 2]
    item = batch[index]
    assert type(item) is FlowFields
    assert len(item) == 2
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(3)
    assert item.grid(1) is batch.grid(2)
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[index]).item()

    index = 0, 2
    item = batch[index]
    assert type(item) is Tensor
    assert paddle.allclose(x=item, y=batch.tensor()[index]).item()

    index = np.array([1, 3])
    item = batch[index]
    assert type(item) is FlowFields
    assert len(item) == 2
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(1)
    assert item.grid(1) is batch.grid(3)
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[index]).item()

    index = paddle.to_tensor(data=[0, 2])
    item = batch[index]
    assert type(item) is FlowFields
    assert len(item) == 2
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(0)
    assert item.grid(1) is batch.grid(2)
    assert paddle.allclose(x=item.tensor(), y=batch.tensor()[index]).item()

    item = batch[:, 0]
    assert type(item) is Tensor
    assert paddle.allclose(x=item, y=batch.tensor()[:, 0]).item()

    item = batch[:, 1:2]
    assert type(item) is ImageBatch
    assert all(a is b for a, b in zip(item.grids(), batch.grids()))
    assert paddle.allclose(x=item, y=batch.tensor()[:, 1:2]).item()

    if batch.ndim == 5:
        item = batch[:, :2, 0]
        assert type(item) is Tensor
        assert tuple(item.shape)[1] == item.ndim - 2
        assert paddle.allclose(x=item, y=batch.tensor()[:, :2, 0]).item()

    item = batch[3, 1:2]
    assert type(item) is Image
    assert item.grid() is batch.grid(3)
    assert paddle.allclose(x=item, y=batch.tensor()[3, 1:2]).item()
