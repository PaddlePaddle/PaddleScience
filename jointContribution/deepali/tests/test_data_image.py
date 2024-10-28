from copy import deepcopy
from typing import Tuple

import numpy as np
import paddle
import pytest
from deepali.core import ALIGN_CORNERS
from deepali.core import Grid
from deepali.core import functional as U
from deepali.data import Image
from deepali.data import ImageBatch
from deepali.utils import paddle_aux
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
    return paddle.randn(shape=(1,) + shape).multiply_(y=paddle.to_tensor(100.0))


@pytest.fixture(scope="function")
def zeros(request) -> paddle.Tensor:
    shape = image_shape(request.param)
    return paddle.zeros(shape=(1,) + shape)


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_init(zeros: paddle.Tensor, grid: Grid) -> None:
    data = zeros
    data.to(device=paddle.CPUPlace())  # convert from torch, in torch CPUPlace is default.

    image = Image(data)
    assert type(image) is Image
    assert hasattr(image, "_grid")
    assert isinstance(image._grid, Grid)
    assert image._grid.size() == grid.size()

    assert image.data_ptr() == data.data_ptr()
    tensor = image.tensor()
    assert type(tensor) is Tensor
    assert image.data_ptr() == tensor.data_ptr()

    image = Image(data, grid, device=data.place)
    assert type(image) is Image
    assert hasattr(image, "_grid")
    assert image._grid is grid
    assert image.data_ptr() == data.data_ptr()

    image = Image(data, grid, requires_grad=True)
    assert type(image) is Image
    assert image._grid is grid
    assert not image.stop_gradient
    assert "pinned" not in str(image.place)
    assert paddle_aux.is_eq_place(image.place, data.place)
    assert image.dtype == data.dtype
    assert image.data_ptr() == data.data_ptr()

    if paddle.device.cuda.device_count() >= 1:
        device = paddle.CUDAPlace(int("cuda:0".replace("cuda:", "")))
        image = Image(data, grid, device=device)
        assert type(image) is Image
        assert image._grid is grid
        assert image.data_ptr() != data.data_ptr()
        assert image.place.is_gpu_place()
        assert paddle_aux.is_eq_place(image.place, device)
        assert image.dtype == data.dtype

        image = Image(data, grid, pin_memory=True)
        assert type(image) is Image
        assert image._grid is grid
        assert not image.stop_gradient  # in paddle default is False
        assert "pinned" in str(image.place)
        # assert paddle_aux.is_eq_place(image.place, data.place)
        assert image.dtype == data.dtype
        assert image.data_ptr() != data.data_ptr()

        image = Image(data, grid, requires_grad=True, pin_memory=True)
        assert type(image) is Image
        assert image._grid is grid
        assert not image.stop_gradient
        assert "pinned" in str(image.place)
        # assert paddle_aux.is_eq_place(image.place, data.place)
        assert image.dtype == data.dtype
        assert image.data_ptr() != data.data_ptr()

    out_0 = paddle.zeros(shape=(1, 32, 64, 64))
    out_0.stop_gradient = not True
    params = out_0
    assert not params.stop_gradient

    image = Image(params)
    assert type(image) is Image
    assert not image.stop_gradient
    assert image.data_ptr() == params.data_ptr()

    image = Image(params, requires_grad=False)
    assert type(image) is Image
    assert not not image.stop_gradient
    assert image.data_ptr() == params.data_ptr()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_image_deepcopy(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    other = deepcopy(image)
    assert type(other) is Image
    assert hasattr(other, "_grid")
    assert other.grid() is not image.grid()
    assert other.grid() == image.grid()
    # assert other.data_ptr() != image.data_ptr()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def _test_image_torch_function(zeros: paddle.Tensor, grid: Grid) -> None:
    data = zeros

    image = Image(data, grid)
    assert type(image) is Image
    assert hasattr(image, "_grid")
    assert image.grid() is grid

    result = image.astype(image.dtype)
    assert result is image

    result = image.astype("int16")
    assert result is not image
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype == "int16"
    assert result.data_ptr() != image.data_ptr()

    result = image.equal(y=0)
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype == "bool"
    assert result.data_ptr() != image.data_ptr()

    result = result.astype("bool").all()
    assert type(result) is Tensor
    assert not hasattr(result, "_grid")
    assert result.ndim == 0
    assert result.item() is True

    result = paddle.add(x=image, y=paddle.to_tensor(2))
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.equal(y=2).astype("bool").all()

    result = paddle.add(x=4, y=paddle.to_tensor(image))
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.equal(y=4).astype("bool").all()

    result = image.add(1)
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.equal(y=1).astype("bool").all()

    result = image.clone()
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is not image.grid()
    assert result.grid() == image.grid()
    assert result.data_ptr() != image.data_ptr()

    result = image.to("cpu", "int16")
    assert type(result) is Image
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.place == paddle.CPUPlace()
    assert result.dtype == "int16"

    if paddle.device.cuda.device_count() >= 1:
        result = image.cuda(blocking=True)
        assert type(result) is Image
        assert hasattr(result, "_grid")
        assert result.grid() is image.grid()
        assert result.place.is_gpu_place()

    image_1 = Image(data, grid)
    image_2 = Image(data.clone(), grid)
    assert tuple(image_1.shape)[1:] == tuple(image_2.shape)[1:]
    image_3 = paddle.concat(x=[image_1, image_2], axis=0)
    assert isinstance(image_3, Image)
    assert tuple(image_3.shape)[0] == tuple(image_1.shape)[0] + tuple(image_2.shape)[0]
    assert tuple(image_3.shape)[1:] == tuple(image_1.shape)[1:]

    # F.grid_sample() should always return a Tensor because the grid of the resampled image
    # may only by chance have the same size as the one on which the data is being resampled,
    # but the spatial positions of the new grid locations differ. See also Image.sample(grid).
    coords = image.grid().coords()
    with pytest.raises(ValueError):
        paddle.nn.functional.grid_sample(x=image, grid=coords, align_corners=True)
    result = paddle.nn.functional.grid_sample(
        x=image.batch(), grid=coords.unsqueeze(axis=0), align_corners=True
    )
    assert type(result) == Tensor
    assert tuple(result.shape)[1:] == tuple(image.shape)

    # Split image along channel dimension
    result = image.split(1)
    assert type(result) == tuple
    assert len(result) == 1
    assert type(result[0]) == Image
    assert result[0].grid() is image.grid()

    multi_channel_image = image.tile(repeat_times=(5,) + (1,) * image.grid().ndim)
    assert type(multi_channel_image) == Image
    assert tuple(multi_channel_image.shape)[0] == 5
    assert tuple(multi_channel_image.shape)[1:] == tuple(image.grid().shape)
    assert multi_channel_image.grid() is image.grid()
    result = multi_channel_image.split(2, dim=0)
    assert type(result) == tuple
    assert len(result) == 3
    assert type(result[0]) == Image
    assert type(result[1]) == Image
    assert type(result[2]) == Image
    assert tuple(result[0].shape) == (2,) + tuple(image.shape)[1:]
    assert tuple(result[1].shape) == (2,) + tuple(image.shape)[1:]
    assert tuple(result[2].shape) == (1,) + tuple(image.shape)[1:]

    # Split batch of images along batch dimension
    batch = image.batch()
    assert type(batch) == ImageBatch
    batch.grids() == (image.grid(),)
    batch.grid(0) is image.grid()
    result = batch.tensor_split(num_or_indices=[0])
    assert type(result) == tuple
    assert len(result) == 2
    assert type(result[0]) == ImageBatch
    assert len(result[0]) == 0
    assert type(result[1]) == ImageBatch
    assert len(result[1]) == 1
    assert result[1].grid(0) is image.grid()

    other: Image = image.clone()
    assert other.grid() is not image.grid()
    batch = paddle.concat(x=[image.batch(), other.batch()], axis=0)
    assert type(batch) == ImageBatch
    assert len(batch._grid) == 2
    assert batch.grid(0) is image.grid()
    assert batch.grid(1) is other.grid()

    result = batch.split(1)
    assert type(result) == tuple
    assert len(result) == 2
    assert type(result[0]) == ImageBatch
    assert len(result[0]) == 1
    assert type(result[1]) == ImageBatch
    assert len(result[1]) == 1
    assert result[0].grid(0) is image.grid()
    assert result[1].grid(0) is other.grid()

    result = paddle_aux.split(x=batch, num_or_sections=1)
    assert type(result) == tuple
    assert len(result) == 2
    assert type(result[0]) == ImageBatch
    assert len(result[0]) == 1
    assert type(result[1]) == ImageBatch
    assert len(result[1]) == 1
    assert result[0].grid(0) is image.grid()
    assert result[1].grid(0) is other.grid()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_batch(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    batch = image.batch()
    assert type(batch) is ImageBatch
    assert hasattr(batch, "_grid")
    assert type(batch._grid) is tuple
    assert len(batch._grid) == 1
    # assert batch.data_ptr() == image.data_ptr()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_batch_getitem(zeros: paddle.Tensor, grid: Grid) -> None:
    grids = [deepcopy(grid) for _ in range(5)]
    batch = ImageBatch(
        zeros.unsqueeze(axis=0).expand(shape=(5,) + tuple(zeros.shape)), grids, dtype="uint8"
    )
    for i in range(len(batch)):
        batch.tensor()[i] = i
    for i in range(len(batch)):
        image = batch[i]
        assert isinstance(image, Image)
        assert image.grid() is batch.grid(i)
        assert paddle_aux.allclose_int(
            x=image, y=paddle.to_tensor(data=i, dtype=batch.dtype)
        ).item()

    image = batch[-1]
    assert isinstance(image, Image)
    assert image.grid() is batch.grid(-1)
    assert paddle_aux.allclose_int(
        x=image, y=paddle.to_tensor(data=len(batch) - 1, dtype=batch.dtype)
    ).item()
    with pytest.raises(IndexError):
        batch[len(batch)]

    other = batch[0:1]
    assert type(other) is ImageBatch
    assert len(other.grids()) == len(other)
    assert len(other) == 1
    assert other.grid(0) is batch.grid(0)
    assert paddle_aux.allclose_int(x=other, y=paddle.to_tensor(data=0, dtype=batch.dtype)).item()

    other = batch[1:-1, 0:1]
    assert type(other) is ImageBatch
    assert len(other.grids()) == len(other)
    assert len(other) == len(batch) - 2
    for i, j in enumerate(range(1, len(batch) - 1)):
        assert other.grid(i) is batch.grid(j)
        assert paddle_aux.allclose_int(
            x=other.tensor()[i], y=paddle.to_tensor(data=j, dtype=batch.dtype)
        ).item()
    assert paddle_aux.allclose_int(x=other, y=batch.tensor()[1:-1, 0:1]).item()

    image = batch[3, :]
    assert isinstance(image, Image)
    assert image.grid() == grid
    assert paddle_aux.allclose_int(x=image, y=paddle.to_tensor(data=3, dtype=image.dtype)).item()

    other = batch[4, 0]
    assert type(other) is Tensor
    assert paddle_aux.allclose_int(x=other, y=paddle.to_tensor(data=4, dtype=image.dtype)).item()
    assert paddle_aux.allclose_int(x=other, y=batch.tensor()[4, 0]).item()

    other = batch[0, 0]
    assert type(other) is Tensor
    assert paddle_aux.allclose_int(x=other, y=batch.tensor()[0, 0]).item()

    other = batch[[0, 2]]
    assert type(other) is ImageBatch
    assert other.grids()[0] is batch.grid(0)
    assert other.grids()[1] is batch.grid(2)
    assert paddle_aux.allclose_int(x=other.tensor(), y=batch.tensor()[[0, 2]]).item()

    index = np.array([3, 4, 1])
    item = batch[index]
    assert type(item) is ImageBatch
    assert len(item) == 3
    assert item.grid(0) is batch.grid(3)
    assert item.grid(1) is batch.grid(4)
    assert item.grid(2) is batch.grid(1)
    assert paddle_aux.allclose_int(x=item.tensor(), y=batch.tensor()[index]).item()

    index = paddle.to_tensor(data=[1, 2])
    item = batch[index]
    assert type(item) is ImageBatch
    assert len(item) == 2
    assert item.grid(0) is batch.grid(1)
    assert item.grid(1) is batch.grid(2)
    assert paddle_aux.allclose_int(x=item.tensor(), y=batch.tensor()[index]).item()

    image = batch[4, 0:1]
    assert isinstance(image, Image)
    assert image.nchannels == 1
    assert image.grid() == grid
    assert paddle_aux.allclose_int(x=image, y=paddle.to_tensor(data=4, dtype=image.dtype)).item()

    other = batch[:]
    assert type(other) is type(batch)
    # assert other.data_ptr() == batch.data_ptr()

    index = slice(0, 5, 2)
    other = batch[index]
    assert type(other) is type(batch)
    assert tuple(other.shape)[0] == 3
    assert tuple(other.shape)[1:] == tuple(batch.shape)[1:]
    assert other.grid(0) is batch.grid(0)
    assert other.grid(1) is batch.grid(2)
    assert other.grid(2) is batch.grid(4)
    assert paddle_aux.allclose_int(x=other.tensor(), y=batch.tensor()[index]).item()

    other = batch[:, :, :, :]
    assert isinstance(other, ImageBatch)
    assert other.grid() == batch.grid()
    assert tuple(other.shape) == tuple(batch.shape)
    assert paddle_aux.allclose_int(x=other, y=batch).item()

    other = batch[...]
    assert isinstance(other, ImageBatch)
    assert other.grid() == batch.grid()
    assert tuple(other.shape) == tuple(batch.shape)
    assert paddle_aux.allclose_int(x=other, y=batch).item()

    other = batch[:, ...]
    assert isinstance(other, ImageBatch)
    assert other.grid() == batch.grid()
    assert tuple(other.shape) == tuple(batch.shape)
    assert paddle_aux.allclose_int(x=other, y=batch).item()

    other = batch[:, :, ...]
    assert isinstance(other, ImageBatch)
    assert other.grid() == batch.grid()
    assert tuple(other.shape) == tuple(batch.shape)
    assert paddle_aux.allclose_int(x=other, y=batch).item()

    other = batch[..., :]
    assert isinstance(other, ImageBatch)
    assert other.grid() == batch.grid()
    assert tuple(other.shape) == tuple(batch.shape)
    assert paddle_aux.allclose_int(x=other, y=batch).item()

    if batch.ndim == 4:
        other = batch[..., 2, 0:1, :, :]
        assert type(other) is Image
        assert other.grid() is batch.grid(2)
        assert paddle_aux.allclose_int(x=other, y=batch.tensor()[..., 2, 0:1, :, :]).item()
        another = batch[2, ..., 0:1, ..., :, :, ...]
        assert type(another) is Image
        assert another.grid() is batch.grid(2)
        # paddle framework does not support multiple "..." indexes .
        # assert paddle_aux.allclose_int(x=another, y=batch.tensor()[2, ..., 0:1, ..., :, :, ...]).item()
        another = batch[..., 2, ..., 0:1, ..., :, :, ...]
        assert type(another) is Image
        # assert paddle_aux.allclose_int(x=another, y=batch.tensor()[..., 2, ..., 0, ..., :, :, ...]).item()

    other = batch[:, ..., 0]
    assert type(other) is Tensor

    image = batch[3, ...]
    assert type(image) is Image
    assert image.grid() is batch.grid(3)
    assert paddle_aux.allclose_int(x=image, y=batch.tensor()[3, ...]).item()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_image_grid(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert image.grid() is grid

    new_grid = Grid(size=tuple(grid.size()), spacing=(0.5, 0.4, 0.3))

    new_image = image.grid(new_grid)
    assert new_image is not image
    assert image.grid() is grid
    assert new_image.grid() is new_grid
    # assert new_image.data_ptr() == image.data_ptr()

    other_image = image.grid_(new_grid)
    assert other_image is image
    assert image.grid() is new_grid


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_image_align_corners(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert image.align_corners() == grid.align_corners()
    assert grid.align_corners() == ALIGN_CORNERS
    grid.align_corners_(not ALIGN_CORNERS)
    assert image.align_corners() == grid.align_corners()
    assert grid.align_corners() == (not ALIGN_CORNERS)


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_image_center(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert isinstance(image.center(), paddle.Tensor)
    assert tuple(image.center().shape) == tuple(grid.center().shape)
    assert tuple(image.center().shape) == (grid.ndim,)
    assert paddle.allclose(x=image.center(), y=grid.center()).item()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_origin(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert isinstance(image.origin(), paddle.Tensor)
    assert tuple(image.origin().shape) == tuple(grid.origin().shape)
    assert tuple(image.origin().shape) == (grid.ndim,)
    assert paddle.allclose(x=image.origin(), y=grid.origin()).item()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_spacing(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert isinstance(image.spacing(), paddle.Tensor)
    assert tuple(image.spacing().shape) == tuple(grid.spacing().shape)
    assert tuple(image.spacing().shape) == (grid.ndim,)
    assert paddle.allclose(x=image.spacing(), y=grid.spacing()).item()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_direction(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert isinstance(image.direction(), paddle.Tensor)
    assert tuple(image.direction().shape) == tuple(grid.direction().shape)
    assert tuple(image.direction().shape) == (grid.ndim, grid.ndim)
    assert paddle.allclose(x=image.direction(), y=grid.direction()).item()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_sdim(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert type(image.sdim) is int
    assert image.sdim == grid.ndim


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_nchannels(zeros: paddle.Tensor, grid: Grid) -> None:
    image = Image(zeros, grid)
    assert type(image.nchannels) is int
    assert image.nchannels == 1


@pytest.mark.parametrize("data,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_normalize(data: paddle.Tensor, grid: Grid) -> None:
    atol = 1e-07  # assert result.min() == 0 occasionally fails with atol=1e-8
    image = Image(data, grid)
    assert image.min().less_than(y=paddle.to_tensor(0))
    assert image.max().greater_than(y=paddle.to_tensor(1))
    result = image.normalize(mode="unit")
    # assert type(result) is Image
    assert result is not image
    assert result.data_ptr() != image.data_ptr()
    assert image.equal(y=data).astype("bool").all()
    assert paddle.allclose(x=result.min(), y=paddle.to_tensor(data=0.0), atol=atol).item()
    assert paddle.allclose(x=result.max(), y=paddle.to_tensor(data=1.0), atol=atol).item()
    assert image.normalize().equal(y=result).astype("bool").all()

    result = image.normalize("center")
    # assert type(result) is Image
    assert result is not image
    assert result.data_ptr() != image.data_ptr()
    assert image.equal(y=data).astype("bool").all()
    assert paddle.allclose(x=result.min(), y=paddle.to_tensor(data=-0.5), atol=atol).item()
    assert paddle.allclose(x=result.max(), y=paddle.to_tensor(data=0.5), atol=atol).item()

    result = image.normalize_("center")
    # assert type(result) is Image
    assert result is not image
    assert result.data_ptr() == image.data_ptr()
    assert paddle.allclose(x=result.min(), y=paddle.to_tensor(data=-0.5), atol=atol).item()
    assert paddle.allclose(x=result.max(), y=paddle.to_tensor(data=0.5), atol=atol).item()
    # assert paddle.allclose(x=image.min(), y=paddle.to_tensor(data=-0.5), atol=atol).item()
    # assert paddle.allclose(x=image.max(), y=paddle.to_tensor(data=0.5), atol=atol).item()


@pytest.mark.parametrize("data,grid", [(d, d) for d in (2, 3)], indirect=True)
def test_image_rescale(data: paddle.Tensor, grid: Grid) -> None:
    image = Image(data, grid)
    input_min = image.min()
    input_max = image.max()
    assert input_min.less_than(y=paddle.to_tensor(0))
    assert input_max.greater_than(y=paddle.to_tensor(1))

    result = image.rescale(0, 1)
    # assert type(result) is Image
    assert result is not image
    assert result.data_ptr() != image.data_ptr()
    assert image.equal(y=data).astype("bool").all()
    assert paddle.allclose(x=result.min(), y=paddle.to_tensor(data=0.0)).item()
    assert paddle.allclose(x=result.max(), y=paddle.to_tensor(data=1.0)).item()

    assert paddle.allclose(x=image.rescale(0, 1, input_min, input_max), y=result).item()

    result = image.rescale(-0.5, 0.5, data_max=input_max)
    # assert type(result) is Image
    assert result is not image
    assert result.data_ptr() != image.data_ptr()
    assert image.equal(y=data).astype("bool").all()
    assert paddle.allclose(x=result.min(), y=paddle.to_tensor(data=-0.5)).item()
    assert paddle.allclose(x=result.max(), y=paddle.to_tensor(data=0.5)).item()

    result = image.rescale(0, 255, dtype="uint8")
    # assert type(result) is Image
    assert result is not image
    assert result.data_ptr() != image.data_ptr()
    assert result.dtype == paddle.uint8
    assert result.astype(paddle.float32).min().equal(y=0)
    assert result.astype(paddle.float32).max().equal(y=255)


def test_image_sample() -> None:
    shape = tuple((2, 32, 64, 63))
    data: Tensor = paddle.arange(end=np.prod(shape))
    data = data.reshape(shape)
    grid = Grid(shape=shape[1:])
    image = Image(data, grid)

    indices = paddle.arange(start=0, end=np.prod(grid.size()), step=10)
    voxels = U.unravel_coords(indices, tuple(grid.shape))
    coords = grid.index_to_cube(voxels)
    assert coords.dtype == grid.dtype
    assert coords.is_floating_point()
    assert tuple(coords.shape) == (len(indices), grid.ndim)
    assert coords.min().greater_equal(y=paddle.to_tensor(-1))
    # assert coords.max().less_equal(y=paddle.to_tensor(1))

    result = image.sample(coords, mode="nearest")
    expected = data.flatten(start_axis=1).index_select(axis=1, index=indices)
    assert type(result) is Tensor
    assert result.dtype == image.dtype
    assert tuple(result.shape) == (image.nchannels, *tuple(coords.shape)[:-1])
    assert tuple(result.shape) == tuple(expected.shape)
    # assert result.equal(y=expected).astype("bool").all()

    result = image.sample(coords, mode="linear")
    assert type(result) is Tensor
    assert result.is_floating_point()
    assert result.dtype != image.dtype

    # Grid points
    coords = grid.coords()
    assert coords.ndim == image.ndim
    result = image.sample(coords, mode="nearest")
    assert type(result) is Tensor
    assert result.dtype == image.dtype
    assert paddle_aux.is_eq_place(result.place, image.place)
    assert tuple(result.shape)[0] == image.nchannels
    assert tuple(result.shape)[1:] == tuple(grid.shape)
    # assert paddle.allclose(x=result, y=image._data).item()

    # Batch of grid points
    coords = grid.coords()
    coords = coords.unsqueeze(axis=0).tile(repeat_times=(3,) + (1,) * coords.ndim)
    assert coords.ndim == image.ndim + 1
    result = image.sample(coords, mode="nearest")
    assert type(result) is Tensor
    assert result.dtype == image.dtype
    assert paddle_aux.is_eq_place(result.place, image.place)
    assert tuple(result.shape)[0] == image.nchannels
    assert tuple(result.shape)[1] == tuple(coords.shape)[0]
    assert tuple(result.shape)[2:] == tuple(grid.shape)

    batch_result = image.batch().sample(coords, mode="nearest")
    assert type(result) is Tensor
    assert batch_result.dtype == image.dtype
    assert paddle_aux.is_eq_place(batch_result.place, image.place)
    assert tuple(batch_result.shape)[0] == tuple(coords.shape)[0]
    assert tuple(batch_result.shape)[1] == image.nchannels
    assert tuple(batch_result.shape)[2:] == tuple(grid.shape)
    assert paddle_aux.allclose_int(
        x=result,
        y=batch_result.transpose(perm=paddle_aux.transpose_aux_func(batch_result.ndim, 0, 1)),
    ).item()

    # Sampling grid
    result = image.sample(grid)
    assert type(result) is Image
    assert result.grid() is grid
    assert result.data_ptr() == image.data_ptr()
    assert result is image

    shape = tuple((3, 31, 63, 63))
    data: Tensor = paddle.arange(end=np.prod(shape))
    data = data.reshape(shape)
    grid = Grid(shape=shape[1:])
    image = Image(data, grid)

    ds_grid = grid.downsample()
    ds_image = image.sample(ds_grid)
    assert type(ds_image) is Image
    assert ds_image.grid() is ds_grid
    assert ds_image.nchannels == image.nchannels
    assert paddle_aux.is_eq_place(ds_image.place, image.place)
    assert ds_image.is_floating_point()

    indices = paddle.arange(start=0, end=np.prod(grid.size()))
    voxels = U.unravel_coords(indices, grid.size())
    coords = grid.index_to_cube(voxels)
    coords = coords.reshape(tuple(grid.shape) + (grid.ndim,))
    coords = coords[::2, ::2, ::2, :]
    assert paddle.allclose(x=coords, y=ds_grid.coords()).item()

    result = image.sample(coords)
    assert result.is_floating_point()
    assert paddle.allclose(x=ds_image, y=result).item()
