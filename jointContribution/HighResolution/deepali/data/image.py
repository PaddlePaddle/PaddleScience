from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import overload

import numpy as np
import paddle

try:
    import SimpleITK as sitk

    from ..utils.sitk.imageio import read_image
    from ..utils.sitk.paddle import image_from_tensor
    from ..utils.sitk.paddle import tensor_from_image
except ImportError:
    sitk = None
from ..core import image as U
from ..core.cube import Cube
from ..core.enum import PaddingMode
from ..core.enum import Sampling
from ..core.enum import SpatialDimArg
from ..core.grid import ALIGN_CORNERS
from ..core.grid import Axes
from ..core.grid import Grid
from ..core.grid import grid_transform_points
from ..core.itertools import zip_longest_repeat_last
from ..core.path import unlink_or_mkdir
from ..core.tensor import cat_scalars
from ..core.types import Array
from ..core.types import Device
from ..core.types import DType
from ..core.types import EllipsisType
from ..core.types import PathStr
from ..core.types import Scalar
from ..core.types import ScalarOrTuple
from ..core.types import Size
from .tensor import DataTensor

Domain = Cube
TImage = TypeVar("TImage", bound="Image")
TImageBatch = TypeVar("TImageBatch", bound="ImageBatch")
__all__ = "Image", "ImageBatch"


class ImageBatch(DataTensor):
    """Batch of images sampled on regular oriented grids."""

    def __init__(
        self: TImageBatch,
        data: Array,
        grid: Optional[Union[Grid, Sequence[Grid]]] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        """Initialize image decorator.

        Args:
            data: Image batch data tensor of shape (N, C, ...X).
            grid: Sampling grid of image data oriented in world space. Can be either a single shared
                sampling grid, or a separate grid for each image in the batch. Note that operations
                which would result in differently sized images (e.g., resampling to a certain voxel
                size, when images have different resolutions) will raise an exception. All images in
                a batch must have the same number of channels and spatial size. If ``None``, a default
                grid whose world space coordinate axes are aligned with the image axes, unit spacing,
                and origin at the image centers is created. By default, image grid attributes are always
                stored in CPU memory, regardless of the ``device`` on which the image data is located.
            dtype: Data type of the image data. A copy of the data is only made when the desired ``dtype``
                is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store image data. A copy of the data is only made when the data
                has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned image tensor.
            pin_memory: If set, returned image tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        if self.ndim < 4:
            raise ValueError("Image batch tensor must have at least four dimensions")
        self.grid_(grid)

    def _make_instance(
        self: TImageBatch,
        data: Optional[paddle.Tensor] = None,
        grid: Optional[Sequence[Grid]] = None,
        **kwargs,
    ) -> TImageBatch:
        """Create a new instance while preserving subclass (meta-)data."""
        kwargs["grid"] = self._grid if grid is None else grid
        return super()._make_instance(data, **kwargs)

    def _make_subitem(self, data: paddle.Tensor, grid: Grid) -> Image:
        """Create Image in __getitem__. Can be overridden by subclasses to return a subtype."""
        return Image(data, grid)

    def __deepcopy__(self: TImageBatch, memo) -> TImageBatch:
        if id(self) in memo:
            return memo[id(self)]
        result = self._make_instance(
            self.data.clone(),
            grid=tuple(grid.clone() for grid in self._grid),
            requires_grad=self.requires_grad,
            pin_memory=self.is_pinned(),
        )
        memo[id(self)] = result
        return result

    @staticmethod
    def _paddle_function_grid(
        func, args, kwargs: Dict[str, Any]
    ) -> Optional[Union[Sequence[Grid], Sequence[Sequence[Grid]]]]:
        """Get spatial sampling grids from args passed to __paddle_function__."""
        if not args:
            return None
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        grids: Sequence[Sequence[Grid]]
        grids = [
            g for g in (getattr(arg, "_grid", None) for arg in args) if g is not None
        ]
        if not grids:
            return None
        if kwargs.get("dim", 0) == 0:
            if func == paddle.concat:
                return [g for grid in grids for g in grid]
            if func in (paddle.split, paddle.Tensor.split):
                grids = grids[0]
                split_grids = []
                split_size_or_sections = args[1]
                if isinstance(split_size_or_sections, int):
                    for start in range(0, len(grids), split_size_or_sections):
                        split_grids.append(
                            grids[start : start + split_size_or_sections]
                        )
                elif isinstance(split_size_or_sections, Sequence):
                    start = 0
                    for num in split_size_or_sections:
                        split_grids.append(grids[start : start + num])
                return split_grids
            if func in (paddle.split_with_sizes, paddle.Tensor.split_with_sizes):
                grids = grids[0]
                split_grids = []
                split_sizes = args[1]
                start = 0
                for num in split_sizes:
                    split_grids.append(grids[start : start + num])
                return split_grids
            if func in (paddle.tensor_split, paddle.Tensor.tensor_split):
                grids = grids[0]
                split_grids = []
                tensor_indices_or_sections = args[1]
                if isinstance(tensor_indices_or_sections, int):
                    for start in range(0, len(grids), tensor_indices_or_sections):
                        split_grids.append(
                            grids[start : start + tensor_indices_or_sections]
                        )
                elif isinstance(tensor_indices_or_sections, Sequence):
                    indices = list(tensor_indices_or_sections)
                    for start, end in zip([0] + indices, indices + [len(grids)]):
                        split_grids.append(grids[start:end])
                return split_grids
        return grids[0]

    @classmethod
    def _paddle_function_result(cls, func, data, grid: Optional[Sequence[Grid]]) -> Any:
        if not isinstance(data, paddle.Tensor):
            return data
        if (
            grid
            and data.ndim == grid[0].ndim + 2
            and tuple(data.shape)[0] == len(grid)
            and tuple(data.shape)[2:] == tuple(grid[0].shape)
            or grid is not None
            and len(grid) == 0
            and data.ndim >= 4
            and tuple(data.shape)[0] == 0
        ):
            if func in (paddle.clone, paddle.Tensor.clone):
                grid = [g.clone() for g in grid]
            if isinstance(data, cls):
                data._grid = grid
            else:
                data = cls(data, grid)
        elif type(data) != paddle.Tensor:
            data = data.as_subclass(paddle.Tensor)
        return data

    @classmethod
    def __paddle_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = tuple(arg.batch() if isinstance(arg, Image) else arg for arg in args)
        data = paddle.Tensor.__paddle_function__(func, (paddle.Tensor,), args, kwargs)
        grid = cls._paddle_function_grid(func, args, kwargs)
        if func in (paddle.nn.functional.grid_sample,):
            grid = None
        elif func in (
            paddle.split,
            paddle.Tensor.split,
            paddle.split_with_sizes,
            paddle.Tensor.split_with_sizes,
            paddle.tensor_split,
            paddle.Tensor.tensor_split,
        ):
            if type(data) not in (tuple, list):
                raise AssertionError(
                    f"expected split 'data' to be tuple or list, got {type(data)}"
                )
            if type(grid) not in (tuple, list):
                raise AssertionError(
                    f"expected split 'grid' to be tuple or list, got {type(grid)}"
                )
            if len(grid) != len(data):
                raise AssertionError(
                    f"expected 'grid' tuple length to be equal batch size, but {len(grid)} != {len(data)}"
                )
            assert all(isinstance(d, paddle.Tensor) for d in data)
            assert all(isinstance(g, (tuple, list)) for g in grid)
            assert all(len(d) == len(g) for d, g in zip(data, grid))
            return tuple(
                cls._paddle_function_result(func, d, g) for d, g in zip(data, grid)
            )
        return cls._paddle_function_result(func, data, grid)

    def cube(self: TImageBatch, n: int = 0) -> Cube:
        """Get cube of n-th image in batch defining its normalized coordinates space with respect to the world."""
        return self._grid[n].cube()

    def cubes(self: TImageBatch) -> Tuple[Cube, ...]:
        """Get cubes of all images which define their normalized coordinates spaces with respect to the world."""
        return tuple(grid.cube() for grid in self._grid)

    def domain(self: TImageBatch, n: int = 0) -> Domain:
        """Get oriented bounding box of n-th image in world space which defines the domain within which it is defined."""
        return self._grid[n].domain()

    def domains(self: TImageBatch) -> Tuple[Domain, ...]:
        """Get oriented bounding boxes of all images in world space which define the domains within which these are defined."""
        return tuple(grid.domain() for grid in self._grid)

    def same_domains_as(self, other: ImageBatch) -> bool:
        """Check if images in this batch and another batch have the same cube domains."""
        if len(self) != len(other):
            return False
        return all(a.same_domain_as(b) for a, b in zip(self.grids(), other.grids()))

    @overload
    def grid(self: TImageBatch, n: int = 0) -> Grid:
        """Get sampling grid of n-th image in batch."""
        ...

    @overload
    def grid(self: TImageBatch, arg: Union[Grid, Sequence[Grid]]) -> TImageBatch:
        """Get new image batch with specified sampling grid(s)."""
        ...

    def grid(
        self: TImageBatch, arg: Optional[Union[int, Grid, Sequence[Grid]]] = None
    ) -> Union[Grid, TImageBatch]:
        """Get sampling grid of images in batch or new batch with specified grid, respectively."""
        if arg is None:
            arg = 0
        if isinstance(arg, int):
            return self._grid[arg]
        return self._make_instance(grid=arg)

    def grid_(self: TImageBatch, arg: Union[Grid, Sequence[Grid], None]) -> TImageBatch:
        """Change image sampling grid of this image batch."""
        shape = self.shape
        if arg is None:
            arg = (Grid(shape=shape[2:]),) * shape[0]
        elif isinstance(arg, Grid):
            grid = arg
            if tuple(grid.shape) != tuple(shape[2:]):
                raise ValueError(
                    "Image grid size does not match spatial dimensions of image batch tensor"
                )
            arg = (grid,) * shape[0]
        else:
            arg = tuple(arg)
            if any(tuple(grid.shape) != tuple(shape[2:]) for grid in arg):
                raise ValueError(
                    "Image grid sizes must match spatial dimensions of image batch tensor"
                )
        self._grid = arg
        return self

    def grids(self: TImageBatch) -> Tuple[Grid, ...]:
        """Get sampling grids of images in batch."""
        return self._grid

    def align_corners(self: TImageBatch) -> bool:
        """Whether image resizing operations by default preserve corner points or grid extent."""
        return self._grid[0].align_corners()

    def center(self: TImageBatch) -> paddle.Tensor:
        """Image centers in world space coordinates as tensor of shape (N, D)."""
        return paddle.concat(
            x=[grid.center().unsqueeze(axis=0) for grid in self.grids()], axis=0
        )

    def origin(self: TImageBatch) -> paddle.Tensor:
        """Image origins in world space coordinates as tensor of shape (N, D)."""
        return paddle.concat(
            x=[grid.origin().unsqueeze(axis=0) for grid in self.grids()], axis=0
        )

    def spacing(self: TImageBatch) -> paddle.Tensor:
        """Image spacings in world units as tensor of shape (N, D)."""
        return paddle.concat(
            x=[grid.spacing().unsqueeze(axis=0) for grid in self.grids()], axis=0
        )

    def direction(self: TImageBatch) -> paddle.Tensor:
        """Image direction cosines matrices as tensor of shape (N, D, D)."""
        return paddle.concat(
            x=[grid.direction().unsqueeze(axis=0) for grid in self.grids()], axis=0
        )

    def __len__(self: TImageBatch) -> int:
        """Number of images in batch."""
        return self.shape[0]

    @overload
    def __getitem__(self: TImageBatch, index: int) -> Image:
        ...

    @overload
    def __getitem__(self: TImageBatch, index: EllipsisType) -> TImageBatch:
        ...

    @overload
    def __getitem__(
        self: TImageBatch, index: Union[list, slice, paddle.Tensor]
    ) -> TImageBatch:
        ...

    def __getitem__(
        self: TImageBatch,
        index: Union[
            EllipsisType, int, slice, Sequence[Union[EllipsisType, int, slice]]
        ],
    ) -> Union[Image, TImageBatch, paddle.Tensor]:
        """Get image at specified batch index, get a sub-batch, or a region of interest tensor."""
        if index is ...:
            return self._make_instance(self.tensor(), self.grid())
        if type(index) is tuple:
            index = [
                j for i, j in enumerate(index) if j is not ... or ... not in index[:i]
            ]
            if index[-1] is ...:
                index = index[:-1]
            try:
                i = index.index(...)
                j = len(index) - i - 1
                index = index[:i] + [slice(None)] * (self.ndim - i - j) + index[-j:]
            except ValueError:
                pass
            index = tuple(index)
            is_multi_index = True
        elif isinstance(index, (np.ndarray, slice, Sequence, paddle.Tensor)):
            index = (index,)
            is_multi_index = True
        else:
            index = int(index)
            is_multi_index = False
        data = self.tensor()[index]
        if is_multi_index and len(index) > 1 and isinstance(index[1], int):
            return data
        grid_index = index[0] if is_multi_index else index
        if isinstance(grid_index, (np.ndarray, Sequence, paddle.Tensor)):
            grid = tuple(self._grid[i] for i in grid_index)
        else:
            grid = self._grid[grid_index]
        if is_multi_index and len(index) > 2:
            same_grid = True
            for i, n in zip(index[2:], self.shape[2:]):
                if isinstance(i, int) or not isinstance(i, slice):
                    same_grid = False
                    break
                if (
                    i.start not in (None, 0)
                    or i.stop not in (None, n)
                    or i.step not in (None, 1)
                ):
                    same_grid = False
                    break
            if not same_grid:
                return data
        if isinstance(grid, Grid):
            if data.ndim < 3:
                return data
            return self._make_subitem(data, grid)
        elif data.ndim < 4:
            return data
        return self._make_instance(data, grid)

    @property
    def sdim(self: TImageBatch) -> int:
        """Number of spatial dimensions."""
        return self.ndim - 2

    @property
    def nchannels(self: TImageBatch) -> int:
        """Number of image channels."""
        return self.shape[1]

    def normalize(
        self: TImageBatch,
        mode: str = "unit",
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> TImageBatch:
        """Normalize image intensities in [min, max]."""
        return U.normalize_image(self, mode=mode, min=min, max=max)

    def normalize_(
        self: TImageBatch,
        mode: str = "unit",
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> TImageBatch:
        """Normalize image intensities in [min, max]."""
        return U.normalize_image(self, mode=mode, min=min, max=max, inplace=True)

    def rescale(
        self: TImageBatch,
        min: Optional[Scalar] = None,
        max: Optional[Scalar] = None,
        data_min: Optional[Scalar] = None,
        data_max: Optional[Scalar] = None,
        dtype: Optional[DType] = None,
    ) -> TImageBatch:
        """Clamp and linearly rescale image values."""
        return U.rescale(
            self, min, max, data_min=data_min, data_max=data_max, dtype=dtype
        )

    def narrow(self: TImageBatch, dim: int, start: int, length: int) -> TImageBatch:
        """Narrow image batch along specified tensor dimension."""
        start_27 = self.tensor().shape[dim] + start if start < 0 else start
        data = paddle.slice(self.tensor(), [dim], [start_27], [start_27 + length])
        grid = self.grid()
        if dim > 1:
            start_28 = grid.shape[self.ndim - dim - 1] + start if start < 0 else start
            grid = paddle.slice(
                grid, [self.ndim - dim - 1], [start_28], [start_28 + length]
            )
        return self._make_instance(data, grid)

    def resize(
        self: TImageBatch,
        size: Union[int, Array, Size],
        *args: int,
        mode: Union[Sampling, str] = Sampling.LINEAR,
        align_corners: Optional[bool] = None,
    ) -> TImageBatch:
        """Interpolate images on grid with specified size.

        Args:
            size: Size of spatial image dimensions, where the size of the last tensor dimension,
                which corresponds to the first grid dimension, must be given first, e.g., ``(nx, ny, nz)``.
            mode: Image data interpolation mode.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Image batch with specified size of spatial dimensions.

        """
        if align_corners is None:
            align_corners = self.align_corners()
        size = cat_scalars(size, *args, num=self.sdim, device=self.device)
        data = U.grid_resize(self, size, mode=mode, align_corners=align_corners)
        grid = tuple(
            grid.resize(size, align_corners=align_corners) for grid in self._grid
        )
        return self._make_instance(data, grid)

    def resample(
        self: TImageBatch,
        spacing: Union[float, Array, str],
        *args: float,
        mode: Union[Sampling, str] = Sampling.LINEAR,
    ) -> TImageBatch:
        """Interpolate images on grid with specified spacing.

        Args:
            spacing: Spacing of grid on which to resample image data, where the spacing corresponding
                to first grid dimension, which corresponds to the last tensor dimension, must be given
                first, e.g., ``(sx, sy, sz)``. Alternatively, can be string 'min' or 'max' to resample
                to the minimum or maximum voxel size, respectively.
            mode: Image data interpolation mode.

        Returns:
            Image batch with given grid spacing.

        """
        in_spacing = self._grid[0].spacing()
        if not all(
            paddle.allclose(x=grid.spacing(), y=in_spacing).item()
            for grid in self._grid
        ):
            raise AssertionError(
                f"{type(self).__name__}.resample() requires all images in batch to have the same grid spacing"
            )
        if spacing == "min":
            assert not args
            out_spacing = in_spacing.min()
        elif spacing == "max":
            assert not args
            out_spacing = in_spacing.max()
        else:
            out_spacing = spacing
        out_spacing = cat_scalars(out_spacing, *args, num=self.sdim, device=self.device)
        data = U.grid_resample(
            self, in_spacing=in_spacing, out_spacing=out_spacing, mode=mode
        )
        grid = tuple(grid.resample(out_spacing) for grid in self._grid)
        return self._make_instance(data, grid)

    def avg_pool(
        self: TImageBatch,
        kernel_size: ScalarOrTuple[int],
        stride: Optional[ScalarOrTuple[int]] = None,
        padding: ScalarOrTuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> TImageBatch:
        """Average pooling of image data."""
        data = U.avg_pool(
            self,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        grid = tuple(
            grid.avg_pool(
                kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
            )
            for grid in self._grid
        )
        return self._make_instance(data, grid)

    def downsample(
        self: TImageBatch,
        levels: int = 1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        min_size: int = 0,
        align_corners: Optional[bool] = None,
    ) -> TImageBatch:
        """Downsample images in batch by halving their size the specified number of times.

        Args:
            levels: Number of times the image size is halved (>0) or doubled (<0).
            dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
            sigma: Standard deviation of Gaussian filter applied at each downsampling level.
            mode: Image interpolation mode.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Batch of downsampled images.

        """
        if not isinstance(levels, int):
            raise TypeError(
                f"{type(self).__name__}.downsample() 'levels' must be of type int"
            )
        if align_corners is None:
            align_corners = self.align_corners()
        data = U.downsample(
            self,
            levels,
            dims=dims,
            sigma=sigma,
            mode=mode,
            min_size=min_size,
            align_corners=align_corners,
        )
        grid = tuple(
            grid.downsample(
                levels, dims=dims, min_size=min_size, align_corners=align_corners
            )
            for grid in self._grid
        )
        return self._make_instance(data, grid)

    def upsample(
        self: TImageBatch,
        levels: int = 1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> TImageBatch:
        """Upsample image in batch by doubling their size the specified number of times.

        Args:
            levels: Number of times the image size is doubled (>0) or halved (<0).
            dims: Spatial dimensions along which to upsample. If not specified, consider all spatial dimensions.
            sigma: Standard deviation of Gaussian filter applied at each downsampling level.
            mode: Image interpolation mode.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Batch of upsampled images.

        """
        if not isinstance(levels, int):
            raise TypeError(
                f"{type(self).__name__}.upsample() 'levels' must be of type int"
            )
        if align_corners is None:
            align_corners = self.align_corners()
        data = U.upsample(
            self, levels, dims=dims, sigma=sigma, mode=mode, align_corners=align_corners
        )
        grid = tuple(
            grid.upsample(levels, dims=dims, align_corners=align_corners)
            for grid in self._grid
        )
        return self._make_instance(data, grid)

    def pyramid(
        self: TImageBatch,
        levels: int,
        start: int = 0,
        end: int = -1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        spacing: Optional[float] = None,
        min_size: int = 0,
        align_corners: Optional[bool] = None,
    ) -> Dict[int, TImageBatch]:
        """Create Gaussian resolution pyramid.

        Args:
            levels: Number of resolution levels.
            start: Highest resolution level to return, where 0 corresponds to the finest resolution.
            end: Lowest resolution level to return (inclusive).
            dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
            sigma: Standard deviation of Gaussian filter applied at each downsampling level.
            mode: Interpolation mode for resampling image data on downsampled grid.
            spacing: Grid spacing at finest resolution level. Note that this option may increase the
                cube extent of the multi-resolution pyramid sampling grids.
            min_size: Minimum grid size.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Dictionary of downsampled image batches with keys corresponding to level indices.

        """
        if not isinstance(levels, int):
            raise TypeError(f"{type(self).__name__}.pyramid() 'levels' must be int")
        if not isinstance(start, int):
            raise TypeError(f"{type(self).__name__}.pyramid() 'start' must be int")
        if not isinstance(end, int):
            raise TypeError(f"{type(self).__name__}.pyramid() 'end' must be int")
        if start < 0:
            start = levels + start
        if start < 0 or start > levels - 1:
            raise ValueError(
                f"{type(self).__name__}.pyramid() 'start' must be in [{-levels}, {levels - 1}]"
            )
        if end < 0:
            end = levels + end
        if end < 0 or end > levels - 1:
            raise ValueError(
                f"{type(self).__name__}.pyramid() 'end' must be in [{-levels}, {levels - 1}]"
            )
        if start > end:
            return {}
        if align_corners is None:
            align_corners = self.align_corners()
        grids = tuple(grid.align_corners(align_corners) for grid in self._grid)
        if spacing is not None:
            spacing0 = grids[0].spacing()
            if not all(
                paddle.allclose(x=grid.spacing(), y=spacing0).item() for grid in grids
            ):
                raise AssertionError(
                    f"{type(self).__name__}.pyramid() requires all images to have the same grid spacing when output 'spacing' at finest level is specified"
                )
            grids = tuple(grid.resample(spacing) for grid in grids)
        grids = tuple(
            grid.pyramid(levels, dims=dims, min_size=min_size)[0] for grid in grids
        )
        assert all(tuple(grid.shape) == tuple(grids[0].shape) for grid in grids)
        if paddle.allclose(
            x=grids[0].cube_extent(), y=self._grid[0].cube_extent()
        ).item():
            size = tuple(grids[0].shape)
            data = U.grid_resize(self, size, mode=mode, align_corners=align_corners)
        else:
            points = grids[0].coords(device=self.device)
            data = U.grid_sample(self, points, mode=mode, align_corners=align_corners)
        pyramid = {}
        batch = self._make_instance(data, grids)
        if start == 0:
            pyramid[0] = batch
        for level in range(1, end + 1):
            batch = batch.downsample(
                dims=dims, sigma=sigma, mode=mode, min_size=min_size
            )
            if level >= start:
                pyramid[level] = batch
        return pyramid

    def crop(
        self: TImageBatch,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImageBatch:
        """Crop images at boundary.

        Args:
            margin: Number of spatial grid points to remove (positive) or add (negative) at each border.
                Use instead of ``num`` in order to symmetrically crop the input ``data`` tensor, e.g.,
                ``(nx, ny, nz)`` is equivalent to ``num=(nx, nx, ny, ny, nz, nz)``.
            num: Number of spatial gird points to remove (positive) or add (negative) at each border,
                where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
                ``(nx, nx, ny, ny)``. If a scalar is given, the input is cropped equally at all borders.
                Otherwise, the given sequence must have an even length.
            mode: Image extrapolation mode in case of negative crop value.
            value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

        Returns:
            Batch of images with modified size, but unchanged spacing.

        """
        data = U.crop(self, margin=margin, num=num, mode=mode, value=value)
        grid = tuple(grid.crop(margin=margin, num=num) for grid in self._grid)
        return self._make_instance(data, grid)

    def pad(
        self: TImageBatch,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImageBatch:
        """Pad images at boundary.

        Args:
            margin: Number of spatial grid points to add (positive) or remove (negative) at each border,
                Use instead of ``num`` in order to symmetrically pad the input ``data`` tensor.
            num: Number of spatial gird points to add (positive) or remove (negative) at each border,
                where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
                ``(nx, ny, nz)``. If a scalar is given, the input is padded equally at all borders.
                Otherwise, the given sequence must have an even length.
            mode: Image extrapolation mode in case of positive pad value.
            value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

        Returns:
            Batch of images with modified size, but unchanged spacing.

        """
        data = U.pad(self, margin=margin, num=num, mode=mode, value=value)
        grid = tuple(grid.pad(margin=margin, num=num) for grid in self._grid)
        return self._make_instance(data, grid)

    def center_crop(
        self: TImageBatch, size: Union[int, Array], *args: int
    ) -> TImageBatch:
        """Crop image to specified maximum size.

        Args:
            size: Maximum output size, where the size of the last tensor
                dimension must be given first, i.e., ``(X, ...)``.
                If an ``int`` is given, all spatial output dimensions
                are cropped to this maximum size. If the length of size
                is less than the spatial dimensions of the ``data`` tensor,
                then only the last ``len(size)`` dimensions are modified.

        Returns:
            Batch of cropped images.

        """
        size = cat_scalars(size, *args, num=self.sdim, device=self.device)
        data = U.center_crop(self.tensor(), size)
        grid = tuple(grid.center_crop(size) for grid in self._grid)
        return self._make_instance(data, grid)

    def center_pad(
        self: TImageBatch,
        size: Union[int, Array],
        *args: int,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImageBatch:
        """Pad image to specified minimum size.

        Args:
            size: Minimum output size, where the size of the last tensor
                dimension must be given first, i.e., ``(X, ...)``.
                If an ``int`` is given, all spatial output dimensions
                are cropped to this maximum size. If the length of size
                is less than the spatial dimensions of the ``data`` tensor,
                then only the last ``len(size)`` dimensions are modified.
            mode: Padding mode (cf. ``paddle.nn.functional.pad()``).
            value: Value for padding mode "constant".

        Returns:
            Batch of images with modified size, but unchanged spacing.

        """
        size = cat_scalars(size, *args, num=self.sdim, device=self.device)
        data = U.center_pad(self, size, mode=mode, value=value)
        grid = tuple(grid.center_pad(size) for grid in self._grid)
        return self._make_instance(data, grid)

    def region_of_interest(
        self: TImageBatch,
        start: Union[int, Array],
        size: Union[int, Array],
        padding: Union[PaddingMode, str, float] = PaddingMode.CONSTANT,
        value: float = 0,
    ) -> TImageBatch:
        """Extract region of interest from images in batch.

        Args:
            start: Indices of first spatial point to include in region of interest.
            size: Size of region of interest.
            padding: Image extrapolation mode or fill value.
            value: Fill value to use when ``padding=Padding.CONSTANT``.

        Returns:
            Image batch of extracted image region of interest.

        """
        data = U.region_of_interest(self, start, size, padding=padding, value=value)
        grid = tuple(grid.region_of_interest(start, size) for grid in self._grid)
        return self._make_instance(data, grid)

    def conv(
        self: TImageBatch,
        kernel: Union[paddle.Tensor, Sequence[Optional[paddle.Tensor]]],
        padding: Union[PaddingMode, str, int] = None,
    ) -> TImageBatch:
        """Filter images in batch with a given convolutional kernel.

        Args:
            kernel: Weights of kernel used to filter the images in this batch by.
                The dtype of the kernel defines the intermediate data type used for convolutions.
                If a 1-dimensional kernel is given, it is used as seperable convolution kernel in
                all spatial image dimensions. Otherwise, the kernel is applied to the last spatial
                image dimensions. For example, a 2D kernel applied to a batch of 3D image volumes
                is applied slice-by-slice by convolving along the X and Y image axes.
            padding: Image padding mode or margin size. If ``None``, use default mode ``PaddingMode.ZEROS``.

        Returns:
            Result of filtering operation with data type set to the image data type before convolution.
            If this data type is not a floating point data type, the filtered data is rounded and clamped
            before it is being cast to the original dtype.

        """
        data = U.conv(self, kernel, padding=padding)
        crop = tuple(
            (m - n) // 2 for m, n in zip(self.shape[2:], tuple(data.shape)[2:])
        )
        crop = tuple(reversed(crop))
        grid = tuple(grid.crop(crop) for grid in self._grid)
        return self._make_instance(data, grid)

    @overload
    def sample(
        self: TImageBatch,
        grid: Union[Grid, Sequence[Grid]],
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> TImageBatch:
        """Sample images at optionally deformed unit grid points.

        Args:
            grid: Spatial grid points at which to sample image values.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            A new image batch of the resampled data with the given sampling grids.

        """
        ...

    @overload
    def sample(
        self: TImageBatch,
        coords: paddle.Tensor,
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> paddle.Tensor:
        """Sample images at optionally deformed unit grid points.

        Args:
            coords: Normalized coordinates at which to sample image values as tensor of
                shape ``(N, ..., D)`` or ``(1, ..., D)``. Note that the shape ``...`` need
                not correspond to a (deformed) grid as required by ``grid_sample()``.
                It can be an arbitrary shape, e.g., ``M`` to sample at ``M`` given points.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            A tensor of shape (N, C, ...) of sampled image values.

        """
        ...

    def sample(
        self: TImageBatch,
        arg: Union[Grid, Sequence[Grid], paddle.Tensor],
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> Union[TImageBatch, paddle.Tensor]:
        """Sample images at optionally deformed unit grid points.

        Args:
            arg: Either a single grid which defines the sampling points for all images in the batch,
                a different grid for each image in the batch, or a tensor of normalized coordinates
                with shape ``(N, ..., D)`` or ``(1, ..., D)``. In the latter case, note that the
                shape ``...`` need not correspond to a (deformed) grid as required by ``grid_sample()``.
                It can be an arbitrary shape, e.g., ``M`` to sample at ``M`` given points.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            If ``arg`` is of type ``Grid`` or ``Sequence[Grid]``, an ``ImageBatch`` is returned.
            When these grids match the grids of this image batch, ``self`` is returned.
            Otherwise, a ``paddle.Tensor`` of shape (N, C, ...) of sampled image values is returned.

        """
        data = self.tensor()
        align_corners = self.align_corners()
        if isinstance(arg, paddle.Tensor):
            return U.sample_image(
                data, arg, mode=mode, padding=padding, align_corners=align_corners
            )
        if isinstance(arg, Grid):
            arg = (arg,)
        elif not isinstance(arg, Sequence) or any(
            not isinstance(item, Grid) for item in arg
        ):
            raise TypeError(
                f"{type(self).__name__}.sample() 'arg' must be Grid, Sequence[Grid], or paddle.Tensor"
            )
        if len(arg) not in (1, len(self)):
            raise ValueError(
                f"{type(self).__name__}.sample() 'arg' must be one or {len(self)} grids"
            )
        if all(grid == g for grid, g in zip_longest_repeat_last(arg, self._grid)):
            return self
        axes = Axes.from_align_corners(align_corners)
        coords = [
            grid.coords(align_corners=align_corners, device=self.device) for grid in arg
        ]
        coords = paddle.concat(
            x=[
                grid_transform_points(p, grid, axes, to_grid, axes).unsqueeze(0)
                for p, grid, to_grid in zip_longest_repeat_last(coords, arg, self._grid)
            ],
            axis=0,
        )
        data = U.grid_sample(
            data, coords, mode=mode, padding=padding, align_corners=align_corners
        )
        return self._make_instance(data, arg)

    def __repr__(self) -> str:
        return type(self).__name__ + f"(data={self.tensor()!r}, grid={self.grids()!r})"

    def __str__(self) -> str:
        return type(self).__name__ + f"(data={self.tensor()!s}, grid={self.grids()!s})"


class Image(DataTensor):
    """Image sampled on oriented grid."""

    def __init__(
        self: TImage,
        data: Array,
        grid: Optional[Grid] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        """Initialize image tensor.

        Args:
            data: Image data tensor of shape (C, ...X). To create an ``Image`` instance from an
                image of a mini-batch without creating a copy of the data, simply provide the
                respective slice of the mini-batch corresponding to this image, e.g., ``batch[i]``.
            grid: Sampling grid of image ``data`` oriented in world space.
                If ``None``, a default grid whose world space coordinate axes are aligned with the
                image axes, unit spacing, and origin at the image center is created on CPU.
            dtype: Data type of the image data. A copy of the data is only made when the desired
                ``dtype`` is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store image data. A copy of the data is only made when
                the data has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned image tensor.
            pin_memory: If set, returned image tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        if self.ndim < 3:
            raise ValueError(
                "Image tensor must have at least three dimensions (C, H, W)"
            )
        self.grid_(grid)

    def _make_instance(
        self: TImage,
        data: Optional[paddle.Tensor] = None,
        grid: Optional[Grid] = None,
        **kwargs,
    ) -> TImage:
        """Create a new instance while preserving subclass meta-data."""
        kwargs["grid"] = self._grid if grid is None else grid
        return super()._make_instance(data, **kwargs)

    def __deepcopy__(self: TImage, memo) -> TImage:
        if id(self) in memo:
            return memo[id(self)]
        result = self._make_instance(
            self.data.clone(),
            grid=self._grid.clone(),
            requires_grad=self.requires_grad,
            pin_memory=self.is_pinned(),
        )
        memo[id(self)] = result
        return result

    @staticmethod
    def _paddle_function_grid(args) -> Optional[Grid]:
        """Get spatial sampling grid from args passed to __paddle_function__."""
        if not args:
            return None
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        grids: Sequence[Grid]
        grids = [
            g for g in (getattr(arg, "_grid", None) for arg in args) if g is not None
        ]
        if not grids:
            return None
        return grids[0]

    @classmethod
    def _paddle_function_result(cls, func, data, grid: Optional[Grid]) -> Any:
        if not isinstance(data, paddle.Tensor):
            return data
        if (
            grid is not None
            and data.ndim == grid.ndim + 1
            and tuple(data.shape)[1:] == tuple(grid.shape)
        ):
            if func in (paddle.clone, paddle.Tensor.clone):
                grid = grid.clone()
            if isinstance(data, cls):
                data._grid = grid
            else:
                data = cls(data, grid)
        elif type(data) != paddle.Tensor:
            data = data.as_subclass(paddle.Tensor)
        return data

    @classmethod
    def __paddle_function__(cls, func, types, args=(), kwargs=None):
        if func == paddle.nn.functional.grid_sample:
            raise ValueError(
                "Argument of F.grid_sample() must be a batch, not a single image"
            )
        data = paddle.Tensor.__paddle_function__(func, (paddle.Tensor,), args, kwargs)
        grid = cls._paddle_function_grid(args)
        if func in (
            paddle.split,
            paddle.Tensor.split,
            paddle.split_with_sizes,
            paddle.Tensor.split_with_sizes,
            paddle.tensor_split,
            paddle.Tensor.tensor_split,
        ):
            return tuple(cls._paddle_function_result(func, sub, grid) for sub in data)
        return cls._paddle_function_result(func, data, grid)

    def batch(self: TImage) -> ImageBatch:
        """Image batch consisting of this image only.

        Because batched operations are generally more efficient, especially when executed on the GPU,
        most ``Image`` operations are implemented by ``ImageBatch``. The single-image batch instance
        property of this ``Image`` instance is used to execute single-image operations of ``self``.
        The ``ImageBatch`` uses a view on the tensor data of this ``Image``, as well as the ``Grid``
        object reference. No copies are made.

        """
        data = self.unsqueeze(0)
        grid = self._grid
        return ImageBatch(data, grid)

    def cube(self: TImage) -> Cube:
        """Get cube which defines the normalized coordinates space of the image with respect to the world."""
        return self._grid.cube()

    def domain(self: TImage) -> Domain:
        """Get oriented bounding box in world space which defines the domain within which the image is defined."""
        return self._grid.domain()

    def same_domain_as(self, other: Image) -> bool:
        """Check if this and another image have the same cube domain."""
        return self.same_domain_as(other.grid())

    @overload
    def grid(self: TImage) -> Grid:
        """Get sampling grid."""
        ...

    @overload
    def grid(self: TImage, grid: Grid) -> Image:
        """Get new image with given sampling grid."""
        ...

    def grid(self: TImage, grid: Optional[Grid] = None) -> Union[Grid, TImage]:
        """Get sampling grid or image with given grid, respectively."""
        if grid is None:
            return self._grid
        return self._make_instance(grid=grid)

    def grid_(self: TImage, grid: Optional[Grid]) -> TImage:
        """Change image sampling grid of this image."""
        if grid is None:
            grid = Grid(shape=self.shape[1:])
        elif tuple(grid.shape) != tuple(self.shape[1:]):
            raise ValueError(
                "Image grid size does not match spatial dimensions of image tensor"
            )
        self._grid = grid
        return self

    def align_corners(self: TImage) -> bool:
        """Whether image resizing operations by default preserve corner points or grid extent."""
        return self._grid.align_corners()

    def center(self: TImage) -> paddle.Tensor:
        """Image center in world space coordinates as tensor of shape (D,)."""
        return self._grid.center()

    def origin(self: TImage) -> paddle.Tensor:
        """Image origin in world space coordinates as tensor of shape (D,)."""
        return self._grid.origin()

    def spacing(self: TImage) -> paddle.Tensor:
        """Image spacing in world units as tensor of shape (D,)."""
        return self._grid.spacing()

    def direction(self: TImage) -> paddle.Tensor:
        """Image direction cosines matrix as tensor of shape (D, D)."""
        return self._grid.direction()

    @property
    def sdim(self: TImage) -> int:
        """Number of spatial dimensions."""
        return self._grid.ndim

    @property
    def nchannels(self: TImage) -> int:
        """Number of image channels."""
        return self.shape[0]

    @classmethod
    def from_sitk(
        cls: Type[TImage],
        image: "sitk.Image",
        align_corners: bool = ALIGN_CORNERS,
        dtype: Optional[paddle.dtype] = None,
        device: Optional[Device] = None,
    ) -> TImage:
        """Create image from ``SimpleITK.Image`` instance."""
        if sitk is None:
            raise RuntimeError(f"{cls.__name__}.from_sitk() requires SimpleITK")
        data = tensor_from_image(image, dtype=dtype, device=device)
        grid = Grid.from_sitk(image, align_corners=align_corners)
        return cls(data, grid)

    def sitk(self: TImage) -> "sitk.Image":
        """Create ``SimpleITK.Image`` from this image."""
        if sitk is None:
            raise RuntimeError(f"{type(self).__name__}.sitk() requires SimpleITK")
        grid = self._grid
        origin = grid.origin().tolist()
        spacing = grid.spacing().tolist()
        direction = grid.direction().flatten().tolist()
        return image_from_tensor(
            self, origin=origin, spacing=spacing, direction=direction
        )

    @classmethod
    def read(
        cls: Type[TImage],
        path: PathStr,
        align_corners: bool = ALIGN_CORNERS,
        dtype: Optional[paddle.dtype] = None,
        device: Optional[Device] = None,
    ) -> TImage:
        """Read image data from file."""
        image = cls._read_sitk(path)
        return cls.from_sitk(
            image, align_corners=align_corners, dtype=dtype, device=device
        )

    @classmethod
    def _read_sitk(cls, path: PathStr) -> "sitk.Image":
        """Read SimpleITK.Image from file path."""
        if sitk is None:
            raise RuntimeError(f"{cls.__name__}.read() requires SimpleITK")
        return read_image(path)

    def write(self: TImage, path: PathStr, compress: bool = True) -> None:
        """Write image data to file."""
        if sitk is None:
            raise RuntimeError(f"{type(self).__name__}.write() requires SimpleITK")
        image = self.sitk()
        path = unlink_or_mkdir(path)
        sitk.WriteImage(image, str(path), compress)

    def normalize(
        self: TImage,
        mode: str = "unit",
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> TImage:
        """Normalize image intensities in [min, max]."""
        batch = self.batch()
        batch = batch.normalize(mode=mode, min=min, max=max)
        return batch[0]

    def normalize_(
        self: TImage,
        mode: str = "unit",
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> TImage:
        """Normalize image intensities in [min, max]."""
        batch = self.batch()
        batch = batch.normalize_(mode=mode, min=min, max=max)
        return batch[0]

    def rescale(
        self: TImage,
        min: Optional[Scalar] = None,
        max: Optional[Scalar] = None,
        data_min: Optional[Scalar] = None,
        data_max: Optional[Scalar] = None,
        dtype: Optional[DType] = None,
    ) -> TImage:
        """Clamp and linearly rescale image values."""
        batch = self.batch()
        batch = batch.rescale(min, max, data_min, data_max, dtype=dtype)
        return batch[0]

    def narrow(self: TImage, dim: int, start: int, length: int) -> TImage:
        """Narrow image along specified dimension."""
        batch = self.batch()
        start_29 = batch.shape[dim + 1] + start if start < 0 else start
        batch = paddle.slice(batch, [dim + 1], [start_29], [start_29 + length])
        return batch[0]

    def resize(
        self: TImage,
        size: Union[int, Array, Size],
        *args: int,
        mode: Union[Sampling, str] = Sampling.LINEAR,
        align_corners: Optional[bool] = None,
    ) -> TImage:
        """Interpolate image with specified spatial image grid size."""
        batch = self.batch()
        batch = batch.resize(size, *args, mode=mode, align_corners=align_corners)
        return batch[0]

    def resample(
        self: TImage,
        spacing: Union[float, Array, str],
        *args: float,
        mode: Union[Sampling, str] = Sampling.LINEAR,
    ) -> TImage:
        """Interpolate image with specified spacing."""
        batch = self.batch()
        batch = batch.resample(spacing, *args, mode=mode)
        return batch[0]

    def avg_pool(
        self: TImage,
        kernel_size: ScalarOrTuple[int],
        stride: Optional[ScalarOrTuple[int]] = None,
        padding: ScalarOrTuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> TImage:
        """Average pooling of image data."""
        batch = self.batch()
        batch = batch.avg_pool(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        return batch[0]

    def downsample(
        self: TImage,
        levels: int = 1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        min_size: int = 0,
        align_corners: Optional[bool] = None,
    ) -> TImage:
        """Downsample image a given number of times."""
        batch = self.batch()
        batch = batch.downsample(
            levels,
            dims=dims,
            sigma=sigma,
            mode=mode,
            min_size=min_size,
            align_corners=align_corners,
        )
        return batch[0]

    def upsample(
        self: TImage,
        levels: int = 1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> TImage:
        """Upsample image a given number of times."""
        batch = self.batch()
        batch = batch.upsample(
            levels, dims=dims, sigma=sigma, mode=mode, align_corners=align_corners
        )
        return batch[0]

    def pyramid(
        self: TImage,
        levels: int,
        start: int = 0,
        end: int = -1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        spacing: Optional[float] = None,
        min_size: int = 0,
        align_corners: Optional[bool] = None,
    ) -> Dict[int, TImage]:
        """Create Gaussian resolution pyramid."""
        batch = self.batch()
        batches = batch.pyramid(
            levels,
            start,
            end,
            dims=dims,
            sigma=sigma,
            mode=mode,
            spacing=spacing,
            min_size=min_size,
            align_corners=align_corners,
        )
        return {level: batch[0] for level, batch in batches.items()}

    def crop(
        self: TImage,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImage:
        """Crop image at boundary."""
        batch = self.batch()
        batch = batch.crop(margin=margin, num=num, mode=mode, value=value)
        return batch[0]

    def pad(
        self: TImage,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImage:
        """Pad image at boundary."""
        batch = self.batch()
        batch = batch.pad(margin=margin, num=num, mode=mode, value=value)
        return batch[0]

    def center_crop(self: TImage, size: Union[int, Array], *args: int) -> TImage:
        """Crop image to specified maximum size."""
        batch = self.batch()
        batch = batch.center_crop(size, *args)
        return batch[0]

    def center_pad(
        self: TImage,
        size: Union[int, Array],
        *args: int,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImage:
        """Pad image to specified minimum size."""
        batch = self.batch()
        batch = batch.center_pad(size, *args, mode=mode, value=value)
        return batch[0]

    def region_of_interest(self: TImage, *args, **kwargs) -> TImage:
        """Extract image region of interest."""
        batch = self.batch()
        batch = batch.region_of_interest(*args, **kwargs)
        return batch[0]

    def conv(
        self: TImage,
        kernel: Union[paddle.Tensor, Sequence[Optional[paddle.paddle.Tensor]]],
        padding: Union[PaddingMode, str, int] = None,
    ) -> TImage:
        """Filter image with a given (separable) kernel."""
        batch = self.batch()
        batch = batch.conv(kernel, padding=padding)
        return batch[0]

    @overload
    def sample(
        self: TImage,
        coords: paddle.Tensor,
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> paddle.Tensor:
        """Sample image at optionally deformed unit grid points.

        Note, to sample a set of 2D patches from a 3D volume, it may be beneficial to use ``ImageBatch.sample()``
        instead such that the output tensor shape is ``(N, C, Y, X)`` instead of ``(C, N, Y, X)`` given an
        input ``coords`` shape of ``(N, Y, X, D)``. For example, use ``image.batch().sample()``.

        Args:
            coords: Normalized coordinates of points at which to sample image as tensor of shape ``(..., D)``.
                Typical tensor shapes are: ``(Y, X, D)`` or ``(Z, Y, X, D)`` to sample an image at (deformed)
                2D or 3D grid points (cf. ``grid_sample()``), ``(N, D)`` to sample an image at a set of ``N``
                points, and ``(N, Y, X, D)`` to sample ``N`` 2D patches of size ``(X, Y)`` from a 3D volume.
            mode: Interpolation mode.
            padding: Extrapolation mode or scalar padding value.

        Returns:
            paddle.Tensor of sampled image values with shape ``(C, ...)``, where ``C`` is the number of channels
            of this image and ``...`` are the leading dimensions of ``coords`` (i.e., ``coords.shape[:-1]``).

        """
        ...

    @overload
    def sample(
        self: TImage,
        grid: Grid,
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> TImage:
        """Sample image at optionally deformed unit grid points.

        Args:
            grid: Sample this image at the points of the given sampling grid.
            mode: Interpolation mode.
            padding: Extrapolation mode or scalar padding value.

        Returns:
            Image sampled at grid points.

        """
        ...

    def sample(
        self: TImage,
        arg: Union[Grid, paddle.Tensor],
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> Union[paddle.Tensor, TImage]:
        """Sample image at points given as normalized coordinates.

        Note, to sample a set of 2D patches from a 3D volume, it may be beneficial to use ``ImageBatch.sample()``
        instead such that the output tensor shape is ``(N, C, Y, X)`` instead of ``(C, N, Y, X)`` given an
        input ``arg`` shape of ``(N, Y, X, D)``. For example, use ``image.batch().sample()``.

        Args:
            arg: Sampling grid defining points at which to sample image data, or normalized coordinates of
                points at which to sample image as tensor of shape ``(..., D)``. Typical tensor shapes are:
                ``(Y, X, D)`` or ``(Z, Y, X, D)`` to sample an image at (deformed) 2D or 3D grid points
                (cf. ``grid_sample()``), ``(N, D)`` to sample an image at a set of ``N`` points, and
                ``(N, Y, X, D)`` to sample ``N`` 2D patches of size ``(X, Y)`` from a 3D volume.
            mode: Interpolation mode.
            padding: Extrapolation mode or scalar padding value.

        Returns:
            If ``arg`` is of type ``Grid``, an ``Image`` with the sampled values and given sampling grid is returend.
            When ``arg == self.grid()``, a reference to ``self`` is returned. Otherwise, a ``paddle.Tensor`` of sampled image
            values with shape ``(C, ...)`` is returned, where ``C`` is the number of channels of this image and ``...``
            are the leading dimensions of ``grid`` (i.e., ``grid.shape[:-1]``).

        """
        batch = self.batch()
        if isinstance(arg, Grid):
            if arg == self.grid():
                return self
            batch = batch.sample(arg, mode=mode, padding=padding)
            assert isinstance(batch, ImageBatch)
            return batch[0]
        if isinstance(arg, paddle.Tensor):
            grid = arg.unsqueeze(0)
            data = batch.sample(grid, mode=mode, padding=padding)
            assert type(data) is paddle.Tensor
            assert tuple(data.shape)[0] == 1
            data = data.squeeze(axis=0)
            return data
        raise TypeError(
            f"{type(self).__name__}.sample() 'arg' must be Grid or paddle.Tensor"
        )

    def __repr__(self) -> str:
        return type(self).__name__ + f"(data={self.tensor()!r}, grid={self.grid()!r})"

    def __str__(self) -> str:
        return type(self).__name__ + f"(data={self.tensor()!s}, grid={self.grid()!s})"
