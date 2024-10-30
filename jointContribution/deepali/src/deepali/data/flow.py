r"""Flow vector fields."""

from __future__ import annotations  # noqa

from typing import Any
from typing import Optional
from typing import Sequence
from typing import Type
from typing import TypeVar
from typing import Union
from typing import overload

import paddle
from deepali.core import flow as U
from deepali.core.enum import PaddingMode
from deepali.core.enum import Sampling
from deepali.core.grid import ALIGN_CORNERS
from deepali.core.grid import Axes
from deepali.core.grid import Grid
from deepali.core.grid import grid_transform_vectors
from deepali.core.tensor import move_dim
from deepali.core.typing import Array
from deepali.core.typing import Device
from deepali.core.typing import DType
from deepali.core.typing import EllipsisType
from deepali.core.typing import PathStr
from deepali.core.typing import Scalar
from deepali.core.typing import ScalarOrTuple
from paddle import Tensor

from .image import Image
from .image import ImageBatch

TFlowField = TypeVar("TFlowField", bound="FlowField")
TFlowFields = TypeVar("TFlowFields", bound="FlowFields")


__all__ = "FlowField", "FlowFields"


class FlowFields(ImageBatch):
    r"""Batch of flow fields."""

    def __init__(
        self: TFlowFields,
        data: Union[Array, ImageBatch],
        grid: Optional[Union[Grid, Sequence[Grid]]] = None,
        axes: Optional[Axes] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize flow fields.

        Args:
            data: Batch data tensor of shape (N, D, ...X), where N is the batch size, and D
                must be equal the number of spatial dimensions. The order of the image channels
                must be such that vector components are in the order ``(x, ...)``.
            grid: Flow field sampling grids. If not otherwise specified, this attribute
                defines the fixed target image domain on which to resample a moving source image.
            axes: Axes with respect to which vectors are defined. By default, it is assumed that
                vectors are with respect to the unit ``grid`` cube in ``[-1, 1]^D``, where D are the
                number of spatial dimensions. If ``grid.align_corners() == False``, the extrema
                ``(-1, 1)`` refer to the boundary of the vector field ``grid``. Otherwise, the
                extrema coincide with the corner points of the sampling grid.
            dtype: Data type of the image data. A copy of the data is only made when the desired ``dtype``
                is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store image data. A copy of the data is only made when the data
                has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned image tensor.
            pin_memory: If set, returned image tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        # DataTensor.__new__() creates the tensor subclass given arguments:
        # data, dtype, device, requires_grad, pin_memory
        if grid is None and isinstance(data, ImageBatch):
            grid = data.grid()
            data = data.tensor()
        super().__init__(
            data,
            grid,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
        if self.shape[1] != self.sdim:
            raise ValueError(
                f"{type(self).__name__}() 'data' nchannels={self.shape[1]} must be equal spatial ndim={self.sdim}"
            )
        if axes is None:
            axes = Axes.from_grid(self._grid[0])
        else:
            axes = Axes.from_arg(axes)
        self._axes = axes

    def _make_instance(
        self: TFlowFields,
        data: paddle.Tensor,
        grid: Optional[Sequence[Grid]] = None,
        axes: Optional[Axes] = None,
        **kwargs,
    ) -> Union[TFlowFields, ImageBatch]:
        r"""Create a new instance while preserving subclass meta-data."""
        if tuple(data.shape)[1] != data.ndim - 2:
            return ImageBatch(data, grid)
        kwargs["axes"] = axes or self._axes
        return super()._make_instance(data, grid, **kwargs)

    def _make_subitem(self, data: paddle.Tensor, grid: Grid) -> Union[FlowField, Image]:
        r"""Create FlowField in __getitem__. Can be overridden by subclasses to return a subtype."""
        if tuple(data.shape)[0] == data.ndim - 1:
            return FlowField(data, grid, self._axes)
        return super()._make_subitem(data, grid)

    @staticmethod
    def _paddle_function_axes(args) -> Optional[Axes]:
        r"""Get flow field Axes from args passed to __paddle_function__."""
        if not args:
            return None
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        axes: Sequence[Axes]
        axes = [ax for ax in (getattr(arg, "_axes", None) for arg in args) if ax is not None]
        if not axes:
            return None
        if any(ax != axes[0] for ax in axes[1:]):
            raise ValueError(
                "Cannot apply __paddle_function__ to flow fields with mismatching axes"
            )
        return axes[0]

    @classmethod
    def _paddle_function_result(
        cls, func, data, grid: Optional[Sequence[Grid]], axes: Optional[Axes]
    ) -> Any:
        if not isinstance(data, paddle.Tensor):
            return data
        if (
            grid
            and axes is not None
            and data.ndim == grid[0].ndim + 2
            and tuple(data.shape)[1] == grid[0].ndim
            and tuple(data.shape)[2:] == tuple(grid[0].shape)
            or grid is not None
            and not grid
            and data.ndim >= 4
            and tuple(data.shape)[0] == 0
        ):
            if func in (paddle.clone, paddle.Tensor.clone):
                grid = [g.clone() for g in grid]
            if isinstance(data, cls):
                data._grid = grid
                data._axes = axes
            else:
                data = cls(data, grid, axes)
        else:
            data = ImageBatch._paddle_function_result(func, data, grid)
        return data

    @classmethod
    def __paddle_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        data = paddle.Tensor.__paddle_function__(func, (Tensor,), args, kwargs)
        grid = cls._paddle_function_grid(func, args, kwargs)
        axes = cls._paddle_function_axes(args)
        if func in (
            paddle.split,
            paddle.Tensor.split,
            paddle.split_with_sizes,
            paddle.Tensor.split_with_sizes,
            paddle.tensor_split,
            paddle.Tensor.tensor_split,
        ):
            return tuple(cls._paddle_function_result(func, res, grid, axes) for res in data)
        return cls._paddle_function_result(func, data, grid, axes)

    @overload
    def __getitem__(self: TFlowFields, index: int) -> FlowField:
        ...

    @overload
    def __getitem__(self: TFlowFields, index: EllipsisType) -> TFlowFields:
        ...

    @overload
    def __getitem__(self: TFlowFields, index: Union[list, slice, paddle.Tensor]) -> TFlowFields:
        ...

    def __getitem__(
        self: TFlowFields,
        index: Union[EllipsisType, int, slice, Sequence[Union[EllipsisType, int, slice]]],
    ) -> Union[FlowField, Image, TFlowFields, ImageBatch, paddle.Tensor]:
        r"""Get flow field at specified batch index, get a sub-batch, or extract region of interest tensor."""
        return super().__getitem__(index)

    @overload
    def axes(self: TFlowFields) -> Axes:
        r"""Get axes with respect to which flow vectors are defined."""
        ...

    @overload
    def axes(self: TFlowFields, axes: Axes) -> TFlowFields:
        r"""Get new batch of flow fields with flow vectors defined with respect to specified axes."""
        ...

    def axes(self: TFlowFields, axes: Optional[Axes] = None) -> Union[Axes, TFlowFields]:
        r"""Rescale and reorient vectors."""
        if axes is None:
            return self._axes
        data = self.tensor()
        data = move_dim(data, 1, -1)
        data = tuple(
            grid.transform_vectors(data[i : i + 1], axes=self._axes, to_axes=axes)
            for i, grid in enumerate(self._grid)
        )
        data = paddle.concat(x=data, axis=0)
        data = move_dim(data, -1, 1)
        return self._make_instance(data, self._grid, axes)

    def curl(
        self: TFlowFields,
        mode: Optional[str] = None,
        sigma: Optional[float] = None,
        spacing: Optional[Union[Scalar, Array]] = None,
        stride: Optional[ScalarOrTuple[int]] = None,
    ) -> ImageBatch:
        if self.ndim not in (2, 3):
            raise RuntimeError(f"Cannot compute curl of {self.ndim}-dimensional flow field")
        if spacing is None:
            if self.axes() is Axes.GRID:
                spacing = 1
            elif self.axes() is Axes.WORLD:
                spacing = self.spacing()
            elif self.axes() is Axes.CUBE:
                spacing = tuple(2 / n for n in tuple(self.grid().shape))
            else:
                spacing = tuple(2 / (n - 1) for n in tuple(self.grid().shape))
        data = U.curl(self.tensor(), mode=mode, sigma=sigma, spacing=spacing, stride=stride)
        return ImageBatch(data, self._grid)

    def exp(
        self: TFlowFields,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    ) -> TFlowFields:
        r"""Group exponential maps of flow fields computed using scaling and squaring."""
        axes = self._axes
        align_corners = axes is Axes.CUBE_CORNERS
        flow = self.axes(Axes.CUBE_CORNERS if align_corners else Axes.CUBE)
        data = self.tensor()
        data = U.expv(
            data,
            scale=scale,
            steps=steps,
            sampling=sampling,
            padding=padding,
            align_corners=align_corners,
        )
        flow = self._make_instance(data, flow._grid, flow._axes)
        flow = flow.axes(axes)
        return flow

    def sample(
        self: TFlowFields,
        arg: Union[Grid, Sequence[Grid], paddle.Tensor],
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> Union[TFlowFields, paddle.Tensor]:
        r"""Sample flow fields at optionally deformed unit grid points.

        Args:
            arg: Either a single grid which defines the sampling points for all images in the batch,
                a different grid for each image in the batch, or a tensor of normalized coordinates
                with shape ``(N, ..., D)`` or ``(1, ..., D)``. In the latter case, note that the
                shape ``...`` need not correspond to a (deformed) grid as required by ``grid_sample()``.
                It can be an arbitrary shape, e.g., ``M`` to sample at ``M`` given points.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            If ``arg`` is of type ``Grid`` or ``Sequence[Grid]``, a ``FlowFields`` batch is returned.
            When these grids match the grids of this batch of flow fields, ``self`` is returned.
            Otherwise, a ``Tensor`` of shape (N, C, ...) of sampled flow values is returned.
            Note that when ``arg`` is of type ``Grid`` or ``Sequence[Grid]``, flow vectors that are
            not expressed with respect to the world coordinate system will be implicitly converted to
            flow vectors with respect to the new sampling grids. If this is not desired, use a ``Tensor``
            type with sampling coordinates instead of ``Grid`` instances.

        """
        flow = super().sample(arg, mode=mode, padding=padding)
        if isinstance(flow, FlowFields):
            axes = flow.axes()
            if axes != Axes.WORLD:
                data = flow.tensor()
                data = U.move_dim(data, 1, -1)
                data = paddle.concat(
                    x=[
                        grid_transform_vectors(v, grid, axes, to_grid, axes).unsqueeze_(0)
                        for v, grid, to_grid in zip(data, self._grid, flow.grids())
                    ],
                    axis=0,
                )
                data = U.move_dim(data, -1, 1)
                flow = self._make_instance(data, flow.grids())
        return flow

    def warp_image(
        self: TFlowFields,
        image: Union[Image, ImageBatch],
        sampling: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str]] = None,
    ) -> ImageBatch:
        r"""Deform given image (batch) using this batch of vector fields.

        Args:
            image: Single image or image batch. If a single ``Image`` is given, it is deformed by
                all the displacement fields in this batch. If an ``ImageBatch`` is given, the number
                of images in the batch must match the number of displacement fields in this batch.
            sampling: Interpolation mode for sampling values from ``image`` at deformed grid points.
            padding: Extrapolation mode for sampling values outside ``image`` domain.

        Returns:
            Batch of input images deformed by the vector fields of this batch.

        """
        if isinstance(image, Image):
            image = image.batch()
        align_corners = self._axes is Axes.CUBE_CORNERS
        grid = (g.coords(align_corners=align_corners, device=self.device) for g in self._grid)
        grid = paddle.concat(x=tuple(g.unsqueeze(axis=0) for g in grid), axis=0)
        flow = self.axes(Axes.from_align_corners(align_corners))
        flow = flow.tensor()
        flow = move_dim(flow, 1, -1)
        data = image.tensor()
        data = U.warp_image(
            data, grid, flow=flow, mode=sampling, padding=padding, align_corners=align_corners
        )
        return image._make_instance(data, self._grid)

    def __repr__(self) -> str:
        return (
            type(self).__name__
            + f"(data={self.tensor()!r}, grids={self.grids()!r}, axes={self.axes()!r})"
        )

    def __str__(self) -> str:
        return (
            type(self).__name__
            + f"(data={self.tensor()!s}, grids={self.grids()!s}, axes={self.axes()!r})"
        )


class FlowField(Image):
    r"""Flow field image.

    A (dense) flow field is a vector image where the number of channels equals the number of spatial dimensions.
    The starting points of the vectors are defined on a regular oriented sampling grid positioned in world space.
    Orientation and scale of the vectors are defined with respect to a specified regular grid domain, which either
    coincides with the sampling grid, the world coordinate system, or the unit cube with side length 2 centered at
    the center of the sampling grid with axes parallel to the sampling grid. This unit cube domain is used by the
    ``paddle.nn.functional.grid_sample()`` and ``paddle.nn.functional.interpolate()`` functions.

    When a flow field is convert to a ``SimpleITK.Image``, the vectors are by default reoriented and rescaled such
    that these are with respect to the world coordinate system, a format common to ITK functions and other toolkits.

    """

    def __init__(
        self: TFlowField,
        data: Array,
        grid: Optional[Grid] = None,
        axes: Optional[Axes] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize flow field.

        Args:
            data: Flow field data tensor of shape (C, ...X), where C must be equal the number of spatial dimensions.
                The order of the image channels must be such that vector components are in the order X, Y,...
            grid: Flow field sampling grid. If not otherwise specified, this attribute often also defines the fixed
                target image domain on which to resample a moving source image.
            axes: Axes with respect to which vectors are defined. By default, it is assumed that vectors are with
                respect to the unit ``grid`` cube in ``[-1, 1]^D``, where D are the number of spatial dimensions.
                If ``None`` and ``grid.align_corners() == False``, the extrema ``(-1, 1)`` refer to the boundary of
                the vector field ``grid``, and coincide with the grid corner points otherwise.
            dtype: Data type of the image data. A copy of the data is only made when the desired ``dtype``
                is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store image data. A copy of the data is only made when the data
                has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned image tensor.
            pin_memory: If set, returned image tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        # DataTensor.__new__() creates the tensor subclass given arguments:
        # data, dtype, device, requires_grad, pin_memory
        super().__init__(
            data,
            grid,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
        if self.nchannels != self._grid.ndim:
            raise ValueError(
                f"{type(self).__name__} nchannels={self.nchannels} must be equal grid.ndim={self._grid.ndim}"
            )
        if axes is None:
            axes = Axes.from_grid(self._grid)
        else:
            axes = Axes.from_arg(axes)
        self._axes = axes

    def _make_instance(
        self: TFlowField,
        data: paddle.Tensor,
        grid: Optional[Grid] = None,
        axes: Optional[Axes] = None,
        **kwargs,
    ) -> TFlowField:
        r"""Create a new instance while preserving subclass meta-data."""
        kwargs["axes"] = axes or self._axes
        return super()._make_instance(data, grid, **kwargs)

    @staticmethod
    def _paddle_function_axes(args) -> Optional[Axes]:
        r"""Get flow field Axes from args passed to __paddle_function__."""
        if not args:
            return None
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        axes: Sequence[Axes]
        axes = [ax for ax in (getattr(arg, "_axes", None) for arg in args) if ax is not None]
        if not axes:
            return None
        if any(ax != axes[0] for ax in axes[1:]):
            raise ValueError(
                "Cannot apply __paddle_function__ to flow fields with mismatching axes"
            )
        return axes[0]

    @classmethod
    def _paddle_function_result(cls, func, data, grid: Optional[Grid], axes: Optional[Axes]) -> Any:
        if not isinstance(data, paddle.Tensor):
            return data
        if (
            grid is not None
            and axes is not None
            and data.ndim == grid.ndim + 1
            and tuple(data.shape)[0] == grid.ndim
            and tuple(data.shape)[1:] == tuple(grid.shape)
        ):
            if func in (paddle.clone, paddle.Tensor.clone):
                grid = grid.clone()
            if isinstance(data, cls):
                data._grid = grid
                data._axes = axes
            else:
                data = cls(data, grid, axes)
        else:
            data = Image._paddle_function_result(func, data, grid)
        return data

    @classmethod
    def __paddle_function__(cls, func, types, args=(), kwargs=None):
        if func == paddle.nn.functional.grid_sample:
            raise ValueError("Argument of F.grid_sample() must be a batch, not a single image")
        if kwargs is None:
            kwargs = {}
        data = paddle.Tensor.__paddle_function__(func, (Tensor,), args, kwargs)
        grid = cls._paddle_function_grid(args)
        axes = cls._paddle_function_axes(args)
        if func in (
            paddle.split,
            paddle.Tensor.split,
            # paddle.split_with_sizes, paddle.Tensor.split_with_sizes,
            paddle.tensor_split,
            paddle.Tensor.tensor_split,
        ):
            return tuple(cls._paddle_function_result(func, res, grid, axes) for res in data)
        return cls._paddle_function_result(func, data, grid, axes)

    @classmethod
    def from_image(cls: Type[TFlowField], image: Image, axes: Optional[Axes] = None) -> TFlowField:
        r"""Create flow field from image instance."""
        return cls(image, image._grid, axes)

    def batch(self: TFlowField) -> FlowFields:
        r"""Batch of flow fields containing only this flow field."""
        data = self.unsqueeze(0)
        return FlowFields(data, self._grid, self._axes)

    @overload
    def axes(self: TFlowField) -> Axes:
        r"""Get axes with respect to which flow vectors are defined."""
        ...

    @overload
    def axes(self: TFlowField, axes: Axes) -> TFlowField:
        r"""Get new flow field with flow vectors defined with respect to specified axes."""
        ...

    def axes(self: TFlowField, axes: Optional[Axes] = None) -> Union[Axes, TFlowField]:
        r"""Rescale and reorient vectors with respect to specified axes."""
        if axes is None:
            return self._axes
        batch = self.batch()
        batch = batch.axes(axes)
        return batch[0]

    @classmethod
    def from_sitk(
        cls: Type[TFlowField],
        image: "sitk.Image",
        axes: Optional[Axes] = None,
        align_corners: bool = ALIGN_CORNERS,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ) -> TFlowField:
        r"""Create vector field from ``SimpleITK.Image``."""
        image = Image.from_sitk(image, align_corners=align_corners, dtype=dtype, device=device)
        return cls.from_image(image, axes=axes or Axes.WORLD)

    def sitk(self: TFlowField, axes: Optional[Axes] = None) -> "sitk.Image":
        r"""Create ``SimpleITK.Image`` from this vector field."""
        disp: FlowField = self.detach()
        disp = disp.axes(axes or Axes.WORLD)
        return Image.sitk(disp)

    @classmethod
    def read(
        cls: Type[TFlowField],
        path: PathStr,
        axes: Optional[Axes] = None,
        align_corners: bool = ALIGN_CORNERS,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ) -> TFlowField:
        r"""Read image data from file."""
        image = Image.read(path, align_corners=align_corners, dtype=dtype, device=device)
        return cls.from_image(image, axes=axes or Axes.WORLD)

    def write(self, path: PathStr, axes: Optional[Axes] = None, compress: bool = True) -> None:
        r"""Write flow field data to file."""
        disp: FlowField = self.detach()
        disp = disp.axes(axes or Axes.WORLD)
        Image.write(disp, path, compress=compress)

    def curl(
        self: TFlowField,
        mode: Optional[str] = None,
        sigma: Optional[float] = None,
        spacing: Optional[Union[Scalar, Array]] = None,
        stride: Optional[ScalarOrTuple[int]] = None,
    ) -> Image:
        r"""Compute curl of vector field."""
        batch = self.batch()
        rotvec = batch.curl(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
        return rotvec[0]

    def exp(
        self: TFlowField,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    ) -> TFlowField:
        r"""Group exponential map of vector field computed using scaling and squaring."""
        batch = self.batch()
        batch = batch.exp(scale=scale, steps=steps, sampling=sampling, padding=padding)
        return batch[0]

    @overload
    def warp_image(
        self: TFlowField,
        image: Image,
        sampling: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str]] = None,
    ) -> Image:
        r"""Deform given image using this displacement field."""
        ...

    @overload
    def warp_image(
        self: TFlowField,
        image: ImageBatch,
        sampling: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str]] = None,
    ) -> ImageBatch:
        r"""Deform images in batch using this displacement field."""
        ...

    def warp_image(
        self: TFlowField,
        image: Union[Image, ImageBatch],
        sampling: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str]] = None,
    ) -> Union[Image, ImageBatch]:
        r"""Deform given image (batch) using this displacement field.

        Args:
            image: Single image or batch of images.
            kwargs: Keyword arguments to pass on to ``ImageBatch.warp()``.

        Returns:
            If ``image`` is an ``ImageBatch``, each image in the batch is deformed by this flow field
            and a batch of deformed images is returned. Otherwise, a single deformed image is returned.

        """
        batch = self.batch()
        result = batch.warp_image(image, sampling=sampling, padding=padding)
        if isinstance(image, Image) and len(result) == 1:
            return result[0]
        return result

    def __repr__(self) -> str:
        return (
            type(self).__name__
            + f"(data={self.tensor()!r}, grid={self.grid()!r}, axes={self.axes()!r})"
        )

    def __str__(self) -> str:
        return (
            type(self).__name__
            + f"(data={self.tensor()!s}, grid={self.grid()!s}, axes={self.axes()!r})"
        )
