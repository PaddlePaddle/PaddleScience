r"""Modules for sampling of image data, e.g., after spatial transformation."""

from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast
from typing import overload

import paddle
from deepali.core import functional as U
from deepali.core.enum import PaddingMode
from deepali.core.enum import Sampling
from deepali.core.grid import Axes
from deepali.core.grid import Grid
from deepali.core.grid import grid_points_transform
from deepali.core.linalg import homogeneous_matmul
from deepali.core.linalg import homogeneous_transform
from deepali.core.typing import Scalar
from paddle import Tensor


class SampleImage(paddle.nn.Layer):
    r"""Sample images at grid points."""

    def __init__(
        self,
        target: Grid,
        source: Optional[Grid] = None,
        axes: Optional[Union[Axes, str]] = None,
        sampling: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
        align_centers: bool = False,
    ):
        r"""Initialize base class.

        Args:
            target: Grid on which to sample transformed images.
            source: Grid on which input source images are sampled.
            axes: Axes with respect to which grid coordinates are defined.
                If ``None``, use cube axes corresponding to ``target.align_corners()`` setting.
            sampling: Image interpolation mode.
            padding: Image extrapolation mode.
            align_centers: Whether to implicitly align the ``target`` and ``source`` centers.
                If ``True``, only the affine component of the target to source transformation
                is applied. If ``False``, also the translation of grid center points is considered.

        """
        super().__init__()
        self._target = target
        self._source = source or target
        if axes is None:
            axes = Axes.from_grid(target)
        self._axes = Axes(axes)
        self._sampling = Sampling.from_arg(sampling)
        if padding is None or isinstance(padding, (PaddingMode, str)):
            self._padding = PaddingMode.from_arg(padding)
        else:
            self._padding = float(padding)
        self._align_centers = bool(align_centers)
        # Precompute target points to source cube transformation **AFTER** attributes are set!
        self.register_buffer(name="matrix", tensor=self._matrix())

    def axes(self) -> Axes:
        r"""Axes with respect to which target grid points and transformations thereof are defined."""
        return self._axes

    def target_grid(self) -> Grid:
        r"""Target sampling grid."""
        return self._target

    def source_grid(self) -> Grid:
        r"""Source sampling grid."""
        return self._source

    def sampling(self) -> Sampling:
        r"""Image sampling mode."""
        return self._sampling

    def padding(self) -> Union[PaddingMode, Scalar]:
        r"""Image padding mode or value, respectively."""
        return self._padding

    def padding_mode(self) -> PaddingMode:
        r"""Image padding mode."""
        return self._padding if isinstance(self._padding, PaddingMode) else PaddingMode.CONSTANT

    def padding_value(self) -> float:
        r"""Image padding value if mode is "constant"."""
        return 0.0 if isinstance(self._padding, PaddingMode) else float(self._padding)

    def align_centers(self) -> bool:
        r"""Whether grid center points are implicitly aligned."""
        return self._align_centers

    def align_corners(self) -> bool:
        r"""Whether to sample images using ``align_corners=False`` or ``align_corners=True``."""
        return self._target.align_corners()

    def _matrix(self) -> paddle.Tensor:
        r"""Homogeneous coordinate transformation from target grid points to source grid cube."""
        align_corners = self.align_corners()
        to_axes = Axes.from_align_corners(align_corners)
        matrix = grid_points_transform(self._target, self._axes, self._source, to_axes)
        if self._align_centers:
            offset = self._target.world_to_cube(self._source.center(), align_corners=align_corners)
            matrix = homogeneous_matmul(matrix, offset)
        return matrix.unsqueeze(axis=0)

    def _transform_target_to_source(self, grid: paddle.Tensor) -> paddle.Tensor:
        r"""Transform target grid points to source cube."""
        matrix = cast(Tensor, self.matrix)
        return homogeneous_transform(matrix, grid)

    def _sample_source_image(
        self,
        grid: paddle.Tensor,
        input: Optional[Union[paddle.Tensor, Mapping[str, paddle.Tensor]]] = None,
        data: Optional[paddle.Tensor] = None,
        mask: Optional[paddle.Tensor] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor], Dict[str, paddle.Tensor]]:
        r"""Sample images at specified source grid points."""
        source = {}
        output = {}
        shape = None
        align_corners = self.align_corners()
        if not isinstance(grid, paddle.Tensor):
            raise TypeError(f"{type(self).__name__}() 'grid' must be Tensor")
        if isinstance(input, dict):
            if data is not None or mask is not None:
                raise ValueError(
                    f"{type(self).__name__}() 'input' dict and 'data'/'mask' are mutually exclusive"
                )
            for data in input.values():
                if not isinstance(data, paddle.Tensor):
                    raise TypeError(f"{type(self).__name__}() 'input' dict values must be Tensor")
            source: Dict[str, paddle.Tensor] = {
                name: data for name, data in input.items() if name != "mask"
            }
            shape = tuple(next(iter(source.values())).shape) if source else None
            mask = input.get("mask")
        elif isinstance(input, paddle.Tensor):
            if data is not None:
                raise ValueError(
                    f"{type(self).__name__}() 'input' and 'data' are mutually exclusive"
                )
            source = {"data": input}
            shape = tuple(input.shape)
        elif input is None:
            if data is None and mask is None:
                raise ValueError(
                    f"{type(self).__name__} 'input', 'data', and/or 'mask' is required"
                )
        else:
            raise TypeError(
                f"{type(self).__name__}() 'input' must be Tensor or Mapping[str, Tensor]"
            )
        for name, data in source.items():
            is_unbatched = data.ndim == tuple(grid.shape)[-1] + 1
            if is_unbatched:
                data = data.unsqueeze(axis=0)
            data = U.grid_sample(
                data, grid, mode=self._sampling, padding=self._padding, align_corners=align_corners
            )
            if is_unbatched:
                data = data.squeeze(axis=0)
            output[name] = data
        if mask is not None:
            if not isinstance(mask, paddle.Tensor):
                raise TypeError(f"{type(self).__name__}() 'mask' must be Tensor")
            if shape is not None:
                if mask.ndim != len(shape):
                    raise ValueError(
                        f"{type(self).__name__}() 'mask' must have same ndim as 'input' data"
                    )
                if tuple(mask.shape)[0] != shape[0]:
                    raise ValueError(
                        f"{type(self).__name__}() 'mask' must have same batch size as 'input' data"
                    )
                if tuple(mask.shape)[2:] != shape[2:]:
                    raise ValueError(
                        f"{type(self).__name__}() 'mask' must have same spatial shape as 'input' data"
                    )
            temp = U.grid_sample_mask(mask, grid, align_corners=align_corners)
            output["mask"] = temp > 0.9
        if isinstance(input, dict):
            return output
        if data is None:
            return output["mask"]
        if mask is None:
            return output["data"]
        return output["data"], output["mask"]

    @overload
    def forward(
        self, grid: paddle.Tensor, input: paddle.Tensor, data=None, mask=None
    ) -> paddle.Tensor:
        r"""Sample batch of images at spatially transformed target grid points."""
        ...

    @overload
    def forward(
        self, grid: paddle.Tensor, input: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        r"""Sample batch of optionally masked images at spatially transformed target grid points."""
        ...

    def forward(
        self,
        grid: paddle.Tensor,
        input: Optional[Union[paddle.Tensor, Dict[str, paddle.Tensor]]] = None,
        data: Optional[paddle.Tensor] = None,
        mask: Optional[paddle.Tensor] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor], Dict[str, paddle.Tensor]]:
        r"""Sample images at target grid points after mapping these to the source grid cube."""
        if grid.ndim == tuple(grid.shape)[-1] + 1:
            grid = grid.unsqueeze(axis=0)
        grid = self._transform_target_to_source(grid)
        return self._sample_source_image(grid, input=input, data=data, mask=mask)

    def extra_repr(self) -> str:
        return (
            f"target={repr(self._target)}"
            + f", source={repr(self._source)}"
            + f", axes={repr(self._axes.value)}"
            + f", sampling={repr(self._sampling.value)}"
            + f", padding={repr(self._padding.value if isinstance(self._padding, PaddingMode) else self._padding)}"
            + f", align_centers={repr(self._align_centers)}"
        )


class TransformImage(SampleImage):
    r"""Sample images at transformed target grid points.

    This module can be used for both linear and non-rigid transformations, where the shape of the
    input ``transform`` tensor determines the type of transformation. See also ``transform_grid()``.

    """

    def __init__(
        self,
        target: Grid,
        source: Optional[Grid] = None,
        axes: Optional[Union[Axes, str]] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str, Scalar] = PaddingMode.BORDER,
        align_centers: bool = False,
    ):
        r"""Initialize module.

        Args:
            target: Grid on which to sample transformed images.
            source: Grid on which input source images are sampled.
            axes: Axes with respect to which transformations are defined.
                Use ``Axes.from_grid(target)`` if ``None``.
            sampling: Image interpolation mode.
            padding: Image extrapolation mode.
            align_centers: Whether to implicitly align the ``target`` and ``source`` centers.
                If ``True``, only the affine component of the target to source transformation
                is applied. If ``False``, also the translation of grid center points is considered.

        """
        super().__init__(
            target,
            source,
            axes=axes,
            sampling=sampling,
            padding=padding,
            align_centers=align_centers,
        )
        self.register_buffer(name="grid", tensor=self._grid(), persistable=False)

    def _grid(self) -> paddle.Tensor:
        r"""Target grid points before spatial transformation."""
        return self._target.points(self._axes).unsqueeze(axis=0)

    def forward(
        self,
        transform: Optional[paddle.Tensor],
        input: Optional[Union[paddle.Tensor, Dict[str, paddle.Tensor]]] = None,
        data: Optional[paddle.Tensor] = None,
        mask: Optional[paddle.Tensor] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor], Dict[str, paddle.Tensor]]:
        r"""Sample images at transformed target grid points after mapping these to the source grid cube."""
        grid = cast(Tensor, self.grid)
        if isinstance(transform, paddle.Tensor):
            if transform.ndim == tuple(grid.shape)[-1] + 1:
                transform = transform.unsqueeze(axis=0)
            grid = U.transform_grid(transform, grid, align_corners=self.align_corners())
        elif transform is not None:
            raise TypeError("TransformImage() 'transform' must be Tensor")
        grid = self._transform_target_to_source(grid)
        return self._sample_source_image(grid, input=input, data=data, mask=mask)


class AlignImage(SampleImage):
    r"""Sample images at linearly transformed target grid points.

    Instead of applying two separate linear transformations to the target grid points, this module first composes
    the two linear transformations and then applies the composite transformation to the target grid points.

    """

    def __init__(
        self,
        target: Grid,
        source: Optional[Grid] = None,
        axes: Optional[Union[Axes, str]] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str, Scalar] = PaddingMode.BORDER,
        align_centers: bool = False,
    ):
        r"""Initialize module.

        Args:
            target: Grid on which to sample transformed images.
            source: Grid on which input source images are sampled.
            axes: Axes with respect to which transformations are defined.
                Use ``Axes.from_grid(target)`` if ``None``.
            sampling: Image interpolation mode.
            padding: Image extrapolation mode.
            align_centers: Whether to implicitly align the ``target`` and ``source`` centers.
                If ``True``, only the affine component of the target to source transformation
                is applied. If ``False``, also the translation of grid center points is considered.

        """
        super().__init__(
            target,
            source,
            axes=axes,
            sampling=sampling,
            padding=padding,
            align_centers=align_centers,
        )
        self.register_buffer(name="grid", tensor=self._grid(), persistable=False)

    def _grid(self) -> paddle.Tensor:
        r"""Target grid points before spatial transformation."""
        return self._target.points(self._axes).unsqueeze(axis=0)

    def forward(
        self,
        transform: Optional[paddle.Tensor],
        input: Optional[Union[paddle.Tensor, Dict[str, paddle.Tensor]]] = None,
        data: Optional[paddle.Tensor] = None,
        mask: Optional[paddle.Tensor] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor], Dict[str, paddle.Tensor]]:
        r"""Sample batch of optionally masked images at linearly transformed target grid points."""
        composite_transform = cast(Tensor, self.matrix)
        if transform is not None:
            if not isinstance(transform, paddle.Tensor):
                raise TypeError("AlignImage() 'transform' must be Tensor")
            if transform.ndim != 3:
                raise ValueError("AlignImage() 'transform' must be 3-dimensional tensor")
            composite_transform = homogeneous_matmul(composite_transform, transform)
        grid = cast(Tensor, self.grid)
        grid = homogeneous_transform(composite_transform, grid)
        return self._sample_source_image(grid, input=input, data=data, mask=mask)
