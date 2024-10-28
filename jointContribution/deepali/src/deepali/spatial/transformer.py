r"""Modules which apply a spatial transformation to a given input data tensor.

A spatial transformer applies a :class:`.SpatialTransform`, which takes as input point coordinates
and maps these to new spatial locations, to a given input data tensor.

"""
from __future__ import annotations  # noqa

from copy import copy as shallow_copy
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import overload

import paddle
from deepali.core.enum import PaddingMode
from deepali.core.enum import Sampling
from deepali.core.grid import Axes
from deepali.core.grid import Grid
from deepali.core.typing import Scalar
from deepali.modules import SampleImage
from paddle import Tensor

from .base import SpatialTransform

TSpatialTransformer = TypeVar("TSpatialTransformer", bound="SpatialTransformer")


class SpatialTransformer(paddle.nn.Layer):
    r"""Spatially transform input data.

    A :class:`.SpatialTransformer` applies a :class:`.SpatialTransform` to a given input. How the spatial
    transformation is applied to produce a transformed output is determined by the type of spatial transformer.

    The forward method of a spatial transformer invokes the spatial transform as a functor such that any registered
    forward pre- and post-hooks are executed as part of the spatial transform evaluation. This includes in particular
    the :meth:`.SpatialTransform.update` function if it is registered as a forward pre-hook. Note that this hook is by
    default installed during initialization of a spatial transform. When the update of spatial transform parameters,
    which may be inferred by a neural network, is done explicitly by the application, use :meth:`.SpatialTransform.remove_update_hook`
    to remove this forward pre-hook before subsequent evaluations of the spatial transform. When doing so, ensure to
    update the parameters when necessary using either :meth:`.SpatialTransformer.update` or :meth:`.SpatialTransform.update`.

    """

    def __init__(self, transform: SpatialTransform) -> None:
        r"""Initialize spatial transformer.

        Args:
            transform: Spatial coordinate transformation.

        """
        if not isinstance(transform, SpatialTransform):
            raise TypeError(
                f"{type(self).__name__}() requires 'transform' of type SpatialTransform"
            )
        super().__init__()
        self._transform_new = transform

    @property
    def transform(self) -> SpatialTransform:
        r"""Spatial grid transformation."""
        return self._transform_new

    @overload
    def condition(self) -> Tuple[tuple, dict]:
        r"""Get arguments on which transformation is conditioned.

        Returns:
            args: Positional arguments.
            kwargs: Keyword arguments.

        """
        ...

    @overload
    def condition(self: TSpatialTransformer, *args, **kwargs) -> TSpatialTransformer:
        r"""Get new transformation which is conditioned on the specified arguments."""
        ...

    def condition(
        self: TSpatialTransformer, *args, **kwargs
    ) -> Union[TSpatialTransformer, Tuple[tuple, dict]]:
        r"""Get or set data tensors and parameters on which transformation is conditioned."""
        if args:
            return shallow_copy(self).condition_(*args)
        return self._transform_new.condition()

    def condition_(self: TSpatialTransformer, *args, **kwargs) -> TSpatialTransformer:
        r"""Set data tensors and parameters on which this transformation is conditioned."""
        self._transform_new.condition_(*args, **kwargs)
        return self

    def update(self: TSpatialTransformer) -> TSpatialTransformer:
        r"""Update internal state of spatial transformation (cf. :meth:`.SpatialTransform.update`)."""
        self._transform_new.update()
        return self


class ImageTransformer(SpatialTransformer):
    r"""Spatially transform an image.

    The :class:`.ImageTransformer` applies a :class:`.SpatialTransform` to the sampling grid
    points of the target domain, optionally followed by linear transformation from target to
    source domain, and samples the input image of shape ``(N, C, ..., X)`` at these deformed
    source grid points. If the spatial transformation is non-rigid, this is also commonly
    referred to as warping the input image.

    Note that the :meth:`.ImageTransformer.forward` method invokes the spatial transform as
    functor, i.e., it triggers any pre-forward and post-forward hooks that are registered with
    the spatial transform when evaluating it. This in particular includes the forward pre-hook
    that invokes :meth:`.SpatialTransform.update` (cf. :class:`.SpatialTransformer`).

    """

    def __init__(
        self,
        transform: SpatialTransform,
        target: Optional[Grid] = None,
        source: Optional[Grid] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str, Scalar] = PaddingMode.BORDER,
        align_centers: bool = False,
        flip_coords: bool = False,
    ) -> None:
        r"""Initialize spatial image transformer.

        Args:
            transform: Spatial coordinate transformation which is applied to ``target`` grid points.
            target: Sampling grid of output images. Use ``transform.grid()`` if ``None``.
            source: Sampling grid of input images. The attributes of this sampling grid are used
                to map the transformed normalized coordinates defined with respect to ``transform.grid()``
                to normalized coordinates with respect to the moving ``source`` image grid. These
                normalized coordinates are used to sample the image tensor passed to the forward
                method of the image transformer. If ``None``, it is assumed that the sampling grid
                of the moving source image is identical to the ``target`` image grid. In order to
                avoid having to re-normalize coordinates between image domains, the ``source`` grid
                domain should be the same as the ``transform.grid()`` (cf. :meth:`Grid.same_domain_as`).
            sampling: Image interpolation mode.
            padding: Image extrapolation mode or scalar out-of-domain value.
            align_centers: Whether to implicitly align the ``target`` and ``source`` centers.
                If ``True``, only the affine component of the target to source transformation
                is applied after the spatial grid ``transform``. If ``False``, also the
                translation of grid center points is considered.
            flip_coords: Whether spatial transformation applies to flipped grid point coordinates
                in the order (z, y, x). The default is grid point coordinates in the order (x, y, z).

        """
        super().__init__(transform)
        if target is None:
            target = transform.grid()
        if source is None:
            source = target
        if not isinstance(target, Grid):
            raise TypeError(f"{type(self).__name__}() 'target' must be of type Grid")
        if not isinstance(source, Grid):
            raise TypeError(f"{type(self).__name__}() 'source' must be of type Grid")
        device = transform.device
        sampler = SampleImage(
            target=transform.grid(),
            source=source,
            sampling=sampling,
            padding=padding,
            align_centers=align_centers,
        )
        # self._sample = sampler.to(device)
        self._sample = sampler
        self._target_grid = target
        self._flip_coords = bool(flip_coords)
        x = target.coords(align_corners=transform.align_corners(), flip=flip_coords, device=device)
        x = target.transform_points(x, axes=transform.axes(), to_grid=transform.grid())
        self.register_buffer(name="grid_coords", tensor=x.unsqueeze(axis=0), persistable=False)

    @property
    def sample(self) -> SampleImage:
        r"""Source image sampler."""
        return self._sample

    def target_grid(self) -> Grid:
        r"""Sampling grid of output images."""
        return self._target_grid

    def source_grid(self) -> Grid:
        r"""Sampling grid of input images."""
        return self._sample.source_grid()

    def align_centers(self) -> bool:
        r"""Whether grid center points are implicitly aligned."""
        return self._sample.align_centers()

    @overload
    def forward(self, data: paddle.Tensor) -> paddle.Tensor:
        r"""Sample batch of images at spatially transformed target grid points."""
        ...

    @overload
    def forward(
        self, data: paddle.Tensor, mask: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        r"""Sample batch of masked images at spatially transformed target grid points."""
        ...

    @overload
    def forward(
        self, data: Dict[str, Union[paddle.Tensor, Grid]]
    ) -> Dict[str, Union[paddle.Tensor, Grid]]:
        r"""Sample batch of optionally masked images at spatially transformed target grid points."""
        ...

    def forward(
        self,
        data: Union[paddle.Tensor, Dict[str, Union[paddle.Tensor, Grid]]],
        mask: Optional[paddle.Tensor] = None,
    ) -> Union[
        paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor], Dict[str, Union[paddle.Tensor, Grid]]
    ]:
        r"""Sample batch of images at spatially transformed target grid points."""
        grid: Tensor = self.grid_coords
        grid = self._transform_new(grid, grid=True)
        if self._flip_coords:
            grid = grid.flip(axis=(-1,))
        return self._sample(grid, data, mask)


class PointSetTransformer(SpatialTransformer):
    r"""Spatially transform a set of points.

    The :class:`.PointSetTransformer` applies a :class:`.SpatialTransform` to a set of input points
    with coordinates defined with respect to a specified target domain. This coordinate map may
    further be followed by a linear transformation from the grid domain of the spatial transform
    to a given source domain. When no spatial transform is given, use :func:`.grid_transform_points`.

    The forward method of a point set transformer performs the same operation as :meth:`.SpatialTransform.points`,
    but with the target and source domain arguments specified during transformer initialization. In addition,
    the point set transformer module invokes the spatial transform as a functor such that any registered
    forward pre- and post-hooks are executed as part of the spatial transform evaluation. This includes the
    forward pre-hook that invokes :meth:`.SpatialTransform.update` (cf. :class:`.SpatialTransformer`).

    """

    def __init__(
        self,
        transform: SpatialTransform,
        grid: Optional[Grid] = None,
        axes: Optional[Union[Axes, str]] = None,
        to_grid: Optional[Grid] = None,
        to_axes: Optional[Union[Axes, str]] = None,
    ) -> None:
        r"""Initialize point set transformer.

        Args:
            transform: Spatial coordinate transformation which is applied to input points.
            grid: Grid with respect to which input points are defined. Uses ``transform.grid()`` if ``None``.
            axes: Coordinate axes with respect to which input points are defined. Uses ``transform.axes()`` if ``None``.
            to_grid: Grid with respect to which output points are defined. Same as ``grid`` if ``None``.
            to_axes: Coordinate axes to which input points should be mapped to. Same as ``axes`` if ``None``.

        """
        super().__init__(transform)
        if grid is None:
            grid = transform.grid()
        if axes is None:
            axes = transform.axes()
        else:
            axes = Axes.from_arg(axes)
        if to_grid is None:
            to_grid = grid
        if to_axes is None:
            to_axes = axes
        else:
            to_axes = Axes.from_arg(to_axes)
        self._grid = grid
        self._axes = axes
        self._to_grid = to_grid
        self._to_axes = to_axes

    def target_axes(self) -> Axes:
        r"""Coordinate axes with respect to which input points are defined."""
        return self._axes

    def target_grid(self) -> Grid:
        r"""Sampling grid with respect to which input points are defined."""
        return self._grid

    def source_axes(self) -> Axes:
        r"""Coordinate axes with respect to which output points are defined."""
        return self._to_axes

    def source_grid(self) -> Grid:
        r"""Sampling grid with respect to which output points are defined."""
        return self._to_grid

    def forward(self, points: paddle.Tensor) -> paddle.Tensor:
        r"""Spatially transform a set of points."""
        transform = self.transform
        points = self._grid.transform_points(
            points,
            axes=self._axes,
            to_grid=transform.grid(),
            to_axes=transform.axes(),
            decimals=None,
        )
        points = transform(points)
        points = transform.grid().transform_points(
            points,
            axes=transform.axes(),
            to_grid=self._to_grid,
            to_axes=self._to_axes,
            decimals=None,
        )
        return points
