from __future__ import annotations

from collections import OrderedDict
from copy import copy as shallow_copy
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import overload

import paddle

from ..core.grid import Axes
from ..core.grid import Grid
from ..core.grid import grid_transform_points
from ..core.linalg import as_homogeneous_matrix
from ..core.linalg import homogeneous_matmul
from ..core.tensor import move_dim
from .base import SpatialTransform

TCompositeTransform = TypeVar("TCompositeTransform", bound="CompositeTransform")


class CompositeTransform(SpatialTransform):
    """Base class of composite spatial coordinate transformations.

    Base class of modules that apply one or more spatial transformations to map a tensor of
    spatial points to another tensor of spatial points of the same shape as the input tensor.

    """

    @overload
    def __init__(self, grid: Grid) -> None:
        """Initialize empty composite transformation."""
        ...

    @overload
    def __init__(self, grid: Grid, *args: Optional[SpatialTransform]) -> None:
        """Initialize composite transformation."""
        ...

    @overload
    def __init__(
        self, grid: Grid, transforms: Union[OrderedDict, paddle.nn.LayerDict]
    ) -> None:
        """Initialize composite transformation given named transforms in ordered dictionary."""
        ...

    @overload
    def __init__(self, *args: Optional[SpatialTransform]) -> None:
        """Initialize composite transformation."""
        ...

    @overload
    def __init__(self, transforms: Union[paddle.nn.LayerDict, OrderedDict]) -> None:
        """Initialize composite transformation given named transforms in ordered dictionary."""
        ...

    def __init__(
        self,
        *args: Optional[
            Union[Grid, paddle.nn.LayerDict, OrderedDict, SpatialTransform]
        ],
    ) -> None:
        """Initialize composite transformation."""
        args_ = [arg for arg in args if arg is not None]
        grid = None
        if isinstance(args_[0], Grid):
            grid = args_[0]
            args_ = args_[1:]
        if args_:
            if isinstance(args_[0], (dict, paddle.nn.LayerDict)):
                if len(args_) > 1:
                    raise ValueError(
                        f"{type(self).__name__}() multiple arguments not allowed when dict is given"
                    )
                transforms = args_[0]
            else:
                transforms = OrderedDict([(str(i), t) for i, t in enumerate(args_)])
        else:
            transforms = OrderedDict()
        if grid is None:
            if transforms:
                transform = next(iter(transforms.values()))
                grid = transform.grid()
            else:
                raise ValueError(
                    f"{type(self).__name__}() requires a Grid or at least one SpatialTransform"
                )
        for name, transform in transforms.items():
            if not isinstance(transform, SpatialTransform):
                raise TypeError(
                    f"{type(self).__name__}() module '{name}' must be of type SpatialTransform"
                )
            if not transform.grid().same_domain_as(grid):
                raise ValueError(
                    f"{type(self).__name__}() transform '{name}' has different 'grid' center, direction, or cube extent"
                )
        super().__init__(grid)
        self._transforms = paddle.nn.LayerDict(sublayers=transforms)

    def bool(self) -> bool:
        """Whether this module has at least one transformation."""
        return len(self._transforms) > 0

    def __len__(self) -> int:
        """Number of spatial transformations."""
        return len(self._transforms)

    @property
    def linear(self) -> bool:
        """Whether composite transformation is linear."""
        return all(transform.linear for transform in self.transforms())

    def __contains__(self, name: Union[int, str]) -> bool:
        """Whether composite contains named transformation."""
        if isinstance(name, int):
            name = str(name)
        return name in self._transforms.keys()

    def __getitem__(self, name: Union[int, str]) -> SpatialTransform:
        """Get named transformation."""
        if isinstance(name, int):
            name = str(name)
        return self._transforms[name]

    def get(
        self, name: Union[int, str], default: Optional[SpatialTransform] = None
    ) -> Optional[SpatialTransform]:
        """Get named transformation."""
        if isinstance(name, int):
            name = str(name)
        for key, transform in self._transforms.items():
            if key == name:
                assert isinstance(transform, SpatialTransform)
                return transform
        return default

    def transforms(self) -> Iterable[SpatialTransform]:
        """Iterate transformations in order of composition."""
        return self._transforms.values()

    def named_transforms(self) -> Iterable[Tuple[str, SpatialTransform]]:
        """Iterate transformations in order of composition."""
        return self._transforms.items()

    def condition(
        self: TCompositeTransform, *args, **kwargs
    ) -> Union[TCompositeTransform, Optional[paddle.Tensor]]:
        """Get or set data tensor on which transformations are conditioned."""
        if args or kwargs:
            return shallow_copy(self).condition_(*args, **kwargs)
        return self._args, self._kwargs

    def condition_(self: TCompositeTransform, *args, **kwargs) -> TCompositeTransform:
        """Set data tensor on which transformations are conditioned."""
        assert args or kwargs
        super().condition_(*args, **kwargs)
        for transform in self.transforms():
            transform.condition_(*args, **kwargs)
        return self

    def disp(self, grid: Optional[Grid] = None) -> paddle.Tensor:
        """Get displacement vector field representation of this transformation.

        Args:
            grid: Grid on which to sample vector fields. Use ``self.grid()`` if ``None``.

        Returns:
            Displacement vector fields as tensor of shape ``(N, D, ..., X)``.

        """
        if grid is None:
            grid = self.grid()
        axes = Axes.from_grid(grid)
        x = grid.coords(device=self.device).unsqueeze(axis=0)
        if grid.same_domain_as(self.grid()):
            y = self.forward(x)
        else:
            y = grid_transform_points(x, grid, axes, self.grid(), self.axes())
            y = self.forward(y)
            y = grid_transform_points(y, self.grid(), self.axes(), grid, axes)
        u = y - x
        u = move_dim(u, -1, 1)
        return u

    def update(self: TCompositeTransform) -> TCompositeTransform:
        """Update buffered data such as predicted parameters, velocities, and/or displacements."""
        super().update()
        for transform in self.transforms():
            transform.update()
        return self

    def clear_buffers(self: TCompositeTransform) -> TCompositeTransform:
        """Clear any buffers that are registered by ``self.update()``."""
        super().clear_buffers()
        for transform in self.transforms():
            transform.clear_buffers()
        return self


class MultiLevelTransform(CompositeTransform):
    """Sum of spatial transformations applied to any set of points.

    A :class:`.MultiLevelTransform` adds the sum of the displacement vectors across all
    spatial transforms at the input points that are being mapped to new locations, i.e.,

    .. math::

        \\vec{y} = \\vec{x} + \\sum_{i=0}^{n-1} \\vec{u}_i(\\vec{x})

    """

    def forward(self, points: paddle.Tensor, grid: bool = False) -> paddle.Tensor:
        """Transform set of points by sum of spatial transformations.

        Args:
            points: paddle.Tensor of shape ``(N, M, D)`` or ``(N, ..., Y, X, D)``.
            grid: Whether ``points`` are the positions of undeformed grid points.

        Returns:
            paddle.Tensor of same shape as ``points`` with transformed point coordinates.

        """
        x = points
        if len(self) == 0:
            return x
        if self.linear:
            y = super().forward(points, grid)
        else:
            u = paddle.zeros_like(x=x)
            for i, transform in enumerate(self.transforms()):
                y = transform.forward(x, grid=grid and i == 0)
                u += y - x
            y = x + u
        return y

    def tensor(self) -> paddle.Tensor:
        """Get tensor representation of this transformation.

        The tensor representation of a transformation is with respect to the unit cube axes defined
        by its sampling grid as specified by ``self.axes()``.

        Returns:
            In case of a composition of linear transformations, returns a batch of homogeneous transformation
            matrices as tensor of shape ``(N, D, 1)`` (translation),  ``(N, D, D)`` (affine) or ``(N, D, D + 1)``,
            i.e., a 3-dimensional tensor. If this composite transformation contains a non-rigid transformation,
            a displacement vector field is returned as tensor of shape ``(N, D, ..., X)``.

        """
        if self.linear:
            transforms = list(self.transforms())
            if not transforms:
                identity = paddle.eye(num_rows=self.ndim, num_columns=self.ndim + 1)
                return identity.unsqueeze(axis=0)
            transform = transforms[0]
            mat = as_homogeneous_matrix(transform.tensor())
            for transform in transforms[1:]:
                mat += as_homogeneous_matrix(transform.tensor())
            return mat
        return self.disp()


class SequentialTransform(CompositeTransform):
    """Composition of spatial transformations applied to any set of points.

    A :class:`.SequentialTransform` is the functional composition of spatial transforms, i.e.,

    .. math::

        \\vec{y} = \\vec{u}_{n-1} \\circ \\cdots \\circ \\vec{u}_0 \\circ \\vec{x}

    """

    def forward(self, points: paddle.Tensor, grid: bool = False) -> paddle.Tensor:
        """Transform points by sequence of spatial transformations.

        Args:
            points: paddle.Tensor of shape ``(N, M, D)`` or ``(N, ..., Y, X, D)``.
            grid: Whether ``points`` are the positions of undeformed grid points.

        Returns:
            paddle.Tensor of same shape as ``points`` with transformed point coordinates.

        """
        if self.linear:
            return super().forward(points, grid)
        y = points
        for i, transform in enumerate(self.transforms()):
            y = transform.forward(y, grid=grid and i == 0)
        return y

    def tensor(self) -> paddle.Tensor:
        """Get tensor representation of this transformation.

        The tensor representation of a transformation is with respect to the unit cube axes defined
        by its sampling grid as specified by ``self.axes()``.

        Returns:
            In case of a composition of linear transformations, returns a batch of homogeneous transformation
            matrices as tensor of shape ``(N, D, 1)`` (translation),  ``(N, D, D)`` (affine) or ``(N, D, D + 1)``,
            i.e., a 3-dimensional tensor. If this composite transformation contains a non-rigid transformation,
            a displacement vector field is returned as tensor of shape ``(N, D, ..., X)``.

        """
        if self.linear:
            transforms = list(self.transforms())
            if not transforms:
                identity = paddle.eye(num_rows=self.ndim, num_columns=self.ndim + 1)
                return identity.unsqueeze(axis=0)
            transform = transforms[0]
            mat = transform.tensor()
            for transform in transforms[1:]:
                mat = homogeneous_matmul(transform.tensor(), mat)
            return mat
        return self.disp()

    def inverse(
        self: TCompositeTransform, link: bool = False, update_buffers: bool = False
    ) -> TCompositeTransform:
        """Get inverse of this transformation.

        Args:
            link: Whether to inverse transformation keeps a reference to this transformation.
                If ``True``, the ``update()`` function of the inverse function will not recompute
                shared parameters, e.g., parameters obtained by a callable neural network, but
                directly access the parameters from this transformation. Note that when ``False``,
                the inverse transformation will still share parameters, modules, and buffers with
                this transformation, but these shared tensors may be replaced by a call of ``update()``
                (which is implicitly called as pre-forward hook when ``__call__()`` is invoked).
            update_buffers: Whether buffers of inverse transformation should be update after creating
                the shallow copy. If ``False``, the ``update()`` function of the returned inverse
                transformation has to be called before it is used.

        Returns:
            Shallow copy of this transformation which computes and applied the inverse transformation.
            The inverse transformation will share the parameters with this transformation. Not all
            transformations may implement this functionality.

        Raises:
            NotImplementedError: When a transformation does not support sharing parameters with its inverse.

        """
        copy = shallow_copy(self)
        transforms = paddle.nn.LayerDict()
        for name, transform in reversed(self.named_transforms()):
            assert isinstance(transform, SpatialTransform)
            transforms[name] = transform.inverse(
                link=link, update_buffers=update_buffers
            )
        copy._transforms = transforms
        return copy
