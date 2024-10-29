r"""Non-rigid transformation models."""
from __future__ import annotations  # noqa

import math
from copy import copy as shallow_copy
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union
from typing import cast

import deepali.utils.paddle_aux  # noqa
import paddle
from deepali.core import functional as U
from deepali.core.grid import Axes
from deepali.core.grid import Grid
from deepali.data.flow import FlowFields
from deepali.modules import ExpFlow

from .base import NonRigidTransform
from .parametric import ParametricTransform

TDenseVectorFieldTransform = TypeVar(
    "TDenseVectorFieldTransform", bound="DenseVectorFieldTransform"
)


class DenseVectorFieldTransform(ParametricTransform, NonRigidTransform):
    r"""Dense vector field transformation with linear interpolation at non-grid point locations."""

    def __init__(
        self,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        stride: Optional[Union[float, Sequence[float]]] = None,
        resize: bool = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. A given image batch can either be deformed by a
                single transformation, or a separate transformation for each image in the batch, e.g.,
                for group-wise or batched registration. The default is one transformation for all images
                in the batch, or the batch length of the ``params`` tensor if provided.
            params: Initial parameters. If a tensor is given, it is only registered as optimizable module
                parameters when of type ``paddle.nn.Parameter``. When a callable is given instead, it will be
                called by ``self.update()`` with arguments set and given by ``self.condition()``. When a boolean
                argument is given, a new zero-initialized tensor is created. If ``True``, this tensor is registered
                as optimizable module parameter.
            stride: Spacing between vector field grid points in units of input ``grid`` points.
                Can be used to subsample the dense vector field with respect to the image grid of the fixed target
                image. When ``grid.align_corners() is True``, the corner points of the ``grid`` and the resampled
                vector field grid are aligned. Otherwise, the edges of the grid domains are aligned.
                Must be either a single scalar if the grid is subsampled equally along each spatial dimension,
                or a sequence with length less than or equal to ``grid.ndim``, with subsampling factors given
                in the order (x, ...). When a sequence shorter than ``grid.ndim`` is given, remaining spatial
                dimensions are not being subsampled.
            resize: Whether to resize vector field during transformation update. If ``True``, the buffered vector
                field ``u`` (and ``v`` if applicable) is resized to match the image ``grid`` size. This means that
                transformation constraints defined on these resized vector fields, such as those based on finite
                differences, are evaluated at the image grid resolution rather than the resolution of the underlying
                vector field parameterization. This influences the scale at which these constraints are imposed.

        """
        if stride is None:
            stride = 1
        if isinstance(stride, (int, float)):
            stride = (stride,) * grid.ndim
        if not isinstance(stride, Sequence):
            raise TypeError(f"{type(self).__name__}() 'stride' must be float or Sequence[float]")
        if len(stride) > grid.ndim:
            raise ValueError(
                f"{type(self).__name__}() 'stride' sequence length ({len(stride)}) exceeds grid dimensions ({grid.ndim})"
            )
        stride = tuple(float(s) for s in stride) + (1.0,) * (grid.ndim - len(stride))
        self.stride = stride
        self._resize = resize
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self) -> list:
        r"""Get shape of transformation parameters tensor."""
        grid = self.grid()
        shape = self.data_grid_shape(grid)
        return tuple((grid.ndim,) + shape)

    def data_grid(
        self, grid: Optional[Grid] = None, stride: Optional[Union[float, Sequence[float]]] = None
    ) -> Grid:
        if grid is None:
            grid = self.grid()
        if stride is None:
            stride = self.stride
        return grid.reshape(self.data_grid_shape(grid, stride))

    def data_grid_shape(
        self, grid: Optional[Grid] = None, stride: Optional[Union[float, Sequence[float]]] = None
    ) -> list:
        if grid is None:
            grid = self.grid()
        if stride is None:
            stride = self.stride
        if isinstance(stride, (int, float)):
            stride = (stride,) * grid.ndim
        if not isinstance(stride, Sequence):
            raise TypeError(
                f"{type(self).__name__}.data_grid_shape() 'stride' must be float or Sequence[float]"
            )
        if len(stride) > grid.ndim:
            raise ValueError(
                f"{type(self).__name__}.data_grid_shape() 'stride' sequence length ({len(stride)}) exceeds grid dimensions ({grid.ndim})"
            )
        stride = tuple(float(s) for s in stride) + (1.0,) * (grid.ndim - len(stride))
        return tuple(int(math.ceil(n / s)) for n, s in zip(tuple(grid.shape), reversed(stride)))

    @paddle.no_grad()
    def grid_(self: TDenseVectorFieldTransform, grid: Grid) -> TDenseVectorFieldTransform:
        r"""Set sampling grid of transformation domain and codomain.

        If ``self.params`` is a callable, only the grid attribute is updated, and
        the callable must return a tensor of matching size upon next evaluation.

        """
        params = self.params
        if isinstance(params, paddle.Tensor):
            prev_grid = self._grid
            grid_axes = Axes.from_grid(grid)
            flow_axes = self.axes()
            flow_grid = prev_grid.reshape(tuple(params.shape)[2:])
            flow = FlowFields(params, grid=flow_grid, axes=flow_axes)
            flow = flow.sample(shape=self.data_grid(grid))
            flow = flow.axes(grid_axes)
            # Change self._grid before self.data_() as it defines self.data_shape
            super().grid_(grid)
            try:
                self.data_(flow.tensor())
            except Exception:
                self._grid = prev_grid
                raise
        else:
            super().grid_(grid)
        return self

    def evaluate(self, resize: Optional[bool] = None) -> paddle.Tensor:
        r"""Update buffered displacement vector field."""
        if resize is None:
            resize = self._resize
        u = self.data()
        u = u.view(*tuple(u.shape))  # such that named_buffers() returns both 'u' (or 'v') and 'p'
        if resize:
            align_corners = self.align_corners()
            grid_shape = tuple(self.grid().shape)
            u = U.grid_reshape(u, grid_shape, align_corners=align_corners)
        return u


class DisplacementFieldTransform(DenseVectorFieldTransform):
    r"""Dense displacement field transformation model."""

    def fit(self, flow: FlowFields, **kwargs) -> DisplacementFieldTransform:
        r"""Fit transformation to a given flow field.

        Args:
            flow: Flow fields to approximate.
            kwargs: Optional keyword arguments are ignored.

        Returns:
            Reference to this transformation.

        Raises:
            RuntimeError: When this transformation has no optimizable parameters.

        """
        params = self.params
        if params is None:
            raise AssertionError(f"{type(self).__name__}.data() 'params' must be set first")
        grid = self.grid()
        if not callable(params):
            grid = self.grid().resize(self.data_shape[:1:-1])
        flow = flow.to(self.device)
        flow = flow.sample(shape=grid)
        flow = flow.axes(grid.axes())
        if callable(params):
            self._fit(flow, **kwargs)
        else:
            self.data_(flow.tensor())
        return self

    def update(self) -> DisplacementFieldTransform:
        r"""Update buffered displacement vector field."""
        super().update()
        u = self.evaluate()
        self.register_buffer(name="u", tensor=u, persistable=False)
        return self


class StationaryVelocityFieldTransform(DenseVectorFieldTransform):
    r"""Dense stationary velocity field transformation."""

    def __init__(
        self,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        stride: Optional[Union[float, Sequence[float]]] = None,
        resize: bool = True,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid on which to sample velocity field vectors.
            groups: Number of velocity fields. A given image batch can either be deformed by a
                single displacement field, or one separate displacement field for each image in the
                batch, e.g., for group-wise or batched registration. The default is one displacement
                field for all images in the batch, or the batch length N of ``params`` if provided.
            params: Initial parameters of velocity fields of shape ``(N, C, ...X)``, where N must match
                the value of ``groups``, and vector components are the image channels in the order x, y, z.
                Note that a tensor is only registered as optimizable module parameters when of type
                ``paddle.nn.Parameter``. When a callable is given instead, it will be called each time the
                model parameters are accessed with the arguments set and returned by ``self.condition()``.
                When a boolean argument is given, a new zero-initialized tensor is created. If ``True``,
                it is registered as optimizable parameter.
            stride: Spacing between vector field grid points in units of input ``grid`` points.
                Can be used to subsample the dense vector field with respect to the image grid of the fixed target
                image. When ``grid.align_corners() is True``, the corner points of the ``grid`` and the resampled
                vector field grid are aligned. Otherwise, the edges of the grid domains are aligned.
                Must be either a single scalar if the grid is subsampled equally along each spatial dimension,
                or a sequence with length less than or equal to ``grid.ndim``, with subsampling factors given
                in the order (x, ...). When a sequence shorter than ``grid.ndim`` is given, remaining spatial
                dimensions are not being subsampled.
            resize: Whether to resize vector field during transformation update. If ``True``, the buffered vector
                fields ``v`` and ``u`` are resized to match the image ``grid`` size. This means that transformation
                constraints defined on these resized vector fields, such as those based on finite differences, are
                evaluated at the image grid resolution rather than the resolution of the underlying vector field
                parameterization. This influences the scale at which these constraints are imposed.
            scale: Constant scaling factor of velocity fields.
            steps: Number of scaling and squaring steps.

        """
        super().__init__(grid, groups=groups, params=params, stride=stride, resize=resize)
        self.exp = ExpFlow(scale=scale, steps=steps, align_corners=grid.align_corners())

    def grid_(self, grid: Grid) -> StationaryVelocityFieldTransform:
        r"""Set sampling grid of transformation domain and codomain."""
        super().grid_(grid)
        self.exp.align_corners = grid.align_corners()
        return self

    def inverse(
        self, link: bool = False, update_buffers: bool = False
    ) -> StationaryVelocityFieldTransform:
        r"""Get inverse of this transformation.

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
            Shallow copy of this transformation with ``exp`` module which uses negative scaling factor
            to scale and square the stationary velocity field to computes the inverse displacement field.

        """
        inv = shallow_copy(self)
        if link:
            inv.link_(self)
        inv.exp = cast(ExpFlow, self.exp).inverse()
        if update_buffers:
            v = getattr(inv, "v", None)
            if v is not None:
                u = inv.exp(v)
                inv.register_buffer(name="u", tensor=u, persistable=False)
        return inv

    def update(self) -> StationaryVelocityFieldTransform:
        r"""Update buffered velocity and displacement vector fields."""
        super().update()
        v = self.evaluate()
        u = self.exp(v)
        self.register_buffer(name="v", tensor=v, persistable=False)
        self.register_buffer(name="u", tensor=u, persistable=False)
        return self
