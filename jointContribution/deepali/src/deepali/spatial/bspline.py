r"""Cubic B-spline free-form deformations."""
from __future__ import annotations  # noqa

from copy import copy as shallow_copy
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import cast
from typing import overload

import paddle
from deepali.core import functional as U
from deepali.core import kernels as K
from deepali.core.enum import SpatialDim
from deepali.core.grid import Grid
from deepali.core.typing import ScalarOrTuple
from deepali.modules import ExpFlow

from .base import NonRigidTransform
from .parametric import ParametricTransform

TBSplineTransform = TypeVar("TBSplineTransform", bound="BSplineTransform")


class BSplineTransform(ParametricTransform, NonRigidTransform):
    r"""Non-rigid transformation parameterized by cubic B-spline function."""

    def __init__(
        self,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        stride: Optional[Union[int, Sequence[int]]] = None,
        transpose: bool = False,
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
            stride: Number of grid points between control points plus one. This is the stride of the
                transposed convolution used to upsample the control point displacements to the sampling ``grid``
                size. If ``None``, a stride of 1 is used. If a sequence of values is given, these must be the
                strides for the different spatial grid dimensions in the order ``(sx, sy, sz)``. Note that
                when the control point grid is subdivided in order to double its size along each spatial
                dimension, the stride with respect to this subdivided control point grid remains the same.
            transpose: Whether to use separable transposed convolution as implemented in AIRLab.
                When ``False``, a more efficient implementation using multi-channel convolution followed
                by a reshuffling of the output is performed. This more efficient and also more accurate
                implementation is adapted from the C++ code of MIRTK (``mirtk::BSplineInterpolateImageFunction``).

        """
        if not grid.align_corners():
            raise ValueError("BSplineTransform() requires 'grid.align_corners() == True'")
        if stride is None:
            stride = 5
        if isinstance(stride, int):
            stride = (stride,) * grid.ndim
        if len(stride) != grid.ndim:
            raise ValueError(f"BSplineTransform() 'stride' must be single int or {grid.ndim} ints")
        self.stride = tuple(int(s) for s in stride)
        self._transpose = transpose  # MUST be set before register_kernels() is called
        super().__init__(grid, groups=groups, params=params)
        self.register_kernels(stride)

    @property
    def data_shape(self) -> list:
        r"""Get shape of transformation parameters tensor."""
        grid = self.grid()
        shape = U.cubic_bspline_control_point_grid_size(tuple(grid.shape), self.data_stride)
        return tuple((grid.ndim,) + shape)

    @property
    def data_stride(self) -> Tuple[int, ...]:
        return tuple(reversed([int(s) for s in self.stride]))

    @paddle.no_grad()
    def grid_(self: TBSplineTransform, grid: Grid) -> TBSplineTransform:
        r"""Set sampling grid of transformation domain and codomain.

        If ``self.params`` is a callable, only the grid attribute is updated, and
        the callable must return a tensor of matching size upon next evaluation.

        Args:
            grid: New sampling grid for dense displacement field at which FFD is evaluated.
                This function currently only supports subdivision of the control point grid,
                i.e., the new ``grid`` must have size ``2 * n - 1`` along each spatial dimension
                that should be subdivided, where ``n`` is the current grid size, or have the same
                size as the current grid for dimensions that remain the same.

        Returns:
            Reference to this modified transformation object.

        """
        params = self.params
        current_grid = self._grid
        if grid.ndim != current_grid.ndim:
            raise ValueError(
                f"{type(self).__name__}.grid_() argument must have {current_grid.ndim} dimensions"
            )
        subdivide_dims: List[SpatialDim] = []
        if isinstance(params, paddle.Tensor):
            current_grid = self._grid
            if grid.ndim != current_grid.ndim:
                raise ValueError(
                    f"{type(self).__name__}.grid_() argument must have {current_grid.ndim} dimensions"
                )
            if not grid.align_corners():
                raise ValueError(
                    f"{type(self).__name__}() requires grid.align_corners() to be True"
                )
            if not grid.same_domain_as(current_grid):
                raise ValueError(
                    f"{type(self).__name__}.grid_() argument must define same grid domain as current grid"
                )
            new_size = grid.size()
            current_size = current_grid.size()
            for i in range(grid.ndim):
                if new_size[i] == 2 * current_size[i] - 1:
                    subdivide_dims.append(SpatialDim(i))
                elif new_size[i] != current_size[i]:
                    raise ValueError(
                        f"{type(self).__name__}.grid_() argument must have same size or new size '2n - 1'"
                    )
        self._grid = grid
        if subdivide_dims:
            new_shape = (tuple(params.shape)[0],) + self.data_shape
            new_params = U.subdivide_cubic_bspline(params, dims=subdivide_dims)
            for dim in subdivide_dims:
                dim = dim.tensor_dim(params.ndim)
                start_48 = new_params.shape[dim] + 1 if 1 < 0 else 1
                new_params = paddle.slice(
                    new_params, [dim], [start_48], [start_48 + new_shape[dim]]
                )
            self.data_(new_params.contiguous())
        return self

    @staticmethod
    def kernel_name(stride: int) -> str:
        r"""Get name of buffer for 1-dimensional kernel for given control point spacing."""
        return "kernel_stride_" + str(stride)

    @overload
    def kernel(self) -> Tuple[paddle.Tensor, ...]:
        ...

    @overload
    def kernel(self, stride: int) -> paddle.Tensor:
        ...

    @overload
    def kernel(self, stride: Sequence[int]) -> Tuple[paddle.Tensor, ...]:
        ...

    def kernel(
        self, stride: Optional[ScalarOrTuple[int]] = None
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
        r"""Get 1-dimensional kernels for given control point spacing."""
        if stride is None:
            stride = self.stride
        if isinstance(stride, int):
            return getattr(self, self.kernel_name(stride))
        return tuple(getattr(self, self.kernel_name(s)) for s in stride)

    def register_kernels(self, stride: Union[int, Sequence[int]]) -> None:
        r"""Precompute cubic B-spline kernels."""
        if isinstance(stride, int):
            stride = [stride]
        for s in stride:
            name = self.kernel_name(s)
            if not hasattr(self, name):
                if self._transpose:
                    kernel = K.cubic_bspline1d(s)
                else:
                    kernel = U.bspline_interpolation_weights(degree=3, stride=s)
                self.register_buffer(name=name, tensor=kernel, persistable=False)

    def deregister_kernels(self, stride: Union[int, Sequence[int]]) -> None:
        r"""Remove precomputed cubic B-spline kernels."""
        if isinstance(stride, int):
            stride = [stride]
        for s in stride:
            name = self.kernel_name(s)
            if hasattr(self, name):
                delattr(self, name)

    def evaluate_spline(self) -> paddle.Tensor:
        r"""Evaluate cubic B-spline at sampling grid points."""
        data = self.data()
        grid = self.grid()
        if not grid.align_corners():
            raise AssertionError(
                f"{type(self).__name__}() requires grid.align_corners() to be True"
            )
        stride = self.stride
        kernel = self.kernel(stride)
        u = U.evaluate_cubic_bspline(
            data, shape=tuple(grid.shape), stride=stride, kernel=kernel, transpose=self._transpose
        )
        return u


class FreeFormDeformation(BSplineTransform):
    r"""Cubic B-spline free-form deformation model."""

    def update(self) -> FreeFormDeformation:
        r"""Update buffered displacement vector field."""
        super().update()
        u = self.evaluate_spline()
        self.register_buffer(name="u", tensor=u, persistable=False)
        return self


class StationaryVelocityFreeFormDeformation(BSplineTransform):
    r"""Stationary velocity field based transformation model using cubic B-spline parameterization."""

    def __init__(
        self,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        stride: Optional[Union[int, Sequence[int]]] = None,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
        transpose: bool = False,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid on which to sample flow field vectors.
            groups: Number of velocity fields.
            params: Initial parameters of cubic B-spline velocity fields of shape ``(N, C, ...X)``.
            stride: Number of grid points between control points (minus one).
            scale: Constant scaling factor of velocity fields.
            steps: Number of scaling and squaring steps.
            transpose: Whether to use separable transposed convolution as implemented in AIRLab.
                When ``False``, a more efficient implementation using multi-channel convolution followed
                by a reshuffling of the output is performed. This more efficient and also more accurate
                implementation is adapted from the C++ code of MIRTK (``mirtk::BSplineInterpolateImageFunction``).

        """
        align_corners = grid.align_corners()
        super().__init__(grid, groups=groups, params=params, stride=stride, transpose=transpose)
        self.exp = ExpFlow(scale=scale, steps=steps, align_corners=align_corners)

    def inverse(
        self, link: bool = False, update_buffers: bool = False
    ) -> StationaryVelocityFreeFormDeformation:
        r"""Get inverse of this transformation.

        Args:
            link: Whether the inverse transformation keeps a reference to this transformation.
                If ``True``, the ``update()`` function of the inverse function will not recompute
                shared parameters (e.g., parameters obtained by a callable neural network), but
                directly access the parameters from this transformation. Note that when ``False``,
                the inverse transformation will still share parameters, modules, and buffers with
                this transformation, but these shared tensors may be replaced by a call of ``update()``
                (which is implicitly called as pre-forward hook when ``__call__()`` is invoked).
            update_buffers: Whether buffers of inverse transformation should be updated after creating
                the shallow copy. If ``False``, the ``update()`` function of the returned inverse
                transformation has to be called before it is used.

        Returns:
            Shallow copy of this transformation with ``exp`` module which uses negative scaling factor
            to scale and square the stationary velocity field to compute the inverse displacement field.

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

    def update(self) -> StationaryVelocityFreeFormDeformation:
        r"""Update buffered velocity and displacement vector fields."""
        super().update()
        v = self.evaluate_spline()
        u = self.exp(v)
        self.register_buffer(name="v", tensor=v, persistable=False)
        self.register_buffer(name="u", tensor=u, persistable=False)
        return self
