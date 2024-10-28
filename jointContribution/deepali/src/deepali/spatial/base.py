r"""Base classes of spatial coordinate transformations."""

from __future__ import annotations  # noqa

from abc import ABCMeta
from abc import abstractmethod
from copy import copy as shallow_copy
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import final
from typing import overload

import deepali.utils.paddle_aux  # noqa
import paddle
from deepali.core import functional as U
from deepali.core.grid import Axes
from deepali.core.grid import Grid
from deepali.core.linalg import as_homogeneous_matrix
from deepali.core.typing import Device
from deepali.data.flow import FlowFields
from deepali.modules import DeviceProperty

TSpatialTransform = TypeVar("TSpatialTransform", bound="SpatialTransform")
TLinearTransform = TypeVar("TLinearTransform", bound="LinearTransform")
TNonRigidTransform = TypeVar("TNonRigidTransform", bound="NonRigidTransform")


class ReadOnlyParameters(RuntimeError):
    r"""Exception thrown when attempting to set parameters when these are provided by a callable."""

    ...


class SpatialTransform(DeviceProperty, paddle.nn.Layer, metaclass=ABCMeta):
    r"""Base class of all spatial coordinate transformations."""

    def __init__(self, grid: Grid):
        r"""Initialize base class.

        Args:
            grid: Spatial domain with respect to which transformation is defined.
                The unit cube domain is :attr:`.Axes.CUBE` if ``grid.align_corners() == False``,
                and :attr:`.Axes.CUBE_CORNERS` otherwise.

        """
        if not isinstance(grid, Grid):
            raise TypeError("SpatialTransform() 'grid' must be of type Grid")
        super().__init__()
        self._grid = grid
        self._args = ()
        self._kwargs = {}
        self.register_update_hook()

    def __copy__(self: TSpatialTransform) -> TSpatialTransform:
        r"""Make shallow copy of this transformation.

        The copy shares containers for parameters and hooks with this module, but not containers of
        buffers and modules. References to currently set buffers and modules are however copied, but
        adding/removing a buffer or module to/from the shallow copy will not modify the buffers and
        modules of the original module. The same is the case for adding/removing buffers or modules
        to/from the original module.

        Returns:
            Shallow copy of this spatial transformation module.

        """
        copy = self.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        for name in ("_buffers", "_non_persistent_buffers_set", "_modules"):
            if name in self.__dict__:
                copy.__dict__[name] = self.__dict__[name].copy()
        return copy

    @overload
    def condition(self) -> Tuple[tuple, dict]:
        r"""Get arguments on which transformation is conditioned.

        Returns:
            args: Positional arguments.
            kwargs: Keyword arguments.

        """
        ...

    @overload
    def condition(self: TSpatialTransform, *args, **kwargs) -> TSpatialTransform:
        r"""Get new transformation which is conditioned on the specified arguments."""
        ...

    def condition(
        self: TSpatialTransform, *args, **kwargs
    ) -> Union[TSpatialTransform, Tuple[tuple, dict]]:
        r"""Get or set data tensor on which transformation is conditioned."""
        if args:
            return shallow_copy(self).condition_(*args)
        return self._args, self._kwargs

    def condition_(self: TSpatialTransform, *args, **kwargs) -> TSpatialTransform:
        r"""Set data tensor on which this transformation is conditioned."""
        self.clear_buffers()
        self._args = args
        self._kwargs = kwargs
        return self

    def align_corners(self) -> bool:
        r"""Whether extrema -1 and 1 coincide with grid border (False) or corner points (True)."""
        return self._grid.align_corners()

    def axes(self) -> Axes:
        r"""Axes with respect to which transformation is defined.

        Returns:
            ``Axes.CUBE_CORNERS if self.align_corners() else Axes.CUBE``.

        """
        return Axes.from_align_corners(self.align_corners())

    @overload
    def grid(self) -> Grid:
        ...

    @overload
    def grid(self: TSpatialTransform, grid: Grid) -> TSpatialTransform:
        ...

    def grid(self, grid: Optional[Grid] = None) -> Grid:
        r"""Get grid domain of this transformation or a new transformation with the specified grid."""
        if grid is None:
            return self._grid
        return shallow_copy(self).grid_(grid)

    def grid_(self: TSpatialTransform, grid: Grid) -> TSpatialTransform:
        r"""Set sampling grid which defines domain and codomain of this transformation."""
        if self._grid == grid:
            return self
        if grid.ndim != self.ndim:
            raise ValueError(f"{type(self).__name__}.grid_() must be {self.ndim}-dimensional")
        self.clear_buffers()
        self._grid = grid
        return self

    def dim(self) -> int:
        r"""Number of spatial dimensions."""
        return self._grid.ndim

    @property
    def ndim(self) -> int:
        r"""Number of spatial dimensions."""
        return self.dim()

    @property
    def linear(self) -> bool:
        r"""Whether this transformation is linear."""
        return isinstance(self, LinearTransform)

    @property
    def nonrigid(self) -> bool:
        r"""Whether this transformation is non-rigid."""
        return not self.linear

    def fit(self: TSpatialTransform, flow: FlowFields, **kwargs) -> TSpatialTransform:
        r"""Fit transformation to a given flow field.

        Args:
            flow: Flow fields to approximate.
            kwargs: Optional keyword arguments of fitting algorithm.
                Arguments which are unused by a concrete implementation are ignored
                without raising an error, e.g., a specified gradient descent step length
                when a least square fit is computed instead.

        Returns:
            Reference to this transformation.

        Raises:
            RuntimeError: When this transformation has no optimizable parameters.

        """
        self.clear_buffers()
        grid = self.grid()
        flow = flow.to(self.device)
        flow = flow.sample(grid)
        flow = flow.axes(Axes.from_grid(grid))
        self._fit(flow, **kwargs)
        return self

    def _fit(self, flow: FlowFields, **kwargs) -> None:
        r"""Fit transformation to flow field.

        This function may be overidden by subclasses to implement an analytic least squares
        or otherwise more suitable fitting approach. Optimization related keyword arguments
        may be ignored by these specializations.

        Args:
            flow: Batch of flow vector fields sampled on ``self.grid()`` and
                defined with respect to either ``Axes.CUBE`` or ``Axes.CUBE_CORNERS``
                depending on flag ``self.grid().align_corners()``. These displacement vector
                fields will be approximated by this transformation.
            kwargs: Keyword arguments of iterative optimization. Unused arguments are ignored.
                lr: Initial step size for iterative gradient-based optimization.
                steps: Maximum number of gradient steps after which to terminate.
                epsilon: Upper mean squared error threshold at which to terminate.

        Raises:
            RuntimeError: When this transformation has no optimizable parameters.

        """
        lr = float(kwargs.get("lr", 0.1))
        steps = int(kwargs.get("steps", 1000))
        epsilon = float(kwargs.get("epsilon", 1e-05))
        verbose = int(kwargs.get("verbose", 0))
        params = list(self.parameters())
        if not params:
            raise RuntimeError(
                f"{type(self).__name__}.fit() transformation has no optimizable parameters"
            )
        optimizer = paddle.optimizer.Adam(parameters=params, learning_rate=lr, weight_decay=0.0)
        for step in range(steps):
            optimizer.clear_gradients(set_to_zero=False)
            loss = paddle.nn.functional.mse_loss(input=self.disp(), label=flow.tensor())
            loss.backward()
            optimizer.step()
            error = loss.detach()
            converged = error.less_equal(y=paddle.to_tensor(epsilon)).astype("bool").all()
            if verbose > 0 and (converged or step % verbose == 0):
                print(f"{type(self).__name__}.fit(): step={step}, mse={error.tolist()}")
            if converged:
                break

    def forward(self, points: paddle.Tensor, grid: bool = False) -> paddle.Tensor:
        r"""Transform normalized points by this spatial transformation.

        Args:
            points: Tensor of shape ``(N, M, D)`` or ``(N, ..., Y, X, D)``.
            grid: Whether ``points`` are the positions of undeformed grid points.

        Returns:
            Tensor of same shape as ``points`` with transformed point coordinates.

        """
        transform = self.tensor().to(points.place)
        apply = U.transform_grid if grid else U.transform_points
        align_corners = self.align_corners()
        return apply(transform, points, align_corners=align_corners)

    def points(
        self,
        points: paddle.Tensor,
        grid: Optional[Grid] = None,
        axes: Optional[Union[Axes, str]] = None,
        to_grid: Optional[Grid] = None,
        to_axes: Optional[Union[Axes, str]] = None,
    ) -> paddle.Tensor:
        r"""Transform points by this spatial transformation.

        Args:
            points: Tensor of shape ``(N, M, D)`` or ``(N, ..., Y, X, D)``.
            grid: Grid with respect to which input ``points`` are defined. Uses ``self.grid()`` if ``None``.
            axes: Coordinate axes with respect to which ``points`` are defined. Uses ``self.axes()`` if ``None``.
            to_grid: Grid with respect to which output points are defined. Same as ``grid`` if ``None``.
            to_axes: Coordinate axes to which ``points`` should be mapped to. Same as ``axes`` if ``None``.

        Returns:
            Point coordinates in ``(grid, axes)`` spatially transformed and mapped to coordinates with respect to ``(to_grid, to_axes)``.

        """
        if grid is None:
            grid = self.grid()
        if axes is None:
            axes = self.axes()
        else:
            axes = Axes.from_arg(axes)
        if to_grid is None:
            to_grid = grid
        if to_axes is None:
            to_axes = axes
        else:
            to_axes = Axes.from_arg(to_axes)
        points = grid.transform_points(
            points, axes=axes, to_grid=self.grid(), to_axes=self.axes(), decimals=None
        )
        points = self.forward(points)
        points = self.grid().transform_points(
            points, axes=self.axes(), to_grid=to_grid, to_axes=to_axes, decimals=None
        )
        return points

    def disp(self, grid: Optional[Grid] = None) -> paddle.Tensor:
        r"""Get displacement vector field representation of this transformation.

        Args:
            grid: Grid on which to sample vector fields. Use ``self.grid()`` if ``None``.

        Returns:
            Displacement vector fields as tensor of shape ``(N, D, ..., X)``.

        """
        if grid is None:
            grid = self.grid()
        data = self.tensor()
        if data.ndim < 3:
            raise AssertionError(
                f"SpatialTransform.disp() expected {type(self).__name__}.tensor() to be at least 3-dimensional"
            )
        if tuple(data.shape)[1] != self.dim():
            raise AssertionError(
                f"SpatialTransform.disp() expected {type(self).__name__}.tensor() shape[1] to be equal to {type(self).__name__}.dim()={self.dim()}"
            )
        # Affine transformation with shape:
        # - (N, D, 1): Translation only.
        # - (N, D, D): Affine only, i.e., no translation.
        # - (N, D, D + 1): Affine transformation, including translation.
        if data.ndim == 3:
            assert self.linear
            data = U.affine_flow(data, grid)
        # Non-rigid deformation tensor as displacement field with shape (N, D, ..., X)
        else:
            assert not self.linear
            align_corners = grid.align_corners()
            # Displacement field with domain different from output domain
            # - Use F.grid_sample() to resample displacement field and adjust vectors.
            if grid != self.grid() or align_corners != self.align_corners():
                flow = FlowFields(data, grid=self.grid().reshape(tuple(data.shape)[2:]))
                flow = flow.sample(grid)
                data = flow.tensor()
            # Displacement field with same domain as output grid, but differing size
            # - Use F.interpolate() to resize displacement field.
            elif tuple(grid.shape) != tuple(data.shape)[2:]:
                data = U.grid_reshape(data, tuple(grid.shape), align_corners=align_corners)
        return data

    @final
    def flow(self, grid: Optional[Grid] = None, device: Optional[Device] = None) -> FlowFields:
        r"""Get flow field representation of this transformation.

        Args:
            grid: Grid on which to sample flow fields. Use ``self.grid()`` if ``None``.
            device: Device on which to store returned flow fields.

        Returns:
            Flow fields defined by a tensor of shape ``(N, D, ..., X)`` and a common spatial grid.

        """
        if grid is None:
            grid = self.grid()
        data = self.disp(grid)
        return FlowFields(data, grid=grid, device=device)

    @abstractmethod
    def tensor(self) -> paddle.Tensor:
        r"""Get tensor representation of this transformation.

        The tensor representation of a transformation is with respect to the unit cube axes defined
        by its sampling grid as specified by ``self.axes()``. For a non-rigid transformation it is
        a displacement vector field. For linear transformations, it is a batch of homogeneous
        transformation tensors whose shape determines the type of linear transformation.

        Returns:
            Returns a batch of homogeneous transformation matrices as tensor of shape ``(N, D, 1)``
            (translation),  ``(N, D, D)`` (affine) or ``(N, D, D + 1)``, i.e., a 3-dimensional tensor,
            if this transformation is a :class:`.LinearTransform`. In case of a non-rigid transformation,
            a displacement vector field is returned as tensor of shape ``(N, D, ..., X)``, i.e., a
            higher dimensional tensor, where ``D = self.ndim`` and the number of tensor dimensions is
            equal to ``D + 2``.

        """
        raise NotImplementedError(f"{type(self).__name__}.tensor()")

    def update(self: TSpatialTransform) -> TSpatialTransform:
        r"""Update internal state of this transformation.

        This function is called by a pre-forward hook. It can be overriden by subclasses to
        update their internal state, e.g., to obtain current predictions of transformation
        parameters, to compute a dense vector field from spline coefficients, to compute
        a displacement field from a velocity field, etc. When calling other functions than
        the module's ``__call__`` function, the ``update()`` of the transformation must be
        called explicitly unless it is known that the specific transformation keeps no
        internal state other than its (optimizable) parameters.

        """
        return self

    @staticmethod
    def _update_hook(transform: paddle.nn.Layer, *args, **kwargs) -> None:
        r"""Update callback which is registered as pre-forward hook."""
        assert isinstance(transform, SpatialTransform)
        transform.update()

    def register_update_hook(self) -> None:
        r"""Register a forward pre-hook which invokes :meth:`.SpatialTransform.update`."""
        self._update_hook_handle = self.register_forward_pre_hook(hook=self._update_hook)

    def remove_update_hook(self) -> None:
        r"""Remove previously registered :meth:`.SpatialTransform.update` hook."""
        if self._update_hook_handle is not None:
            self._update_hook_handle.remove()
            self._update_hook_handle = None

    @property
    def inv(self: TSpatialTransform) -> TSpatialTransform:
        r"""Get inverse transformation.

        Convenience property for applying the inverse transformation, e.g.,

        .. code-block:: python

            y = transform(x)
            x = transform.inv(y)

        """
        return self.inverse(link=True, update_buffers=True)

    def inverse(
        self: TSpatialTransform, link: bool = False, update_buffers: bool = False
    ) -> TSpatialTransform:
        r"""Get inverse of this transformation.

        Args:
            link: Whether the inverse transformation keeps a reference to this transformation.
                If ``True``, the ``update()`` function of the inverse function will not recompute
                shared parameters, e.g., parameters obtained by a callable neural network, but
                directly access the parameters from this transformation. Note that when ``False``,
                the inverse transformation will still share parameters, modules, and buffers with
                this transformation, but these shared tensors may be replaced by a call of ``update()``
                (which is implicitly called as pre-forward hook when ``__call__()`` is invoked).
            update_buffers: Whether buffers of inverse transformation should be updated after creating
                the shallow copy. If ``False``, the ``update()`` function of the returned inverse
                transformation has to be called before it is used.

        Returns:
            Shallow copy of this transformation which computes and applies the inverse transformation.
            The inverse transformation will share the parameters with this transformation. Not all
            transformations may implement this functionality.

        Raises:
            NotImplementedError: Transformation does not support sharing parameters with its inverse.

        """
        raise NotImplementedError(f"{type(self).__name__}.inverse()")

    def clear_buffers(self: TSpatialTransform) -> TSpatialTransform:
        r"""Clear any buffers that are registered by ``self.update()``."""
        ...

    def extra_repr(self) -> str:
        return f"grid={repr(self.grid())}"


class LinearTransform(SpatialTransform):
    r"""Homogeneous coordinate transformation."""

    @overload
    def matrix(self) -> paddle.Tensor:
        r"""Get matrix representation of linear transformation."""
        ...

    @overload
    def matrix(self: TLinearTransform, arg: paddle.Tensor) -> TLinearTransform:
        r"""Get shallow copy of this transformation with parameters obtained from given matrix."""
        ...

    @final
    def matrix(
        self: TLinearTransform, arg: Optional[paddle.Tensor] = None
    ) -> Union[TLinearTransform, paddle.Tensor]:
        r"""Get matrix representation of linear transformation or shallow copy with parameters set from matrix."""
        if arg is None:
            return as_homogeneous_matrix(self.tensor())
        return shallow_copy(self).matrix_(arg)

    def matrix_(self: TLinearTransform, arg: paddle.Tensor) -> TLinearTransform:
        raise NotImplementedError(f"{type(self).__name__}.matrix_()")


class NonRigidTransform(SpatialTransform):
    r"""Base class for non-linear transformation models.

    All non-linear transformation models parameterize a dense displacement vector field, either a separate
    displacement for each image in an input batch (e.g., for groupwise registration, batched registration),
    or a single displacement field used to deform all images in an input batch. The parameterization, and
    thereby the set of optimizable parameters, is defined by subclasses. The ``tensor()`` function must be
    implemented by subclasses to evaluate the non-parametric dense displacement field given the current model
    parameters. The flow vectors must be with respect to the grid ``Axes.CUBE``, i.e., where coordinate
    -1 corresponds to the left edge of the unit cube with side length 2, and coordinate 1 the right edge of
    this unit cube, respectively. Note that this corresponds to option ``align_corners=False`` of
    ``torch.nn.functional.grid_sample``.

    """

    @final
    def tensor(self) -> paddle.Tensor:
        r"""Get tensor representation of this transformation.

        Returns:
            Batch of displacement vector fields as tensor of shape ``(N, D, ..., X)``.

        """
        u = getattr(self, "u", None)
        if u is None:
            u = getattr(self.update(), "u", None)
        if u is None or "u" not in {name for name, _ in self.named_buffers()}:
            raise AssertionError(
                f"{type(self).__name__}.update() required to register displacement vector field tensor as buffer named 'u'. See also NonRigidTransform.update() docstring."
            )
        if not isinstance(u, paddle.Tensor):
            raise AssertionError(f"{type(self).__name__}.tensor() 'u' must be tensor")
        if u.ndim != self.ndim + 2:
            raise AssertionError(
                f"{type(self).__name__}.tensor() 'u' must be {self.ndim + 2}-dimensional"
            )
        if tuple(u.shape)[1] != self.ndim:
            raise AssertionError(
                f"{type(self).__name__}.tensor() 'u' must have shape (N, {self.ndim}, ..., X)"
            )
        return u

    def update(self: TNonRigidTransform) -> TNonRigidTransform:
        r"""Update buffered vector fields.

        Required:
            ``u``: Displacement vector field representation of non-rigid transformation.

        Optional:
            ``v``: Velocity vector field representation of non-rigid transformation if applicable.
                When this buffer is set, it can be used in a regularization term to encourage smoothness
                or other desired properties on the (stationary) velocity field. Alternatively, a
                regularization term may be based directly on the optimizable parameters.

        Returns:
            Self reference to this updated transformation.

        """
        return self

    def clear_buffers(self: TNonRigidTransform) -> TNonRigidTransform:
        r"""Clear any buffers that are registered by ``self.update()``."""
        super().clear_buffers()
        for name in ("u", "v"):
            try:
                delattr(self, name)
            except AttributeError:
                pass
        return self
