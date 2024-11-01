r"""Mix-ins for spatial transformations that have (optimizable) parameters.

These `mix-ins <https://en.wikipedia.org/wiki/Mixin>`_ add property ``params`` to a
:class:`.SpatialTransform`, which can be either one of the following. In addition, functional
setter and getter functions are added, which check the type and shape of its arguments.

- ``None``: This spatial transformation has no required parameters set.
    This value can be specified when initializing a spatial transformation whose parameters
    will be set at a later time point, e.g., to the output of a neural network. An exception
    is raised by functions which attempt to access yet uninitialized transformation parameters.
- ``Parameter``: Tensor of optimizable parameters, e.g., for classic registration.
    To temporarily disable optimization of the parameters, set ``params.requires_grad = False``.
- ``Tensor``: Tensor of fixed non-optimizable parameters.
    These parameters are not returned by :meth:`.SpatialTransform.parameters`. This could be a
    tensor of spatial transformation parameters inferred by a neural network.
- ``Callable``: A callable such as a function or ``paddle.nn.Layer``.
    Method :meth:`.SpatialTransform.update`, which is registered as pre-forward hook for any spatial
    transformation, invokes this callable to obtain the current transformation parameters with arguments
    set and obtained by :meth:`.SpatialTransform.condition`. For example, an input batch of a neural
    network can be passed to a ``paddle.nn.Layer`` this way to infer parameters from this input.

"""
from __future__ import annotations  # noqa

from copy import copy as shallow_copy
from typing import Callable
from typing import Optional
from typing import Union
from typing import cast
from typing import overload

import paddle
from deepali.core.grid import Grid

from .base import ReadOnlyParameters
from .base import TSpatialTransform


class ParametricTransform:
    r"""Mix-in for spatial transformations that have (optimizable) parameters."""

    def __init__(
        self: Union[TSpatialTransform, ParametricTransform],
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable]] = True,
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
                called by ``self.update()`` with ``SpatialTransform.condition()`` arguments. When a boolean
                argument is given, a new zero-initialized tensor is created. If ``True``, this tensor is
                registered as optimizable module parameter. If ``None``, parameters must be set using
                ``self.data()`` or ``self.data_()`` before this transformation is evaluated.

        """
        if isinstance(params, paddle.Tensor) and params.ndim < 2:
            raise ValueError(
                f"{type(self).__name__}() 'params' tensor must be at least 2-dimensional"
            )
        super().__init__(grid)
        if groups is None:
            groups = tuple(params.shape)[0] if isinstance(params, paddle.Tensor) else 1
        shape = (groups,) + self.data_shape
        if params is None:
            self.params = None
        elif isinstance(params, bool):
            data = paddle.empty(shape=shape, dtype="float32")
            if params:
                self.params = paddle.create_parameter(
                    shape=data.shape,
                    dtype=data.dtype,
                    default_initializer=paddle.nn.initializer.Assign(data),
                )
                self.params.stop_gradient = False
            else:
                self.register_buffer(name="params", tensor=data, persistable=True)
            self.reset_parameters()
        elif isinstance(params, paddle.Tensor):
            if shape and tuple(params.shape) != shape:
                raise ValueError(
                    f"{type(self).__name__}() 'params' must be tensor of shape {shape!r}"
                )
            if isinstance(params, paddle.base.framework.Parameter):
                self.params = params
            else:
                self.register_buffer(name="params", tensor=params, persistable=True)
        elif callable(params):
            self.params = params
            self.register_buffer(name="p", tensor=paddle.empty(shape=shape), persistable=False)
            self.reset_parameters()
        else:
            raise TypeError(
                f"{type(self).__name__}() 'params' must be bool, Callable, Tensor, or None"
            )

    def has_parameters(self) -> bool:
        r"""Whether this transformation has optimizable parameters."""
        return isinstance(self.params, paddle.base.framework.EagerParamBase)

    @paddle.no_grad()
    def reset_parameters(self: Union[TSpatialTransform, ParametricTransform]) -> None:
        r"""Reset transformation parameters."""
        params = self.params  # Note: May be None!
        if params is None:
            return
        if callable(params):
            params = self.p
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(params)
        self.clear_buffers()

    @property
    def data_shape(self) -> list:
        r"""Get required shape of transformation parameters tensor, excluding batch dimension."""
        raise NotImplementedError(f"{type(self).__name__}.data_shape")

    @overload
    def data(self) -> paddle.Tensor:
        r"""Get (buffered) transformation parameters."""
        ...

    @overload
    def data(self: TSpatialTransform, arg: paddle.Tensor) -> TSpatialTransform:
        r"""Get shallow copy with specified parameters."""
        ...

    def data(
        self: Union[TSpatialTransform, ParametricTransform], arg: Optional[paddle.Tensor] = None
    ) -> Union[TSpatialTransform, paddle.Tensor]:
        r"""Get transformation parameters or shallow copy with specified parameters, respectively."""
        params = self.params  # Note: May be None!
        if arg is None:
            if params is None:
                raise AssertionError(f"{type(self).__name__}.data() 'params' must be set first")
            if callable(params):
                params = getattr(self, "p")
            return params
        if not isinstance(arg, paddle.Tensor):
            raise TypeError(f"{type(self).__name__}.data() 'arg' must be tensor")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"{type(self).__name__}.data() 'arg' must be {len(shape) + 1}-dimensional tensor"
            )
        shape = (arg.shape[0],) + shape
        if tuple(arg.shape) != tuple(shape):
            raise ValueError(f"{type(self).__name__}.data() 'arg' must have shape {shape!r}")
        copy = shallow_copy(self)
        if callable(params):
            delattr(copy, "p")
        if isinstance(params, paddle.base.framework.EagerParamBase) and not isinstance(
            arg, paddle.base.framework.EagerParamBase
        ):
            copy.params = paddle.create_parameter(
                shape=arg.shape,
                dtype=arg.dtype,
                default_initializer=paddle.nn.initializer.Assign(arg),
            )
            copy.params.stop_gradient = params.stop_gradient
        else:
            copy.params = arg
        copy.clear_buffers()
        return copy

    def data_(
        self: Union[TSpatialTransform, ParametricTransform], arg: paddle.Tensor
    ) -> TSpatialTransform:
        r"""Replace transformation parameters.

        Args:
            arg: Tensor of transformation parameters with shape matching ``self.data_shape``,
                excluding the batch dimension whose size may be different from the current tensor.

        Returns:
            Reference to this in-place modified transformation module.

        Raises:
            ReadOnlyParameters: When ``self.params`` is a callable which provides the parameters.

        """
        params = self.params  # Note: May be None!
        if callable(params):
            raise ReadOnlyParameters(
                f"Cannot replace parameters, try {type(self).__name__}.data() instead."
            )
        if not isinstance(arg, paddle.Tensor):
            raise TypeError(f"{type(self).__name__}.data_() 'arg' must be tensor, not {type(arg)}")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"{type(self).__name__}.data_() 'arg' must be {len(shape) + 1}-dimensional tensor, but arg.ndim={arg.ndim}"
            )
        shape = (arg.shape[0],) + shape
        if tuple(arg.shape) != tuple(shape):
            raise ValueError(
                f"{type(self).__name__}.data_() 'arg' must have shape {shape!r}, not {arg.shape!r}"
            )
        if isinstance(params, paddle.base.framework.EagerParamBase) and not isinstance(
            arg, paddle.base.framework.EagerParamBase
        ):
            self.params = paddle.create_parameter(
                shape=arg.shape,
                dtype=arg.dtype,
                default_initializer=paddle.nn.initializer.Assign(arg),
            )
            self.params.stop_gradient = params.stop_gradient
        else:
            self.params = arg
        self.clear_buffers()
        return self

    def _data(self: Union[TSpatialTransform, ParametricTransform]) -> paddle.Tensor:
        r"""Get most recent transformation parameters.

        When transformation parameters are obtained from a callable, this function invokes
        this callable with ``self.condition()`` as arguments if set, and returns the parameter
        obtained returned by this callable function or module. Otherwise, it simply returns a
        reference to the ``self.params`` tensor.

        Returns:
            Reference to ``self.params`` tensor or callable return value, respectively.

        """
        params = self.params
        if params is None:
            raise AssertionError(f"{type(self).__name__}._data() 'params' must be set first")
        if isinstance(params, type(self)):
            assert isinstance(params, ParametricTransform)
            return cast(ParametricTransform, params).data()
        if callable(params):
            args, kwargs = self.condition()
            pred = params(*args, **kwargs)
            if not isinstance(pred, paddle.Tensor):
                raise TypeError(f"{type(self).__name__}.params() value must be tensor")
            shape = self.data_shape
            if pred.ndim != len(shape) + 1:
                raise ValueError(
                    f"{type(self).__name__}.params() tensor must be {len(shape) + 1}-dimensional"
                )
            shape = (tuple(pred.shape)[0],) + shape
            if tuple(pred.shape) != shape:
                raise ValueError(f"{type(self).__name__}.params() tensor must have shape {shape!r}")
            return pred
        assert isinstance(params, paddle.Tensor)
        return params

    def link(
        self: Union[TSpatialTransform, ParametricTransform], other: TSpatialTransform
    ) -> TSpatialTransform:
        r"""Make shallow copy of this transformation which is linked to another instance."""
        return shallow_copy(self).link_(other)

    def link_(
        self: Union[TSpatialTransform, ParametricTransform],
        other: Union[TSpatialTransform, ParametricTransform],
    ) -> TSpatialTransform:
        r"""Link this transformation to another of the same type.

        This transformation is modified to use a reference to the given transformation. After linking,
        the transformation will not have parameters on its own, and its ``update()`` function will not
        recompute possibly previously shared parameters, e.g., parameters obtained by a callable neural
        network. Instead, it directly copies the parameters from the linked transformation.

        Args:
            other: Other transformation of the same type as ``self`` to which this transformation is linked.

        Returns:
            Reference to this transformation.

        """
        if other is self:
            raise ValueError(f"{type(self).__name__}.link() cannot link tranform to itself")
        if type(self) != type(other):
            raise TypeError(
                f"{type(self).__name__}.link() 'other' must be of the same type, got {type(other).__name__}"
            )
        self.params = other
        if not hasattr(self, "p"):
            if other.params is None:
                p = paddle.empty(shape=self.data_shape)
            else:
                p = other.data()
            self.register_buffer(name="p", tensor=p, persistable=False)
            if other.params is None:
                self.reset_parameters()
        return self

    def unlink(self: Union[TSpatialTransform, ParametricTransform]) -> TSpatialTransform:
        r"""Make a shallow copy of this transformation with parameters set to ``None``."""
        return shallow_copy(self).unlink_()

    def unlink_(self: Union[TSpatialTransform, ParametricTransform]) -> TSpatialTransform:
        r"""Resets transformation parameters to ``None``."""
        self.params = None
        if hasattr(self, "p"):
            delattr(self, "p")
        return self

    def update(self: Union[TSpatialTransform, ParametricTransform]) -> TSpatialTransform:
        r"""Update buffered data such as predicted parameters, velocities, and/or displacements."""
        if hasattr(self, "p"):
            p = self._data()
            self.register_buffer(name="p", tensor=p, persistable=False)
        super().update()
        return self


class InvertibleParametricTransform(ParametricTransform):
    r"""Mix-in for spatial transformations that support on-demand inversion."""

    def __init__(
        self,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        invert: bool = False,
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
                called by ``self.update()`` with ``self.condition()`` arguments. When a boolean argument is
                given, a new zero-initialized tensor is created. If ``True``, this tensor is registered as
                optimizable module parameter.
            invert: Whether ``params`` correspond to the inverse transformation. When this flag is ``True``,
                the ``self.tensor()`` and related methods return the transformation corresponding to the
                inverse of the transformations with the given ``params``. For example in case of a rotation,
                the rotation matrix is first constructed from the rotation parameters (e.g., Euler angles),
                and then transposed if ``self.invert == True``. In general, inversion of linear transformations
                and non-rigid transformations parameterized by velocity fields can be done efficiently on-the-fly.

        """
        super().__init__(grid, groups=groups, params=params)
        self.invert = bool(invert)

    def inverse(
        self: Union[TSpatialTransform, InvertibleParametricTransform],
        link: bool = False,
        update_buffers: bool = False,
    ) -> TSpatialTransform:
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
            Shallow copy of this transformation which computes and applies the inverse transformation.
            The inverse transformation will share the parameters with this transformation.

        """
        inv = shallow_copy(self)
        if link:
            inv.link_(self)
        inv.invert = not self.invert
        return inv

    def extra_repr(self: Union[TSpatialTransform, InvertibleParametricTransform]) -> str:
        r"""Print current transformation."""
        return super().extra_repr() + f", invert={self.invert}"
