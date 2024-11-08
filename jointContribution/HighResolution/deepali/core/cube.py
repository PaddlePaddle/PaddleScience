from __future__ import annotations

from copy import copy as shallow_copy
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union
from typing import overload

import numpy as np
import paddle

from .grid import ALIGN_CORNERS
from .grid import Axes
from .grid import Grid
from .linalg import hmm
from .linalg import homogeneous_matrix
from .linalg import homogeneous_transform
from .tensor import as_tensor
from .tensor import cat_scalars
from .types import Array
from .types import Device
from .types import DType
from .types import Shape
from .types import Size


class Cube(object):
    """Bounding box oriented in world space which defines a normalized domain.

    Coordinates of points within this domain can be either with respect to the world coordinate
    system or the cube defined by the bounding box where coordinate axes are parallel to the
    cube edges and have a uniform side length of 2. The latter are the normalized coordinates
    used by ``paddle.nn.functional.grid_sample()``, in particular. In terms of the coordinate
    transformations, a :class:`.Cube` is thus equivalent to a :class:`.Grid` with three points
    along each dimension and ``align_corners=True``.

    A regular sampling :class:`.Grid`, on the other hand, subsamples the world space within the bounds
    defined by the cube into a number of equally sized cells or equally spaced points, respectivey.
    How the grid points relate to the faces of the cube depends on :meth:`.Grid.align_corners`.

    """

    __slots__ = "_center", "_direction", "_extent"

    def __init__(
        self,
        extent: Optional[Union[Array, float]],
        center: Optional[Union[Array, float]] = None,
        origin: Optional[Union[Array, float]] = None,
        direction: Optional[Array] = None,
        device: Optional[Device] = None,
    ):
        """Initialize cube attributes.

        Args:
            extent: Extent ``(extent_x, ...)`` of the cube in world units.
            center: Cube center point ``(x, ...)`` in world space.
            origin: World coordinates ``(x, ...)`` of lower left corner.
            direction: Direction cosines defining orientation of cube in world space.
                The direction cosines are vectors that point along the cube edges.
                Each column of the matrix indicates the direction cosines of the unit vector
                that is parallel to the cube edge corresponding to that dimension.
            device: Device on which to store attributes. Uses ``"cpu"`` if ``None``.

        """
        extent = as_tensor(extent, device=device or "cpu")
        if not extent.is_floating_point():
            extent = extent.astype(dtype="float32")
        self._extent = extent
        if direction is None:
            direction = paddle.eye(num_rows=self.ndim, dtype=self.dtype)
        self.direction_(direction)
        if origin is None:
            self.center_(0 if center is None else center)
        elif center is None:
            self.origin_(origin)
        else:
            self.center_(center)
            if not paddle.allclose(x=origin, y=self.origin()).item():
                raise ValueError("Cube() 'center' and 'origin' are inconsistent")

    def numpy(self) -> np.ndarray:
        """Get cube attributes as 1-dimensional NumPy array."""
        return np.concatenate(
            [
                self._extent.numpy(),
                self._center.numpy(),
                self._direction.flatten().numpy(),
            ],
            axis=0,
        )

    @classmethod
    def from_numpy(
        cls, attrs: Union[Sequence[float], np.ndarray], origin: bool = False
    ) -> Cube:
        """Create Cube from 1-dimensional NumPy array."""
        if isinstance(attrs, np.ndarray):
            seq = attrs.astype(float).tolist()
        else:
            seq = attrs
        return cls.from_seq(seq, origin=origin)

    @classmethod
    def from_seq(cls, attrs: Sequence[float], origin: bool = False) -> Cube:
        """Create Cube from sequence of attribute values.

        Args:
            attrs: Array of length (D + 2) * D, where ``D=2`` or ``D=3`` is the number
                of spatial cube dimensions and array items are given as
                ``(sx, ..., cx, ..., d11, ..., d21, ....)``, where ``(sx, ...)`` is the
                cube extent, ``(cx, ...)`` the cube center coordinates, and ``(d11, ...)``
                are the cube direction cosines. The argument can be a Python list or tuple,
                NumPy array, or tensor.
            origin: Whether ``(cx, ...)`` specifies Cube origin rather than center.

        Returns:
            Cube instance.

        """
        if len(attrs) == 8:
            d = 2
        elif len(attrs) == 15:
            d = 3
        else:
            raise ValueError(
                f"{cls.__name__}.from_seq() expected array of length 8 (D=2) or 15 (D=3)"
            )
        kwargs = dict(extent=attrs[0:d], direction=attrs[2 * d :])
        if origin:
            kwargs["origin"] = attrs[d : 2 * d]
        else:
            kwargs["center"] = attrs[d : 2 * d]
        return Cube(**kwargs)

    @classmethod
    def from_grid(cls, grid: Grid, align_corners: Optional[bool] = None) -> Cube:
        """Get cube with respect to which normalized grid coordinates are defined."""
        if align_corners is not None:
            grid = grid.align_corners(align_corners)
        return cls(
            extent=grid.cube_extent(),
            center=grid.center(),
            direction=grid.direction(),
            device=grid.place,
        )

    def grid(
        self,
        size: Optional[Union[int, Size, Array]] = None,
        shape: Optional[Union[int, Shape, Array]] = None,
        spacing: Optional[Union[Array, float]] = None,
        align_corners: bool = ALIGN_CORNERS,
    ) -> Grid:
        """Create regular sampling grid which covers the world space bounded by the cube."""
        if size is None and shape is None:
            if spacing is None:
                raise ValueError(
                    "Cube.grid() requires either the desired grid 'size'/'shape' or point 'spacing'"
                )
            size = self.extent().div(spacing).round()
            size = tuple(size.astype("int32").tolist())
            if align_corners:
                size = tuple(n + 1 for n in size)
            spacing = None
        else:
            if isinstance(size, int):
                size = (size,) * self.ndim
            if isinstance(shape, int):
                shape = (shape,) * self.ndim
            size = Grid(size=size, shape=shape).size()
        ncells = paddle.to_tensor(data=size)
        if align_corners:
            ncells = ncells.subtract_(y=paddle.to_tensor(1))
        ncells = ncells.to(dtype=self.dtype, device=self.device)
        grid = Grid(
            size=size,
            spacing=self.extent().div(ncells),
            center=self.center(),
            direction=self.direction(),
            align_corners=align_corners,
            device=self.device,
        )
        with paddle.no_grad():
            if not paddle.allclose(x=grid.cube_extent(), y=self.extent()).item():
                raise ValueError(
                    "Cube.grid() 'size'/'shape' times 'spacing' does not match cube extent"
                )
        return grid

    def dim(self) -> int:
        """Number of cube dimensions."""
        return len(self._extent)

    @property
    def ndim(self) -> int:
        """Number of cube dimensions."""
        return len(self._extent)

    @property
    def dtype(self) -> DType:
        """Get data type of cube attribute tensors."""
        return self._extent.dtype

    @property
    def device(self) -> Device:
        """Get device on which cube attribute tensors are stored."""
        return self._extent.place

    def clone(self) -> Cube:
        """Make deep copy of this instance."""
        cube = shallow_copy(self)
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, paddle.Tensor):
                setattr(cube, name, value.clone())
        return cube

    def __deepcopy__(self, memo) -> Cube:
        """Support copy.deepcopy to clone this cube."""
        if id(self) in memo:
            return memo[id(self)]
        copy = self.clone()
        memo[id(self)] = copy
        return copy

    @overload
    def center(self) -> paddle.Tensor:
        """Get center point in world space."""
        ...

    @overload
    def center(self, arg: Union[float, Array], *args: float) -> Cube:
        """Get new cube with same orientation and extent, but specified center point."""
        ...

    def center(self, *args) -> Union[paddle.Tensor, Cube]:
        """Get center point in world space or new cube with specified center point."""
        if args:
            return shallow_copy(self).center_(*args)
        return self._center

    def center_(self, arg: Union[Array, float], *args: float) -> Cube:
        """Set center point in world space of this cube."""
        self._center = cat_scalars(
            arg, *args, num=self.ndim, dtype=self.dtype, device=self.device
        )
        return self

    @overload
    def origin(self) -> paddle.Tensor:
        """Get world coordinates of lower left corner."""
        ...

    @overload
    def origin(self, arg: Union[Array, float], *args: float) -> Cube:
        """Get new cube with specified world coordinates of lower left corner."""
        ...

    def origin(self, *args) -> Union[paddle.Tensor, Cube]:
        """Get origin in world space or new cube with specified origin."""
        if args:
            return shallow_copy(self).origin_(*args)
        offset = paddle.matmul(x=self.direction(), y=self.spacing())
        origin = self._center.sub(offset)
        return origin

    def origin_(self, arg: Union[Array, float], *args: float) -> Cube:
        """Set world coordinates of lower left corner."""
        center = cat_scalars(
            arg, *args, num=self.ndim, dtype=self.dtype, device=self.device
        )
        offset = paddle.matmul(x=self.direction(), y=self.spacing())
        self._center = center.add(offset)
        return self

    def spacing(self) -> paddle.Tensor:
        """Cube unit spacing in world space."""
        return self._extent.div(2)

    @overload
    def direction(self) -> paddle.Tensor:
        """Get edge direction cosines matrix."""
        ...

    @overload
    def direction(self, arg: Union[Array, float], *args: float) -> Cube:
        """Get new cube with specified edge direction cosines."""
        ...

    def direction(self, *args) -> Union[paddle.Tensor, Cube]:
        """Get edge direction cosines matrix or new cube with specified orientation."""
        if args:
            return shallow_copy(self).direction_(*args)
        return self._direction

    def direction_(self, arg: Union[Array, float], *args: float) -> Cube:
        """Set edge direction cosines matrix of this cube."""
        D = self.ndim
        if args:
            direction = paddle.to_tensor(data=(arg,) + args)
        else:
            direction = as_tensor(arg)
        direction = direction.to(dtype=self.dtype, device=self.device)
        if direction.ndim == 1:
            if tuple(direction.shape)[0] != D * D:
                raise ValueError(
                    f"Cube direction must be array or square matrix with numel={D * D}"
                )
            direction = direction.reshape(D, D)
        elif (
            direction.ndim != 2
            or tuple(direction.shape)[0] != tuple(direction.shape)[1]
            or tuple(direction.shape)[0] != D
        ):
            raise ValueError(
                f"Cube direction must be array or square matrix with numel={D * D}"
            )
        with paddle.no_grad():
            if abs(direction.det().abs().item() - 1) > 0.0001:
                raise ValueError(
                    "Cube direction cosines matrix must be valid rotation matrix"
                )
        self._direction = direction
        return self

    @overload
    def extent(self) -> paddle.Tensor:
        """Extent of cube in world space."""
        ...

    @overload
    def extent(self, arg: Union[float, Array], *args, float) -> Cube:
        """Get cube with same center and orientation but different extent."""
        ...

    def extent(self, *args) -> Union[paddle.Tensor, Cube]:
        """Get extent of this cube or a new cube with same center and orientation but specified extent."""
        if args:
            return shallow_copy(self).extent_(*args)
        return self._extent

    def extent_(self, arg: Union[Array, float], *args) -> Cube:
        """Set the extent of this cube, keeping center and orientation the same."""
        self._extent = cat_scalars(
            arg, *args, num=self.ndim, device=self.device, dtype=self.dtype
        )
        return self

    def affine(self) -> paddle.Tensor:
        """Affine transformation from cube to world space, excluding translation."""
        return paddle.mm(input=self.direction(), mat2=paddle.diag(x=self.spacing()))

    def inverse_affine(self) -> paddle.Tensor:
        """Affine transformation from world to cube space, excluding translation."""
        one = paddle.to_tensor(data=1, dtype=self.dtype, place=self.device)
        return paddle.mm(
            input=paddle.diag(x=one / self.spacing()), mat2=self.direction().t()
        )

    def transform(
        self,
        axes: Optional[Union[Axes, str]] = None,
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
        vectors: bool = False,
    ) -> paddle.Tensor:
        """Transformation of coordinates from this cube to another cube.

        Args:
            axes: Axes with respect to which input coordinates are defined.
                If ``None`` and also ``to_axes`` and ``to_cube`` is ``None``,
                returns the transform which maps from cube to world space.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.
            vectors: Whether transformation is used to rescale and reorient vectors.

        Returns:
            If ``vectors=False``, a homogeneous coordinate transformation of shape ``(D, D + 1)``.
            Otherwise, a square transformation matrix of shape ``(D, D)`` is returned.

        """
        if axes is None and to_axes is None and to_cube is None:
            return self.transform(Axes.CUBE, Axes.WORLD, vectors=vectors)
        if axes is None:
            raise ValueError(
                "Cube.transform() 'axes' required when 'to_axes' or 'to_cube' specified"
            )
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        if axes is Axes.GRID or to_axes is Axes.GRID:
            raise ValueError("Cube.transform() Axes.GRID is only valid for a Grid")
        if axes == to_axes and axes is Axes.CUBE_CORNERS:
            axes = to_axes = Axes.CUBE
        elif axes is Axes.CUBE_CORNERS and to_axes is Axes.WORLD:
            axes = Axes.CUBE
        elif axes is Axes.WORLD and to_axes is Axes.CUBE_CORNERS:
            to_axes = Axes.CUBE
        if axes is Axes.CUBE_CORNERS or to_axes is Axes.CUBE_CORNERS:
            raise ValueError(
                "Cube.transform() cannot map between Axes.CUBE and Axes.CUBE_CORNERS. Use Cube.grid().transform() instead."
            )
        if axes == to_axes and (
            axes is Axes.WORLD or to_cube is None or to_cube == self
        ):
            return paddle.eye(num_rows=self.ndim, dtype=self.dtype)
        if axes == to_axes:
            assert axes is Axes.CUBE
            cube_to_world = self.transform(Axes.CUBE, Axes.WORLD, vectors=vectors)
            world_to_cube = to_cube.transform(Axes.WORLD, Axes.CUBE, vectors=vectors)
            if vectors:
                return paddle.mm(input=world_to_cube, mat2=cube_to_world)
            return hmm(world_to_cube, cube_to_world)
        if axes is Axes.CUBE:
            assert to_axes is Axes.WORLD
            if vectors:
                return self.affine()
            return homogeneous_matrix(self.affine(), self.center())
        assert axes is Axes.WORLD
        assert to_axes is Axes.CUBE
        if vectors:
            return self.inverse_affine()
        return hmm(self.inverse_affine(), -self.center())

    def inverse_transform(self, vectors: bool = False) -> paddle.Tensor:
        """Transform which maps from world to cube space."""
        return self.transform(Axes.WORLD, Axes.CUBE, vectors=vectors)

    def apply_transform(
        self,
        arg: Array,
        axes: Union[Axes, str],
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
        vectors: bool = False,
    ) -> paddle.Tensor:
        """Map point coordinates or displacement vectors from one cube to another.

        Args:
            arg: Coordinates of points or displacement vectors as tensor of shape ``(..., D)``.
            axes: Axes of this cube with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.
            vectors: Whether ``arg`` contains displacements (``True``) or point coordinates (``False``).

        Returns:
            Points or displacements with respect to ``to_cube`` and ``to_axes``.
            If ``to_cube == self`` and ``to_axes == axes`` or both ``axes`` and ``to_axes`` are
            ``Axes.WORLD`` and ``arg`` is a ``paddle.Tensor``, a reference to the unmodified input
            tensor is returned.

        """
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        if to_cube is None:
            to_cube = self
        tensor = as_tensor(arg)
        if not tensor.is_floating_point():
            tensor = tensor.astype(self.dtype)
        if axes is to_axes and axes is Axes.WORLD:
            return tensor
        if to_cube is not None and to_cube != self or axes is not to_axes:
            matrix = self.transform(axes, to_axes, to_cube=to_cube, vectors=vectors)
            matrix = matrix.unsqueeze(axis=0).to(device=tensor.place)
            tensor = homogeneous_transform(matrix, tensor)
        return tensor

    def transform_points(
        self,
        points: Array,
        axes: Union[Axes, str],
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
    ) -> paddle.Tensor:
        """Map point coordinates from one cube to another.

        Args:
            points: Coordinates of points to transform as tensor of shape ``(..., D)``.
            axes: Axes of this cube with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.

        Returns:
            Point coordinates with respect to ``to_cube`` and ``to_axes``. If ``to_cube == self``
            and ``to_axes == axes`` or both ``axes`` and ``to_axes`` are ``Axes.WORLD`` and ``arg``
            is a ``paddle.Tensor``, a reference to the unmodified input tensor is returned.

        """
        return self.apply_transform(
            points, axes, to_axes, to_cube=to_cube, vectors=False
        )

    def transform_vectors(
        self,
        vectors: Array,
        axes: Union[Axes, str],
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
    ) -> paddle.Tensor:
        """Rescale and reorient flow vectors.

        Args:
            vectors: Displacement vectors as tensor of shape ``(..., D)``.
            axes: Axes of this cube with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.

        Returns:
            Rescaled and reoriented displacement vectors. If ``to_cube == self`` and
            ``to_axes == axes`` or both ``axes`` and ``to_axes`` are ``Axes.WORLD`` and ``arg``
            is a ``paddle.Tensor``, a reference to the unmodified input tensor is returned.

        """
        return self.apply_transform(
            vectors, axes, to_axes, to_cube=to_cube, vectors=True
        )

    def cube_to_world(self, coords: Array) -> paddle.Tensor:
        """Map point coordinates from cube to world space.

        Args:
            coords: Normalized coordinates with respect to this cube as tensor of shape ``(..., D)``.

        Returns:
            Coordinates of points in world space.

        """
        return self.apply_transform(coords, Axes.CUBE, Axes.WORLD, vectors=False)

    def world_to_cube(self, points: Array) -> paddle.Tensor:
        """Map point coordinates from world to cube space.

        Args:
            points: Coordinates of points in world space as tensor of shape ``(..., D)``.

        Returns:
            Normalized coordinates of points with respect to this cube.

        """
        return self.apply_transform(points, Axes.WORLD, Axes.CUBE, vectors=False)

    def __eq__(self, other: Any) -> bool:
        """Compare this cube to another."""
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        for name in self.__slots__:
            value = getattr(self, name)
            other_value = getattr(other, name)
            if type(value) != type(other_value):
                return False
            if isinstance(value, paddle.Tensor):
                assert isinstance(other_value, paddle.Tensor)
                if tuple(value.shape) != tuple(other_value.shape):
                    return False
                other_value = other_value.to(device=value.place)
                if not paddle.allclose(
                    x=value, y=other_value, rtol=1e-05, atol=1e-08
                ).item():
                    return False
            elif value != other_value:
                return False
        return True

    def __repr__(self) -> str:
        """String representation."""
        origin = ", ".join([f"{v:.5f}" for v in self.origin()])
        center = ", ".join([f"{v:.5f}" for v in self._center])
        direction = ", ".join([f"{v:.5f}" for v in self._direction.flatten()])
        extent = ", ".join([f"{v:.5f}" for v in self._extent])
        return (
            f"{type(self).__name__}("
            + f"origin=({origin})"
            + f", center=({center})"
            + f", extent=({extent})"
            + f", direction=({direction})"
            + f", device={repr(str(self.device))}"
            + ")"
        )


def cube_points_transform(
    cube: Cube, axes: Axes, to_cube: Cube, to_axes: Optional[Axes] = None
):
    """Get linear transformation of points from one cube to another.

    Args:
        cube: Sampling grid with respect to which input points are defined.
        axes: Grid axes with respect to which input points are defined.
        to_cube: Sampling grid with respect to which output points are defined.
        to_axes: Grid axes with respect to which output points are defined.

    Returns:
        Homogeneous coordinate transformation matrix as tensor of shape ``(D, D + 1)``.

    """
    return cube.transform(axes=axes, to_axes=to_axes, to_cube=to_cube, vectors=False)


def cube_vectors_transform(
    cube: Cube, axes: Axes, to_cube: Cube, to_axes: Optional[Axes] = None
):
    """Get affine transformation which maps vectors with respect to one cube to another.

    Args:
        cube: Cube with respect to which (normalized) input vectors are defined.
        axes: Cube axes with respect to which input vectors are defined.
        to_cube: Cube with respect to which (normalized) output vectors are defined.
        to_axes: Cube axes with respect to which output vectors are defined.

    Returns:
        Affine transformation matrix as tensor of shape ``(D, D)``.

    """
    return cube.transform(axes=axes, to_axes=to_axes, to_cube=to_cube, vectors=True)


def cube_transform_points(
    points: paddle.Tensor,
    cube: Cube,
    axes: Axes,
    to_cube: Cube,
    to_axes: Optional[Axes] = None,
):
    return cube.transform_points(points, axes=axes, to_axes=to_axes, to_cube=to_cube)


def cube_transform_vectors(
    vectors: paddle.Tensor,
    cube: Cube,
    axes: Axes,
    to_cube: Cube,
    to_axes: Optional[Axes] = None,
):
    return cube.transform_vectors(vectors, axes=axes, to_axes=to_axes, to_cube=to_cube)
