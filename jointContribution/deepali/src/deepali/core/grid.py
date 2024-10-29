r"""Oriented regularly spaced data sampling grid."""

from __future__ import annotations  # noqa

from copy import copy as shallow_copy
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
from typing import overload

import numpy as np
from pkg_resources import parse_version

try:
    import SimpleITK as _sitk
except ImportError:
    _sitk = None
if TYPE_CHECKING:
    # At runtime, cyclical import resolved by Grid.cube() instead
    from .cube import Cube

import paddle
from deepali.utils import paddle_aux  # noqa

from .enum import SpatialDim
from .enum import SpatialDimArg
from .linalg import hmm
from .linalg import homogeneous_matrix
from .linalg import homogeneous_transform
from .math import round_decimals
from .tensor import as_tensor
from .tensor import cat_scalars
from .typing import Array
from .typing import Device
from .typing import DType
from .typing import PathStr
from .typing import ScalarOrTuple
from .typing import Shape
from .typing import Size
from .typing import is_int_dtype

ALIGN_CORNERS = True
r"""By default, grid corner points define the domain within which the data is defined.

The normalized coordinates with respect to the grid cube are thus in [-1, 1] between
the first and last grid points (grid cell center points). This is such that when a
grid is downsampled or upsampled, respectively, the grid points at the boundary of
the domain remain unchanged. Points are only inserted or removed between these.

"""


class Axes(Enum):
    r"""Enumeration of grid axes with respect to which grid coordinates are defined."""

    GRID = "grid"
    r"""Oriented along grid axes with units corresponding to voxel units with origin
       with respect to world space at grid/image point with zero indices."""
    CUBE = "cube"
    r"""Oriented along grid axes with units corresponding to unit cube :math:`[-1, 1]^D`
       with origin with respect to world space at grid/image center and extrema -1/1
       coinciding with the grid border (``align_corners=False``)."""
    CUBE_CORNERS = "cube_corners"
    r"""Oriented along grid axes with units corresponding to unit cube :math:`[-1, 1]^D`
       with origin with respect to world space at grid/image center and extrema -1/1
       coinciding with the grid corner points (``align_corners=True``)."""
    WORLD = "world"
    r"""Oriented along world axes (physical space) with units corresponding to grid spacing (mm)."""

    @classmethod
    def from_arg(cls, arg: Union[Axes, str, None]) -> Axes:
        r"""Create enumeration value from function argument."""
        if arg is None or arg == "default":
            return cls.CUBE_CORNERS if ALIGN_CORNERS else cls.CUBE
        return cls(arg)

    @classmethod
    def from_grid(cls, grid: Grid) -> Axes:
        r"""Create enumeration value from sampling grid object."""
        return cls.from_align_corners(grid.align_corners())

    @classmethod
    def from_align_corners(cls, align_corners: bool) -> Axes:
        r"""Create enumeration value from sampling grid object."""
        return cls.CUBE_CORNERS if align_corners else cls.CUBE


class Grid(object):
    r"""Oriented regularly spaced data sampling grid.

    The dimensions of :attr:`.Grid.shape` are in reverse order of the dimensions of :meth:`.Grid.size`.
    The latter is consistent with SimpleITK, and the order of coordinates of :meth:`.Grid.origin`,
    :meth:`.Grid.spacing`, and :meth:`.Grid.direction`. Property :attr:`.Grid.shape`, on the other hand,
    is consistent with the order of dimensions of an image data tensor of type ``paddle.Tensor``.

    To not confuse :meth:`.Grid.size` with ``paddle.Tensor.size()``, it is recommended to prefer
    property ``paddle.Tensor.shape``. The ``shape`` property is also known from ``numpy.ndarray.shape``.

    A :class:`.Grid` instance stores the grid center point instead of the origin corresponding to the grid
    point with zero index along each grid dimension. This simplifies resizing and resampling operations,
    which do not need to modify the origin explicitly, but keep the center point fixed. To get the
    coordinates of the grid origin, use :meth:`.Grid.origin`. For convenience, the :class:`.Grid`
    initialization function also accepts an ``origin`` instead of a ``center`` point as keyword argument.
    Conversion between center point and origin are taken care internally. When both ``origin`` and ``center``
    are specified, an error is raised if these are inconsistent with one another.

    In addition, :meth:`.Grid.points`, :meth:`.Grid.transform`, and :meth`.Grid.apply_transform` support
    coordinates with respect to different axes: 1) (continuous) grid indices, 2) world space, and
    3) grid-aligned cube with side length 2. The latter, i.e., :attr:`.Axes.CUBE` or :attr:`.Axes.CUBE_CORNERS`
    makes coordinates independent of :meth:`.Grid.size` and :meth:`.Grid.spacing`. These normalized coordinates
    are furthermore compatible with the ``grid`` argument of ``paddle.nn.functional.grid_sample()``. Use
    :meth:`.Grid.cube` to obtain a :class:`.Cube` defining the data domain without spatial sampling attributes.

    """

    __slots__ = (
        "_size",
        "_center",
        "_spacing",
        "_direction",
        "_align_corners",
    )

    def __init__(
        self,
        size: Optional[Union[Size, Array]] = None,
        shape: Optional[Union[Shape, Array]] = None,
        center: Optional[Union[Array, float]] = None,
        origin: Optional[Union[Array, float]] = None,
        spacing: Optional[Union[Array, float]] = None,
        direction: Optional[Array] = None,
        device: Optional[Device] = None,
        align_corners: bool = ALIGN_CORNERS,
    ):
        r"""Initialize sampling grid attributes.

        Args:
            size: Size of spatial grid dimensions in the order ``(X, ...)``.
            shape: Size of spatial grid dimensions in the order ``(..., X)``.
                Must be ``None`` or ``size`` reversed, if ``size is not None``.
                Either ``size`` or ``shape`` must be specified.
            center: Grid center point in world space.
            origin: World coordinates of grid point with zero indices.
            spacing: Size of each grid square along each dimension.
            direction: Direction cosines defining orientation of grid in world space.
                The direction cosines are vectors that point from one pixel to the next.
                Each column of the matrix indicates the direction cosines of the unit vector
                that is parallel to the lines of the grid corresponding to that dimension.
            device: Device on which to store grid attributes. If ``None``, use ``"cpu"``.
            align_corners: Whether position of grid corner points are preserved by grid
                resampling and resizing operations by default. If ``True``, the grid
                origin remains unchanged by grid resizing operations, but the grid extent
                may in total change by one times the spacing between grid points. If ``False``,
                the extent of the grid remains constant, but the grid origin may shift.

        """
        if device is None:
            device = paddle.CPUPlace()
        # Grid size, optionally given as data tensor shape
        if size is not None:
            size = as_tensor(size, device=device)
            if size.ndim != 1:
                raise ValueError("Grid() 'size' must be 1-dimensional array")
            if len(size) == 0:
                raise ValueError("Grid() 'size' must be non-empty array")
        if shape is None:
            if size is None:
                raise ValueError("Grid() 'size' or 'shape' required")
        else:
            shape = as_tensor(shape, device=device)
            if size is None:
                if shape.ndim != 1:
                    raise ValueError("Grid() 'shape' must be 1-dimensional array")
                if len(shape) == 0:
                    raise ValueError("Grid() 'shape' must be non-empty array")
                size = shape.flip(axis=0)
            else:
                with paddle.no_grad():
                    if (
                        len(size) != len(shape)
                        or shape.flip(axis=0)
                        .not_equal(y=paddle.to_tensor(size))
                        .astype("bool")
                        .any()
                    ):
                        raise ValueError("Grid() 'size' and 'shape' are not compatible")
        # Store size as float such that ``grid.downsample().upsample() == grid```
        self._size = paddle.clip(x=size.astype(dtype="float32"), min=0)
        # Set other properties AFTER _size, which defines 'device', 'dim' properties.
        # Use in-place setters to take care of conversion and value assertions.
        self.spacing_(1 if spacing is None else spacing)
        if direction is None:
            direction = paddle.eye(num_rows=self.ndim)
        self.direction_(direction)
        # Set grid center to default, specified center point, or derived from origin
        if origin is None:
            self.center_(0 if center is None else center)
        elif center is None:
            # ATTENTION: This must be done AFTER size, spacing, and direction are set!
            self.origin_(origin)
        else:
            self.center_(center)
            with paddle.no_grad():
                origin = cat_scalars(origin, num=self.ndim, dtype=self.dtype, device=self.device)
                if not paddle.allclose(x=origin, y=self.origin()).item():
                    raise ValueError("Grid() 'center' and 'origin' are inconsistent")
        # Default align_corners option argument for grid resizing operations
        self._align_corners = bool(align_corners)

    def numpy(self) -> np.ndarray:
        r"""Get grid attributes as 1-dimensional NumPy array."""
        return np.concatenate(
            [
                self._size.numpy(),
                self._spacing.numpy(),
                self._center.numpy(),
                self._direction.flatten().numpy(),
            ],
            axis=0,
        )

    @classmethod
    def from_numpy(
        cls,
        attrs: Union[Sequence[float], np.ndarray],
        origin: bool = False,
        align_corners: bool = ALIGN_CORNERS,
    ) -> Grid:
        r"""Create Grid from 1-dimensional NumPy array."""
        if isinstance(attrs, np.ndarray):
            seq = attrs.astype(float).tolist()
        else:
            seq = attrs
        return cls.from_seq(seq, origin=origin, align_corners=align_corners)

    @classmethod
    def from_seq(
        cls, attrs: Sequence[float], origin: bool = False, align_corners: bool = ALIGN_CORNERS
    ) -> Grid:
        r"""Create Grid from sequence of attribute values.

        Args:
            attrs: Array of length (D + 3) * D, where ``D=2`` or ``D=3`` is the number
                of spatial grid dimensions and array items are given as
                ``(nx, ..., sx, ..., cx, ..., d11, ..., d21, ....)``,
                where ``(nx, ...)`` is the grid size, ``(sx, ...)`` the grid spacing,
                ``(cx, ...)`` the grid center coordinates, and ``(d11, ...)``
                are the grid direction cosines. The argument can be a Python
                list or tuple, NumPy array, or PyTorch tensor.
            origin: Whether ``(cx, ...)`` specifies Grid origin rather than center.

        Returns:
            Grid instance.

        """
        if len(attrs) == 10:
            d = 2
        elif len(attrs) == 18:
            d = 3
        else:
            raise ValueError(
                f"{cls.__name__}.from_seq() expected array of length 10 (D=2) or 18 (D=3)"
            )
        kwargs = dict(
            size=attrs[0:d],
            spacing=attrs[d : 2 * d],
            direction=attrs[3 * d :],
            align_corners=align_corners,
        )
        if origin:
            kwargs["origin"] = attrs[2 * d : 3 * d]
        else:
            kwargs["center"] = attrs[2 * d : 3 * d]
        return Grid(**kwargs)

    @classmethod
    def from_batch(cls, tensor: paddle.Tensor) -> Grid:
        r"""Create default grid from (image) batch tensor.

        Args:
            tensor: Batch tensor of shape ``(N, C, ..., X)``.

        Returns:
            New default grid with size ``(X, ...)``.

        """
        return cls(shape=tuple(tensor.shape)[2:])

    @classmethod
    def from_file(cls, path: PathStr, align_corners: bool = ALIGN_CORNERS) -> Grid:
        r"""Create sampling grid from image file header information."""
        if _sitk is None:
            raise RuntimeError(f"{cls.__name__}.from_file() requires SimpleITK")
        reader = _sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return cls.from_reader(reader, align_corners=align_corners)

    @classmethod
    def from_reader(
        cls, reader: "_sitk.ImageFileReader", align_corners: bool = ALIGN_CORNERS  # type: ignore
    ) -> Grid:
        r"""Create sampling grid from image file reader attributes."""
        return cls(
            size=reader.GetSize(),
            origin=reader.GetOrigin(),
            spacing=reader.GetSpacing(),
            direction=reader.GetDirection(),
            align_corners=align_corners,
        )

    @classmethod
    def from_sitk(cls, image: "_sitk.Image", align_corners: bool = ALIGN_CORNERS) -> Grid:
        r"""Create sampling grid from ``SimpleITK.Image`` attributes."""
        return cls(
            size=image.GetSize(),
            origin=image.GetOrigin(),
            spacing=image.GetSpacing(),
            direction=image.GetDirection(),
            align_corners=align_corners,
        )

    def cube(self) -> "Cube":
        r"""Get oriented cube defining the space of normalized coordinates."""
        # Import locally due to cyclical dependency between Cube and Grid
        from .cube import Cube

        return Cube(
            extent=self.cube_extent(),
            center=self.center(),
            direction=self.direction(),
            device=self.device,
        )

    def domain(self) -> "Cube":
        r"""Get oriented bounding box defining the sampling grid domain in world space."""
        return self.cube()

    def dim(self) -> int:
        r"""Number of spatial grid dimensions."""
        return len(self._size)

    @property
    def ndim(self) -> int:
        r"""Number of grid dimensions."""
        return len(self._size)

    @property
    def dtype(self) -> paddle.dtype:
        r"""Get data type of grid attribute tensors."""
        return self._size.dtype

    @property
    def device(self) -> (paddle.CPUPlace, paddle.CUDAPlace, str):
        r"""Get device on which grid attribute tensors are stored."""
        return self._size.place

    def clone(self) -> Grid:
        r"""Make deep copy of this ``Grid`` instance."""
        grid = shallow_copy(self)
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, paddle.Tensor):
                setattr(grid, name, value.clone())
        return grid

    def __deepcopy__(self, memo) -> Grid:
        r"""Support copy.deepcopy to clone this grid."""
        if id(self) in memo:
            return memo[id(self)]
        copy = self.clone()
        memo[id(self)] = copy
        return copy

    @overload
    def align_corners(self) -> bool:
        r"""Whether resizing operations preserve grid extent (False) or corner points (True)."""
        ...

    @overload
    def align_corners(self, arg: bool) -> Grid:
        r"""Set if resizing operations preserve grid extent (False) or corner points (True)."""
        ...

    def align_corners(self, arg: Optional[bool] = None) -> Union[bool, Grid]:
        r"""Whether resizing operations preserve grid extent (False) or corner points (True)."""
        if arg is None:
            return self._align_corners
        return shallow_copy(self).align_corners_(arg)

    def align_corners_(self, arg: bool) -> Grid:
        r"""Set if resizing operations preserve grid extent (False) or corner points (True)."""
        self._align_corners = bool(arg)
        return self

    def axes(self) -> Axes:
        r"""Grid axes."""
        return Axes.from_grid(self)

    def numel(self) -> int:
        r"""Number of grid points."""
        return self.size().size

    @staticmethod
    def _round_size(size: paddle.Tensor) -> paddle.Tensor:
        r"""Round sampling grid size attribute."""
        zero = paddle.to_tensor(data=0, dtype=size.dtype, place=size.place)
        return paddle.where(condition=size.equal(y=zero), x=zero, y=size.ceil())

    def size_tensor(self) -> paddle.Tensor:
        r"""Sampling grid size as floating point tensor."""
        return self._round_size(self._size)

    @overload
    def size(self, i: int) -> int:
        r"""Sampleing grid size along the specified spatial dimension."""
        ...

    @overload
    def size(self) -> list:
        r"""Sampling grid size for dimensions ordered as ``(X, ...)``."""
        ...

    def size(self, i: Optional[int] = None) -> Union[int, list]:
        r"""Sampling grid size."""
        size = self.size_tensor()
        if i is None:
            return tuple(int(n) for n in size)
        return int(size[i])

    @property
    def shape(self) -> list:
        r"""Sampling grid size for dimensions ordered as ``(..., X)``."""
        return tuple(int(n) for n in self.size_tensor().flip(axis=0))

    def extent(self, i: Optional[int] = None) -> paddle.Tensor:
        r"""Extent of sampling grid in physical world units."""
        if i is None:
            return self.spacing() * self.size_tensor()
        return self._spacing[i] * self.size(i)

    def cube_extent(self, i: Optional[int] = None) -> paddle.Tensor:
        r"""Extent of sampling grid cube in physical world units."""
        if i is None:
            n = self.size_tensor()
            if self._align_corners:
                n = n.sub(1)
            return self.spacing().mul(n)
        n = self.size(i)
        if self._align_corners:
            n -= 1
        return self._spacing[i] * n

    @overload
    def center(self) -> paddle.Tensor:
        r"""Get grid center point in world space."""
        ...

    @overload
    def center(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Get new grid with specified center point in world space."""
        ...

    def center(
        self, arg: Union[float, Array, None] = None, *args: float
    ) -> Union[paddle.Tensor, Grid]:
        r"""Get center point in world space or new grid with specified center point, respectively."""
        if arg is None:
            if args:
                raise ValueError(
                    f"{type(self).__name__}.center() 'args' cannot be given when first 'arg' is None"
                )
            return self._center
        return shallow_copy(self).center_(arg, *args)

    def center_(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Set grid center point in world space."""
        self._center = cat_scalars(arg, *args, num=self.ndim, dtype=self.dtype, device=self.device)
        return self

    @overload
    def origin(self) -> paddle.Tensor:
        r"""Get world coordinates of grid point with index zero."""
        ...

    @overload
    def origin(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Get new grid with specified world coordinates of grid point at index zero."""
        ...

    def origin(
        self, arg: Union[float, Array, None] = None, *args: float
    ) -> Union[paddle.Tensor, Grid]:
        r"""Get grid origin in world space or new grid with specified origin, respectively."""
        if arg is None:
            if args:
                raise ValueError(
                    f"{type(self).__name__}.origin() 'args' cannot be given when first 'arg' is None"
                )
            size = self.size_tensor()
            offset = paddle.where(
                condition=size.greater_than(y=paddle.to_tensor(0)), x=size.sub(1), y=size
            ).div(2)
            offset = paddle.matmul(x=self.affine(), y=offset)
            return self._center.sub(offset)
        return shallow_copy(self).origin_(arg, *args)

    def origin_(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Set world coordinates of grid point with zero index."""
        origin = cat_scalars(arg, *args, num=self.ndim, dtype=self.dtype, device=self.device)
        size = self.size_tensor()
        offset = paddle.where(
            condition=size.greater_than(y=paddle.to_tensor(0)), x=size.sub(1), y=size
        ).div(2)
        offset = paddle.matmul(x=self.affine(), y=offset)
        self._center = origin.add(offset)
        return self

    @overload
    def spacing(self) -> paddle.Tensor:
        r"""Get spacing between grid points in world units."""
        ...

    @overload
    def spacing(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Get new grid with specified spacing between grid points in world units."""
        ...

    def spacing(
        self, arg: Union[float, Array, None] = None, *args: float
    ) -> Union[paddle.Tensor, Grid]:
        r"""Get spacing between grid points in world units or new grid with specified spacing, respectively."""
        if arg is None:
            if args:
                raise ValueError(
                    f"{type(self).__name__}.spacing() 'args' cannot be given when first 'arg' is None"
                )
            return self._spacing
        return shallow_copy(self).spacing_(arg, *args)

    def spacing_(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Set spacing between grid points in physical world units."""
        spacing = cat_scalars(arg, *args, num=self.ndim, dtype=self.dtype, device=self.device)
        if spacing.less_equal(y=paddle.to_tensor(0)).astype("bool").any():
            raise ValueError("Grid spacing must be positive")
        self._spacing = spacing
        return self

    @overload
    def direction(self) -> paddle.Tensor:
        r"""Get grid axes direction cosines matrix."""
        ...

    @overload
    def direction(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Get new grid with specified axes direction cosines."""
        ...

    def direction(
        self, arg: Union[float, Array, None] = None, *args: float
    ) -> Union[paddle.Tensor, Grid]:
        r"""Get grid axes direction cosines matrix or new grid with specified direction, respectively."""
        if arg is None:
            if args:
                raise ValueError(
                    f"{type(self).__name__}.direction() 'args' cannot be given when first 'arg' is None"
                )
            return self._direction
        return shallow_copy(self).direction_(arg, *args)

    def direction_(self, arg: Union[float, Array], *args: float) -> Grid:
        r"""Set grid axes direction cosines matrix of this grid."""
        D = self.ndim
        direction = paddle.to_tensor(data=(arg,) + args) if args else as_tensor(arg)
        direction = direction.to(dtype=self.dtype, device=self.device)
        if direction.ndim == 1:
            if tuple(direction.shape)[0] != D * D:
                raise ValueError(
                    f"Grid direction must be array or square matrix with numel={D * D}"
                )
            direction = direction.reshape(D, D)
        elif (
            direction.ndim != 2
            or tuple(direction.shape)[0] != tuple(direction.shape)[1]
            or tuple(direction.shape)[0] != D
        ):
            raise ValueError(f"Grid direction must be array or square matrix with numel={D * D}")
        with paddle.no_grad():
            if abs(paddle.linalg.det(direction).abs().item() - 1) > 0.0001:
                raise ValueError("Grid direction cosines matrix must be valid rotation matrix")
        self._direction = direction
        return self

    def affine(self) -> paddle.Tensor:
        r"""Affine transformation from ``Axes.GRID`` to ``Axes.WORLD``, excluding translation of origin."""
        return paddle.mm(input=self.direction(), mat2=paddle.diag(x=self.spacing()))

    def inverse_affine(self) -> paddle.Tensor:
        r"""Affine transformation from ``Axes.WORLD`` to ``Axes.GRID``, excluding translation of origin."""
        one = paddle.to_tensor(data=1, dtype=self.dtype, place=self.device)
        return paddle.mm(input=paddle.diag(x=one / self.spacing()), mat2=self.direction().t())

    def transform(
        self,
        axes: Optional[Union[Axes, str]] = None,
        to_axes: Optional[Union[Axes, str]] = None,
        to_grid: Optional[Grid] = None,
        vectors: bool = False,
    ) -> paddle.Tensor:
        r"""Transformation from one grid domain to another.

        Args:
            axes: Axes with respect to which input coordinates are defined.
                If ``None`` and also ``to_axes`` and ``to_cube`` is ``None``,
                returns the transform which maps from cube to world space.
            to_axes: Axes of grid to which coordinates are mapped. Use ``axes`` if ``None``.
            to_grid: Other grid. Use ``self`` if ``None``.
            vectors: Whether transformation is used to rescale and reorient vectors.

        Returns:
            If ``vectors=False``, a homogeneous coordinate transformation of shape ``(D, D + 1)``.
            Otherwise, a square transformation matrix of shape ``(D, D)`` is returned.

        """
        if axes is None and to_axes is None and to_grid is None:
            cube_axes = Axes.CUBE_CORNERS if self._align_corners else Axes.CUBE
            return self.transform(cube_axes, Axes.WORLD, vectors=vectors)
        if axes is None:
            raise ValueError(
                "Grid.transform() 'axes' required when 'to_axes' or 'to_grid' specified"
            )
        matrix = None
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        if to_grid is None or to_grid == self:
            if axes is to_axes:
                matrix = paddle.eye(num_rows=self.ndim, dtype=self.dtype)
                if not vectors:
                    offset = paddle.zeros(shape=self.ndim, dtype=self.dtype)
                    matrix = homogeneous_matrix(matrix, offset=offset)
            elif axes is Axes.GRID:
                if to_axes is Axes.CUBE:
                    size = self.size_tensor()
                    matrix = paddle.diag(x=2 / size)
                    if not vectors:
                        one = paddle.to_tensor(data=1, dtype=size.dtype, place=size.place)
                        matrix = homogeneous_matrix(matrix, offset=one / size - one)
                elif to_axes is Axes.CUBE_CORNERS:
                    size = self.size_tensor()
                    matrix = paddle.diag(x=2 / (size - 1))
                    if not vectors:
                        offset = paddle.to_tensor(data=-1, dtype=size.dtype, place=size.place)
                        matrix = homogeneous_matrix(matrix, offset=offset)
                elif to_axes is Axes.WORLD:
                    matrix = self.affine()
                    if not vectors:
                        matrix = homogeneous_matrix(matrix, offset=self.origin())
            elif axes is Axes.CUBE:
                if to_axes is Axes.CUBE_CORNERS:
                    size = self.size_tensor()
                    matrix = paddle.diag(x=size / (size - 1))
                elif to_axes is Axes.GRID:
                    half_size = 0.5 * self.size_tensor()
                    matrix = paddle.diag(x=half_size)
                    if not vectors:
                        matrix = homogeneous_matrix(matrix, offset=half_size - 0.5)
                elif to_axes is Axes.WORLD:
                    cube_to_grid = self.transform(axes, Axes.GRID, vectors=vectors)
                    grid_to_world = self.transform(Axes.GRID, Axes.WORLD, vectors=vectors)
                    if vectors:
                        matrix = paddle.mm(input=grid_to_world, mat2=cube_to_grid)
                    else:
                        matrix = hmm(grid_to_world, cube_to_grid)
            elif axes is Axes.CUBE_CORNERS:
                if to_axes is Axes.CUBE:
                    size = self.size_tensor()
                    matrix = paddle.diag(x=(size - 1) / size)
                elif to_axes is Axes.GRID:
                    scales = 0.5 * (self.size_tensor() - 1)
                    matrix = paddle.diag(x=scales)
                    if not vectors:
                        matrix = homogeneous_matrix(matrix, offset=scales)
                elif to_axes is Axes.WORLD:
                    interim = Axes.GRID
                    cube_to_grid = self.transform(axes, interim, vectors=vectors)
                    grid_to_world = self.transform(interim, to_axes, vectors=vectors)
                    if vectors:
                        matrix = paddle.mm(input=grid_to_world, mat2=cube_to_grid)
                    else:
                        matrix = hmm(grid_to_world, cube_to_grid)
            elif axes is Axes.WORLD:
                if to_axes is Axes.CUBE or to_axes is Axes.CUBE_CORNERS:
                    interim = Axes.GRID
                    world_to_grid = self.transform(axes, interim, vectors=vectors)
                    grid_to_cube = self.transform(interim, to_axes, vectors=vectors)
                    if vectors:
                        matrix = paddle.mm(input=grid_to_cube, mat2=world_to_grid)
                    else:
                        matrix = hmm(grid_to_cube, world_to_grid)
                elif to_axes is Axes.GRID:
                    matrix = self.inverse_affine()
                    if not vectors:
                        matrix = hmm(matrix, -self.origin())
        elif to_grid.ndim != self.ndim:
            raise ValueError(f"Grid.transform() 'to_grid' must have {self.ndim} spatial dimensions")
        else:
            target_to_world = self.transform(axes, Axes.WORLD, vectors=vectors)
            world_to_source = to_grid.transform(Axes.WORLD, to_axes, vectors=vectors)
            if vectors:
                matrix = paddle.mm(input=world_to_source, mat2=target_to_world)
            else:
                matrix = hmm(world_to_source, target_to_world)
        if matrix is None:
            raise NotImplementedError(f"Grid.transform() for axes={axes} and to_axes={to_axes}")
        return matrix

    def inverse_transform(self, vectors: bool = False) -> paddle.Tensor:
        r"""Transform which maps from world to grid cube space."""
        cube_axes = Axes.CUBE_CORNERS if self._align_corners else Axes.CUBE
        return self.transform(Axes.WORLD, cube_axes, vectors=vectors)

    def apply_transform(
        self,
        input: Array,
        axes: Axes,
        to_axes: Optional[Axes] = None,
        to_grid: Optional[Grid] = None,
        vectors: bool = False,
        decimals: Optional[int] = -1,
    ) -> paddle.Tensor:
        r"""Map point coordinates or displacement vectors from one grid to another.

        Args:
            input: Points or vectors to transform as tensor of shape ``(..., D)``.
            axes: Axes with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.
            vectors: Whether transformation is used to rescale and reorient vectors.
            decimals: If positive or zero, number of digits right of the decimal point to round to.
                When mapping points to ``Axes.GRID``, ``Axes.CUBE``, or ``Axes.CUBE_CORNERS``,
                this function by default (``decimals=-1``) rounds the transformed coordinates.
                Explicitly set ``decimals=None`` to suppress this default rounding.

        Returns:
            If ``vectors=False``, a homogeneous coordinate transformation of shape ``(D, D + 1)``.
            Otherwise, a square transformation matrix of shape ``(D, D)`` is returned.

        """
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        input = as_tensor(input)
        if not input.is_floating_point():
            input = input.astype(self.dtype)
        if to_grid is not None and to_grid != self or axes is not to_axes:
            matrix = self.transform(axes, to_axes, to_grid=to_grid, vectors=vectors)
            matrix = matrix.unsqueeze(axis=0).to(device=input.place)
            result = homogeneous_transform(matrix, input)
        else:
            result = input
        if decimals == -1:
            if to_axes is Axes.CUBE or to_axes is Axes.CUBE_CORNERS:
                decimals = 12
            elif to_axes is Axes.GRID:
                decimals = 6
        if decimals is not None and decimals >= 0:
            result = round_decimals(result, decimals=decimals)
        return result

    def transform_points(
        self,
        points: Array,
        axes: Axes,
        to_axes: Optional[Axes] = None,
        to_grid: Optional[Grid] = None,
        decimals: Optional[int] = -1,
    ) -> paddle.Tensor:
        r"""Map point coordinates from one grid domain to another.

        Args:
            points: Coordinates of points to transform as tensor of shape ``(..., D)``.
            axes: Coordinate axes with respect to which ``points`` are defined.
            to_axes: Coordinate axes to which ``points`` should be mapped to. If ``None``, use ``axes``.
            to_grid: Grid with respect to which the codomain is defined. If ``None``, the target
                and source sampling grids are assumed to be identical.
            decimals: If positive or zero, number of digits right of the decimal point to round to.
                When mapping points to codomain ``Axes.GRID``, ``Axes.CUBE``, or ``Axes.CUBE_CORNERS``,
                this function by default (``decimals=-1``) rounds the transformed coordinates.
                Explicitly set ``decimals=None`` to suppress this default rounding.

        Returns:
            Point coordinates in ``axes`` mapped to coordinates with respect to ``to_axes``.

        """
        return self.apply_transform(points, axes, to_axes, to_grid=to_grid, decimals=decimals)

    def transform_vectors(
        self,
        vectors: Array,
        axes: Axes,
        to_axes: Optional[Axes] = None,
        to_grid: Optional[Grid] = None,
    ) -> paddle.Tensor:
        r"""Rescale and reorient flow vectors.

        Args:
            vectors: Displacement vectors to transform, e.g., as tensor of shape ``(..., D)``.
            axes: Coordinate axes with respect to which ``vectors`` are defined.
            to_axes: Coordinate axes to which ``vectors`` should be mapped to. If ``None``, use ``axes``.
            to_grid: Grid with respect to which ``to_axes`` is defined. If ``None``, the target
                and source sampling grids are assumed to be identical.

        Returns:
            Rescaled and reoriented vectors. If ``to_grid == self`` and ``to_axes == axes``,
            a reference to the unmodified input ``vectors`` tensor is returned.

        """
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        vectors = as_tensor(vectors)
        if not vectors.is_floating_point():
            vectors = vectors.astype(self.dtype)
        if axes is Axes.WORLD and to_axes is Axes.WORLD:
            return vectors
        if to_grid is None or to_grid == self:
            if axes is not to_axes:
                affine = None  # affine transform required if reorientation needed
                scales = None  # otherwise, just scaling of displacements suffices
                if axes is Axes.WORLD:
                    affine = self.inverse_affine()
                elif axes is Axes.CUBE:
                    scales = self.size_tensor() / 2
                elif axes is Axes.CUBE_CORNERS:
                    scales = (self.size_tensor() - 1) / 2
                elif axes is not Axes.GRID:
                    raise NotImplementedError(
                        f"Grid.transform_vectors() for axes={axes} and to_axes={to_axes}"
                    )
                if to_axes is Axes.WORLD:
                    grid_to_world = self.affine()
                    if scales is None:
                        assert affine is None
                        affine = grid_to_world
                    else:
                        affine = paddle.mm(input=grid_to_world, mat2=paddle.diag(x=scales))
                elif to_axes is Axes.CUBE or to_axes is Axes.CUBE_CORNERS:
                    num = self.size_tensor()
                    if to_axes is Axes.CUBE_CORNERS:
                        num -= 1
                    grid_to_cube = 2 / num
                    if affine is None:
                        if scales is None:
                            scales = grid_to_cube
                        else:
                            scales *= grid_to_cube
                    else:
                        affine = paddle.mm(input=paddle.diag(x=grid_to_cube), mat2=affine)
                elif to_axes is not Axes.GRID:
                    raise NotImplementedError(
                        f"Grid.transform_vectors() for axes={axes} and to_axes={to_axes}"
                    )
                if affine is None:
                    assert scales is not None
                    scales = scales.to(vectors)
                    vectors = vectors * scales
                else:
                    affine = affine.to(vectors)
                    tensor = vectors.reshape(-1, tuple(vectors.shape)[-1])
                    vectors = paddle.nn.functional.linear(x=tensor, weight=affine.T).reshape(
                        tuple(vectors.shape)
                    )
        else:
            matrix = self.transform(axes, to_axes, to_grid=to_grid, vectors=True)
            matrix = matrix.to(vectors)
            vectors = homogeneous_transform(matrix, vectors, vectors=True)
        return vectors

    def index_to_cube(
        self, indices: Array, decimals: int = -1, align_corners: Optional[bool] = None
    ) -> paddle.Tensor:
        r"""Map points from grid indices to grid-aligned cube with side length 2.

        Args:
            indices: Grid point indices to transform as tensor of shape ``(..., D)``.
            decimals: If positive or zero, number of digits right of the decimal point to round to.
            align_corners: Whether output cube coordinates should be with respect to
                ``Axes.CUBE_CORNERS`` (True) or ``Axes.CUBE`` (False), respectively.
                If ``None``, use default ``self.align_corners()``.

        Returns:
            Grid point indices transformed to points with respect to cube.

        """
        if align_corners is None:
            align_corners = self._align_corners
        to_axes = Axes.from_align_corners(align_corners)
        return self.transform_points(indices, axes=Axes.GRID, to_axes=to_axes, decimals=decimals)

    def cube_to_index(
        self, coords: Array, decimals: int = -1, align_corners: Optional[bool] = None
    ) -> paddle.Tensor:
        r"""Map points from grid-aligned cube to grid point indices.

        Args:
            coords: Normalized grid points to transform as tensor of shape ``(..., D)``.
            decimals: If positive or zero, number of digits right of the decimal point to round to.
            align_corners: Whether ``coords`` are with respect to ``Axes.CUBE_CORNERS`` (True)
                or ``Axes.CUBE`` (False), respectively. If ``None``, use default ``self.align_corners()``.

        Returns:
            Points in grid-aligned cube transformed to grid indices.

        """
        if align_corners is None:
            align_corners = self._align_corners
        axes = Axes.from_align_corners(align_corners)
        return self.transform_points(coords, axes=axes, to_axes=Axes.GRID, decimals=decimals)

    def index_to_world(self, indices: Array, decimals: int = -1) -> paddle.Tensor:
        r"""Map points from grid indices to world coordinates.

        Args:
            indices: Grid point indices to transform as tensor of shape ``(..., D)``.
            decimals: If positive or zero, number of digits right of the decimal point to round to.

        Returns:
            Grid point indices transformed to points in world space.

        """
        return self.transform_points(indices, axes=Axes.GRID, to_axes=Axes.WORLD, decimals=decimals)

    def world_to_index(self, points: Array, decimals: int = -1) -> paddle.Tensor:
        r"""Map points from world coordinates to grid point indices.

        Args:
            points: World coordinates of points to transform as tensor of shape ``(..., D)``.
            decimals: If positive or zero, number of digits right of the decimal point to round to.

        Returns:
            Points in world space transformed to grid indices.

        """
        return self.transform_points(points, axes=Axes.WORLD, to_axes=Axes.GRID, decimals=decimals)

    def cube_to_world(
        self, coords: Array, decimals: int = -1, align_corners: Optional[bool] = None
    ) -> paddle.Tensor:
        r"""Map point coordinates from grid-aligned cube with side length 2 to world space.

        Args:
            coords: Normalized grid points to transform as tensor of shape ``(..., D)``.
            decimals: If positive or zero, number of digits right of the decimal point to round to.
            align_corners: Whether ``coords`` are with respect to ``Axes.CUBE_CORNERS`` (True)
                or ``Axes.CUBE`` (False), respectively. If ``None``, use default ``self.align_corners()``.

        Returns:
            Normalized grid coordinates transformed to world space coordinates.

        """
        if align_corners is None:
            align_corners = self._align_corners
        axes = Axes.from_align_corners(align_corners)
        return self.transform_points(coords, axes=axes, to_axes=Axes.WORLD, decimals=decimals)

    def world_to_cube(
        self, points: Array, decimals: int = -1, align_corners: Optional[bool] = None
    ) -> paddle.Tensor:
        r"""Map point coordinates from world space to grid-aligned cube with side length 2.

        Args:
            points: World coordinates of points to transform as tensor of shape ``(..., D)``.
            decimals: If positive or zero, number of digits right of the decimal point to round to.
            align_corners: Whether output cube coordinates should be with respect to
                ``Axes.CUBE_CORNERS`` (True) or ``Axes.CUBE`` (False), respectively.
                If ``None``, use default ``self.align_corners()``.

        Returns:
            Points in world space transformed to normalized grid coordinates.

        """
        if align_corners is None:
            align_corners = self._align_corners
        to_axes = Axes.from_align_corners(align_corners)
        return self.transform_points(points, axes=Axes.WORLD, to_axes=to_axes, decimals=decimals)

    def coords(
        self,
        dim: Optional[int] = None,
        center: bool = False,
        normalize: bool = True,
        align_corners: Optional[bool] = None,
        channels_last: bool = True,
        flip: bool = False,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ) -> paddle.Tensor:
        r"""Get tensor of grid point coordinates.

        Args:
            dim: Return coordinates for specified dimension, where ``dim=0`` refers to
                the first grid dimension, i.e., the ``x`` axis.
            center: Whether to center ``Axes.GRID`` coordinates when ``normalize=False``.
            normalize: Normalize coordinates to grid-aligned cube with side length 2.
                The normalized grid point coordinates are in the closed interval ``[-1, +1]``.
            align_corners: If ``normalize=True``, specifies whether the extrema ``(-1, 1)``
                should refer to the centers of the grid corner squares or their boundary.
                Note that in both cases the returned normalized coordinates are associated
                with the center points of the grid squares. When used as ``grid`` argument
                of ``paddle.nn.functional.grid_sample()``, the same ``align_corners`` value
                should be used for both ``Grid.coords()`` and ``grid_sample()`` calls.
                If ``None``, use default ``self.align_corners()``.
            channels_last: Whether to place stacked coordinates at last (``True``) or first (``False``)
                output tensor dimension. Vector fields are represented with channels first after the
                batch dimension, whereas point sets such as the sampling points passed to ``grid_sample``
                are represented with point coordinates in the last tensor dimension.
            flip: Whether to return coordinates in the order (..., x) instead of (x, ...).
            dtype: Data type of coordinates. If ``None``, uses ``paddle.int32`` as data type
                for returned tensor if ``normalize=False``, and ``self.dtype`` otherwise.
            device: Device on which to create paddle tensor. If ``None``, use ``self.device``.

        Returns:
            If ``dim`` is ``None``, returns a tensor of shape (...X, C) if ``channels_last=True`` (default)
            or ``(C, ..., X)`` if ``channels_last=False``, where C is the number of spatial grid dimensions.
            If ``normalize=Falze`` and ``center=False``, the tensor values are the multi-dimensional indices
            in the closed-open interval [0, n) for each grid dimension, where n is the number of points in the
            respective dimension. The first channel with index 0 is the ``x`` coordinate. If ``normalize=False``
            and ``center=True``, the indices are shifted such that index 0 corresponds to the grid center point.
            If ``normalize=True``, the centered coordinates are normalized to ``(-1, 1)``, where the extrema
            either correspond to the corner points of the grid (``align_corners=True``) or the grid boundary
            edges (``align_cornes=False``). If ``dim`` is not ``None``, a 1D tensor with the coordinates for
            this grid axis is returned.

        """
        if align_corners is None:
            align_corners = self._align_corners
        if dtype is None:
            if normalize or center:
                dtype = self.dtype
            else:
                dtype = "int32"
        if device is None:
            device = self.device
        if dim is None:
            shape = self.shape  # order (...X), do NOT use self.size() here!
        else:
            shape = tuple((self.size()[dim],))  # dim may be negative
        if np.prod(shape) == 0:
            return paddle.empty(shape=shape, dtype=dtype)
        coords = []
        for n in shape:
            if n == 1:
                coord = paddle.to_tensor(data=[0], dtype=dtype, place=device)
            elif normalize:
                if align_corners:
                    spacing = 2 / (n - 1)
                    extrema = -1, 1 + 0.1 * spacing
                else:
                    spacing = 2 / n
                    extrema = -1 + 0.5 * spacing, 1
                coord = paddle.arange(*extrema, spacing, dtype=dtype)
            elif center:
                radius = (n - 1) / 2
                coord = paddle.linspace(start=-radius, stop=radius, num=n, dtype=dtype)
            else:
                coord = paddle.arange(dtype=dtype, end=n)
            coords.append(coord)
        channels_dim = len(coords) if channels_last else 0
        if parse_version(paddle.__version__) < parse_version("1.10"):
            coords = paddle.stack(x=paddle.meshgrid(*coords), axis=channels_dim)
        else:
            coords = paddle.stack(x=paddle.meshgrid(*coords), axis=channels_dim)
        if not flip:  # default order (x, ...) requires flipping stacked meshgrid coords
            coords = paddle.flip(x=coords, axis=(channels_dim,))
        return coords

    def points(
        self,
        axes: Axes = Axes.WORLD,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ) -> paddle.Tensor:
        r"""Tensor of grid point coordinates with respect to specified coordinate axes."""
        axes = Axes(axes)
        coords = self.coords(
            normalize=axes is Axes.CUBE, align_corners=False, dtype=dtype, device=device
        )
        if axes is not Axes.CUBE and axes is not Axes.GRID:
            coords = self.apply_transform(coords, Axes.GRID, to_axes=axes)
        return coords

    def _resize(self, size: paddle.Tensor, align_corners: Optional[bool] = None) -> Grid:
        r"""Get new grid of specified size.

        The resulting ``grid``` MUST preserve floating point values of given ``size`` argument such
        as in particular those passed by ``downsample()`` and ``upsample()``. Otherwise, a sequence
        of resampling steps which should produce the original grid may result in a different size.

        Args:
            size: Size of new grid.
            align_corners: Whether to preserve positions of corner points (``True``) or grid extent (``False``).

        """
        if align_corners is None:
            align_corners = self._align_corners
        size = size.to(dtype=self.dtype, device=self.device)
        if size.equal(y=self._size).astype("bool").all():
            return self
        grid = shallow_copy(self)
        grid._size = size
        size = self._round_size(size)
        if align_corners:
            spacing = (self.extent() - self.spacing()) / (size - 1)
            grid._spacing = paddle.where(
                condition=self._size.greater_than(y=paddle.to_tensor(0)), x=spacing, y=self._spacing
            )
            assert paddle.allclose(x=grid.origin(), y=self.origin()).item()
        else:
            spacing = self.extent() / size
            grid._spacing = paddle.where(
                condition=self._size.greater_than(y=paddle.to_tensor(0)), x=spacing, y=self._spacing
            )
            assert paddle.allclose(x=grid.extent(), y=self.extent()).item()
        return grid

    def resize(
        self, size: Union[int, Size, Array], *args: int, align_corners: Optional[bool] = None
    ) -> Grid:
        r"""Create new grid of same extent with specified size.

        Specify new grid size for grid dimensions in the order (X...). Note that this is in reverse order
        of ``Tensor.size()``! To set the grid size given a data tensor shape, use ``Grid.reshape()``.

        Args:
            size: Size of new grid in the order ``(X, ...)``.
            align_corners: Whether to preserve positions of corner points (``True``) or grid extent (``False``).

        Returns:
            New grid with given ``size`` and adjusted ``Grid.spacing()``.

        """
        size = cat_scalars(size, *args, num=self.ndim, device=self.device)
        if not is_int_dtype(size.dtype):
            raise TypeError(f"Grid.resize() 'size' must be integer values, got dtype={size.dtype}")
        if size.less_than(y=paddle.to_tensor(0)).astype("bool").any():
            raise ValueError("Grid.resize() 'size' must be all non-negative numbers")
        return self._resize(size, align_corners=align_corners)

    def reshape(
        self, shape: Union[int, Shape, Array], *args: int, align_corners: Optional[bool] = None
    ) -> Grid:
        r"""Create new grid of same extent with specified data tensor shape.

        The data tensor shape specifies the size of data dimensions in the order ``(..., X)``,
        whereas the ``Grid.size()`` is given in reverse order ``(X, ...)``. This function is a
        convenience function to change the grid size given the ``Tensor.shape`` of a data tensor.

        Args:
            shape: Size of new grid in the order ``(..., X)``.
            align_corners: Whether to preserve positions of corner points (``True``) or grid extent (``False``).

        Returns:
            New grid with given ``shape`` and adjusted ``Grid.spacing()``.

        """
        shape = cat_scalars(shape, *args, num=self.ndim, device=self.device)
        if not is_int_dtype(shape.dtype):
            raise TypeError(
                f"Grid.reshape() 'shape' must be integer values, got dtype={shape.dtype}"
            )
        if shape.less_than(y=paddle.to_tensor(0)).astype("bool").any():
            raise ValueError("Grid.reshape() 'shape' must be all non-negative numbers")
        return self._resize(shape.flip(axis=0), align_corners=align_corners)

    def resample(self, spacing: Union[float, Array, str], *args: float, min_size: int = 1) -> Grid:
        r"""Create new grid with specified spacing.

        Args:
            spacing: Desired spacing between grid points. Uses minimum or maximum grid spacing
                for isotropic resampling when argument is string "min" or "max", respectively.
            min_size: Minimum grid size.

        Returns:
            New grid with specified spacing. The extent of the grid may be greater
            than before, if the original extent is not divisible by the desired spacing.

        """
        if spacing == "min":
            assert not args
            spacing = self._spacing.min()
        elif spacing == "max":
            assert not args
            spacing = self._spacing.max()
        elif isinstance(spacing, str):
            raise ValueError(
                f"{type(self).__name__}.resample() 'spacing' str must be 'min' or 'max'"
            )
        spacing = cat_scalars(spacing, *args, num=self.ndim, dtype=self.dtype, device=self.device)
        if paddle.allclose(x=spacing, y=self._spacing).item():
            return self
        if spacing.less_equal(y=paddle.to_tensor(0)).astype("bool").any():
            raise ValueError("Grid.resample() 'spacing' must be all positive numbers")
        size = self.extent().div(spacing)
        size = paddle.where(
            condition=self._size.greater_than(y=paddle.to_tensor(0)),
            x=size.clip(min=min_size),
            y=size,
        )
        grid = shallow_copy(self)
        grid._size = size
        grid._spacing = spacing
        return grid

    def pool(
        self,
        kernel_size: ScalarOrTuple[int],
        stride: Optional[ScalarOrTuple[int]] = None,
        padding: ScalarOrTuple[int] = 0,
        dilation: ScalarOrTuple[int] = 1,
        ceil_mode: bool = False,
    ) -> Grid:
        r"""Output grid after applying pooling operation.

        Args:
            kernel_size: Size of the pooling region.
            stride: Stride of the pooling operation.
            padding: Implicit zero paddings on both sides of the input.
            dilation: Spacing between pooling kernel elements.
            ceil_mode: When True, will use `ceil` instead of `floor` to compute the output size.

        Returns:
            New grid corresponding to output data tensor after pooling operation.

        """
        if stride is not None:
            raise NotImplementedError("Grid.pool() 'stride' currently not supported")
        if padding != 0:
            raise NotImplementedError("Grid.pool() 'padding' currently not supported")
        if dilation != 1:
            raise NotImplementedError("Grid.pool() 'dilation' currently not supported")
        ks = cat_scalars(kernel_size, num=self.ndim, dtype=self.dtype, device=self.device)
        size = self.size_tensor() / ks
        size = size.ceil() if ceil_mode else size.floor()
        size = size.astype("int32")
        grid = Grid(
            size=size,
            origin=self.index_to_world(ks.sub(1).div(2)),
            spacing=self.spacing().mul(ks),
            direction=self.direction(),
            align_corners=self.align_corners(),
            device=self.device,
        )
        return grid

    def avg_pool(
        self,
        kernel_size: ScalarOrTuple[int],
        stride: Optional[ScalarOrTuple[int]] = None,
        padding: ScalarOrTuple[int] = 0,
        ceil_mode: bool = False,
    ) -> Grid:
        r"""Output grid after applying average pooling."""
        return self.pool(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def downsample(
        self,
        levels: int = 1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        min_size: int = 1,
        align_corners: Optional[bool] = None,
    ) -> Grid:
        r"""Create new grid with size halved the specified number of times.

        Args:
            levels: Number of times the grid size is halved (>0) or doubled (<0).
            dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
            min_size: Minimum grid size. If downsampling the grid along a spatial dimension would reduce its
                size below the given minimum value, the grid is not further downsampled along this dimension.
            align_corners: Whether to preserve positions of corner points (``True``) or grid extent (``False``).

        Returns:
            New grid.

        """
        if not isinstance(levels, int):
            raise TypeError("Grid.downsample() 'levels' must be of type int")
        if not dims:
            dims = tuple(dim for dim in range(self.ndim))
        dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
        size = self._size.clone()
        scale = 2**levels
        for dim in dims:
            size[dim] /= scale
        size = paddle.where(
            condition=size.greater_equal(y=paddle.to_tensor(min_size)), x=size, y=self._size
        )
        return self._resize(size, align_corners=align_corners)

    def upsample(
        self,
        levels: int = 1,
        dims: Optional[Sequence[SpatialDimArg]] = None,
        align_corners: Optional[bool] = None,
    ) -> Grid:
        r"""Create new grid of same extent with size doubled the specified number of times.

        Args:
            levels: Number of times the grid size is doubled (>0) or halved (<0).
            dims: Spatial dimensions along which to upsample. If not specified, consider all spatial dimensions.
            min_size: Minimum grid size. If downsampling the grid along a spatial dimension would reduce its
                size below the given minimum value, the grid is not further downsampled along this dimension.
            align_corners: Whether to preserve positions of corner points (``True``) or grid extent (``False``).

        Returns:
            New grid.

        """
        if not isinstance(levels, int):
            raise TypeError("Grid.upsample() 'levels' must be of type int")
        if not dims:
            dims = tuple(dim for dim in range(self.ndim))
        dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
        size = self._size.clone()
        scale = 2**levels
        for dim in dims:
            size[dim] *= scale
        return self._resize(size, align_corners=align_corners)

    def pyramid(
        self, levels: int, dims: Optional[Sequence[SpatialDimArg]] = None, min_size: int = 0
    ) -> Dict[int, Grid]:
        r"""Compute image size at each image resolution pyramid level.

        This function computes suitable image sizes for each level of a multi-resolution
        image pyramid depending on original image size and spacing, a minimum grid size for
        every level, and whether grid corners (``align_corners=True``) or grid borders
        (``align_corners=False``) should be aligned.

        Args:
            levels: Number of resolution levels.
            dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
            min_size: Minimum grid size at each level. If the grid size after downsampling
                would be smaller than the specified minimum size, the grid size is not reduced
                for this spatial dimension. The number of resolution levels is not affected.

        Returns:
            Dictionary mapping resolution level to sampling grid. The sampling grid at the
            finest resolution level has index 0. The cube extent, i.e., the physical length
            between grid points corresponding to cube interval ``(-1, 1)`` will be the same
            for all resolution levels.

        """
        if not isinstance(levels, int):
            raise TypeError("Grid.pyramid() 'levels' must be int")
        if not isinstance(min_size, int):
            raise TypeError("Grid.pyramid() 'min_size' must be int")
        if not dims:
            dims = tuple(dim for dim in range(self.ndim))
        dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
        m = sum([(2**i) for i in range(levels)]) if self._align_corners else 0
        sizes = {level: list(self.size()) for level in range(levels + 1)}
        for dim in dims:
            sizes[levels][dim] = int(0.5 + (sizes[levels][dim] + m) / 2**levels)
            for level in range(levels - 1, -1, -1):
                sizes[level][dim] = 2 * sizes[level + 1][dim] - 1
            for level in range(1, levels + 1):
                sizes[level][dim] = (sizes[level - 1][dim] + 1) // 2
                if sizes[level][dim] < min_size:
                    sizes[level][dim] = sizes[level - 1][dim]
        return {level: self.resize(size) for level, size in sizes.items()}

    def crop(
        self,
        *args: int,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
    ) -> Grid:
        r"""Create new grid with a margin along each axis removed.

        Args:
            args: Crop ``margin`` specified as int arguments.
            margin: Number of spatial grid points to remove (positive) or add (negative) at each border.
                Use instead of ``num`` in order to symmetrically crop the input ``data`` tensor, e.g.,
                ``(nx, ny, nz)`` is equivalent to ``num=(nx, nx, ny, ny, nz, nz)``.
            num: Number of spatial grid points to remove (positive) or add (negative) at each border,
                where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
                ``(nx, nx, ny, ny)``. If a scalar is given, the input is cropped equally at all borders.
                Otherwise, the given sequence must have an even length.

        Returns:
            New grid with modified ``Grid.size()``, but unchanged ``Grid.spacing``.
            Hence, the ``Grid.extent()`` of the new grid will be different from ``self.extent()``.

        """
        if sum([1 if args else 0, 0 if num is None else 1, 0 if margin is None else 1]) != 1:
            raise AssertionError("Grid.pad() 'args', 'margin', and 'num' are mutually exclusive")
        if len(args) == 1 and not isinstance(args[0], int):
            margin = args[0]
        elif args:
            margin = args
        if isinstance(margin, int):
            num = margin
        elif margin is not None:
            num = tuple(n for n_n in ((n, n) for n in margin) for n in n_n)
        assert num is not None
        if isinstance(num, int):
            num = (num,) * (2 * self.ndim)
        else:
            num = tuple(int(n) for n in num)
        if len(num) % 2 != 0:
            raise ValueError("Grid.crop() 'num' must be int or have even length")
        if all(n == 0 for n in num):
            return self
        num = num + (0,) * (2 * self.ndim - len(num))
        num_ = paddle.to_tensor(data=num, dtype=self.dtype, place=self.device)
        size = paddle.clip(x=self._size - num_[::2] - num_[1::2], min=1)
        size = paddle.where(
            condition=self._size.greater_than(y=paddle.to_tensor(0)), x=size, y=self._size
        )
        origin = self.index_to_world(num_[::2])
        return Grid(
            size=size,
            origin=origin,
            spacing=self.spacing(),
            direction=self.direction(),
            align_corners=self.align_corners(),
            device=self.device,
        )

    def pad(
        self,
        *args: int,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
    ) -> Grid:
        r"""Create new grid with an additional margin along each axis.

        Args:
            args: Pad ``margin`` specified as int arguments.
            margin: Number of spatial grid points to add (positive) or remove (negative) at each border.
                Use instead of ``num`` in order to symmetrically pad the input ``data`` tensor, e.g.,
                ``(nx, ny, nz)`` is equivalent to ``num=(nx, nx, ny, ny, nz, nz)``.
            num: Number of spatial grid points to remove (positive) or add (negative) at each border,
                where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
                ``(nx, nx, ny, ny)``. If a scalar is given, the input is cropped equally at all borders.
                Otherwise, the given sequence must have an even length.

        Returns:
            New grid with modified ``Grid.size()``, but unchanged ``Grid.spacing``.
            Hence, the ``Grid.extent()`` of the new grid will be different from ``self.extent()``.

        """
        if sum([1 if args else 0, 0 if num is None else 1, 0 if margin is None else 1]) != 1:
            raise AssertionError("Grid.pad() 'args', 'margin', and 'num' are mutually exclusive")
        if len(args) == 1 and not isinstance(args[0], int):
            margin = args[0]
        elif args:
            margin = args
        if isinstance(margin, int):
            num = margin
        elif margin is not None:
            num = tuple(n for n_n in ((n, n) for n in margin) for n in n_n)
        assert num is not None
        if isinstance(num, int):
            num = (num,) * (2 * self.ndim)
        else:
            num = tuple(int(n) for n in num)
        if len(num) % 2 != 0:
            raise ValueError("Grid.pad() 'num' must be int or have even length")
        if all(n == 0 for n in num):
            return self
        num = num + (0,) * (2 * self.ndim - len(num))
        num_ = paddle.to_tensor(data=num, dtype=self.dtype, place=self.device)
        size = paddle.clip(x=self._size + num_[::2] + num_[1::2], min=1)
        size = paddle.where(
            condition=self._size.greater_than(y=paddle.to_tensor(0)), x=size, y=self._size
        )
        origin = self.index_to_world(-num_[::2])
        return Grid(
            size=size,
            origin=origin,
            spacing=self.spacing(),
            direction=self.direction(),
            align_corners=self.align_corners(),
            device=self.device,
        )

    def center_crop(self, size: Union[int, Array], *args: int) -> Grid:
        r"""Crop grid to specified maximum size."""
        size = cat_scalars(size, *args, num=self.ndim, device=self.device)
        if not is_int_dtype(size.dtype):
            raise TypeError(
                f"Grid.center_crop() expected scalar or array of integer values, got dtype={size.dtype}"
            )
        size = [min(m, n) for m, n in zip(self.size(), size.tolist())]
        origin = [((m - n) // 2) for m, n in zip(self.size(), size)]
        return Grid(
            size=size,
            origin=self.index_to_world(origin),
            spacing=self.spacing(),
            direction=self.direction(),
            align_corners=self.align_corners(),
            device=self.device,
        )

    def center_pad(self, size: Union[int, Array], *args: int) -> Grid:
        r"""Pad grid to specified minimum size."""
        size = cat_scalars(size, *args, num=self.ndim, device=self.device)
        if not is_int_dtype(size.dtype):
            raise TypeError(
                f"Grid.center_crop() expected scalar or array of integer values, got dtype={size.dtype}"
            )
        size = [max(m, n) for m, n in zip(self.size(), size.tolist())]
        origin = [(-((n - m) // 2)) for m, n in zip(self.size(), size)]
        return Grid(
            size=size,
            origin=self.index_to_world(origin),
            spacing=self.spacing(),
            direction=self.direction(),
            align_corners=self.align_corners(),
            device=self.device,
        )

    def narrow(self, dim: int, start: int, length: int) -> Grid:
        r"""Narrow grid along specified dimension."""
        if dim < 0 or dim > self.ndim:
            raise IndexError("Grid.narrow() 'dim' is out of bounds")
        size = tuple(length if d == dim else n for d, n in enumerate(self.size()))
        origin = tuple(start if d == dim else 0 for d in range(self.ndim))
        return Grid(
            size=size,
            origin=self.index_to_world(origin),
            spacing=self.spacing(),
            direction=self.direction(),
            align_corners=self.align_corners(),
            device=self.device,
        )

    def region_of_interest(self, start: Union[int, Array], size: Union[int, Array]) -> Grid:
        r"""Get region of interest grid."""
        start = cat_scalars(start, num=self.ndim, device=self.device)
        if not is_int_dtype(start.dtype):
            raise TypeError(
                f"Grid.region_of_interest() 'start' must be scalar or array of integer values, got dtype={start.dtype}"
            )
        size = cat_scalars(size, num=self.ndim, device=self.device)
        if not is_int_dtype(size.dtype):
            raise TypeError(
                f"Grid.region_of_interest() 'size' must be scalar or array of integer values, got dtype={size.dtype}"
            )
        grid_size = self.size()
        num = [[start[i], grid_size[i] - (start[i] + size[i])] for i in range(self.ndim)]
        num = [n for nn in num for n in nn]
        return self.crop(num=num)

    def same_domain_as(self, other: Grid) -> bool:
        r"""Check if this and another grid cover the same cube domain."""
        if other is self:
            return True
        return self.domain() == other.domain()

    def __eq__(self, other: Any) -> bool:
        r"""Compare this grid to another."""
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        for name in self.__slots__:
            if name == "_align_corners":
                continue
            value = getattr(self, name)
            other_value = getattr(other, name)
            if type(value) != type(other_value):
                return False
            if isinstance(value, paddle.Tensor):
                assert isinstance(other_value, paddle.Tensor)
                if tuple(value.shape) != tuple(other_value.shape):
                    return False
                other_value = other_value.to(device=value.place)
                if not paddle.allclose(x=value, y=other_value, rtol=1e-05, atol=1e-08).item():
                    return False
            elif (value != other_value).all():
                return False
        return True

    def __repr__(self) -> str:
        r"""String representation."""
        size = ", ".join([f"{v:>6.2f}" for v in self._size.numpy()])
        center = ", ".join([f"{v:.5f}" for v in self._center.numpy()])
        origin = ", ".join([f"{v:.5f}" for v in self.origin().numpy()])
        spacing = ", ".join([f"{v:.5f}" for v in self._spacing.numpy()])
        direction = ", ".join([f"{v:.5f}" for v in self._direction.flatten().numpy()])
        return (
            f"{type(self).__name__}("
            + f"size=({size})"
            + f", origin=({origin})"
            + f", center=({center})"
            + f", spacing=({spacing})"
            + f", direction=({direction})"
            + f", device={repr(str(self.device))}"
            + f", align_corners={repr(self._align_corners)}"
            + ")"
        )


def grid_points_transform(grid: Grid, axes: Axes, to_grid: Grid, to_axes: Optional[Axes] = None):
    r"""Get linear transformation of points from one grid domain to another.

    Args:
        grid: Sampling grid with respect to which input points are defined.
        axes: Grid axes with respect to which input points are defined.
        to_grid: Sampling grid with respect to which output points are defined.
        to_axes: Grid axes with respect to which output points are defined.

    Returns:
        Homogeneous coordinate transformation matrix as tensor of shape ``(D, D + 1)``.

    """
    return grid.transform(axes=axes, to_axes=to_axes, to_grid=to_grid, vectors=False)


def grid_vectors_transform(grid: Grid, axes: Axes, to_grid: Grid, to_axes: Optional[Axes] = None):
    r"""Get affine transformation which maps vectors with respect to one grid domain to another.

    Args:
        grid: Sampling grid with respect to which input vectors are defined.
        axes: Grid axes with respect to which input vectors are defined.
        to_grid: Sampling grid with respect to which output vectors are defined.
        to_axes: Grid axes with respect to which output vectors are defined.

    Returns:
        Affine transformation matrix as tensor of shape ``(D, D)``.

    """
    return grid.transform(axes=axes, to_axes=to_axes, to_grid=to_grid, vectors=True)


def grid_transform_points(
    points: paddle.Tensor,
    grid: Grid,
    axes: Axes,
    to_grid: Grid,
    to_axes: Optional[Axes] = None,
    decimals: Optional[int] = -1,
):
    r"""Map point coordinates from one grid domain to another.

    Args:
        points: Coordinates of points to transform as tensor of shape ``(..., D)``.
        grid: Grid with respect to which input ``points`` are defined.
        axes: Coordinate axes with respect to which ``points`` are defined.
        to_grid: Grid with respect to which the codomain is defined. If ``None``, the target
            and source sampling grids are assumed to be identical.
        to_axes: Coordinate axes to which ``points`` should be mapped to. If ``None``, use ``axes``.
        decimals: If positive or zero, number of digits right of the decimal point to round to.
            When mapping points to codomain ``Axes.GRID``, ``Axes.CUBE``, or ``Axes.CUBE_CORNERS``,
            this function by default (``decimals=-1``) rounds the transformed coordinates.
            Explicitly set ``decimals=None`` to suppress this default rounding.

    Returns:
        Point coordinates in ``axes`` mapped to coordinates with respect to ``to_axes``.

    """
    return grid.transform_points(
        points, axes=axes, to_axes=to_axes, to_grid=to_grid, decimals=decimals
    )


def grid_transform_vectors(
    vectors: paddle.Tensor, grid: Grid, axes: Axes, to_grid: Grid, to_axes: Optional[Axes] = None
):
    r"""Rescale and reorient flow vectors.

    Args:
        vectors: Displacement vectors to transform, e.g., as tensor of shape ``(..., D)``.
        grid: Grid with respect to which input ``vectors`` are defined.
        axes: Coordinate axes with respect to which input ``vectors`` are defined.
        to_grid: Grid with respect to which ``to_axes`` is defined. If ``None``, the target
            and source sampling grids are assumed to be identical.
        to_axes: Coordinate axes to which ``vectors`` should be mapped to. If ``None``, use ``axes``.

    Returns:
        Rescaled and reoriented vectors. If ``to_grid == grid`` and ``to_axes == axes``,
        a reference to the unmodified input ``vectors`` tensor is returned.

    """
    return grid.transform_vectors(vectors, axes=axes, to_axes=to_axes, to_grid=to_grid)
