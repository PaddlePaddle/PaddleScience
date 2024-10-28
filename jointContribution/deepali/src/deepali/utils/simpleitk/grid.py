r"""Attributes of data sampling grid oriented in space."""
import itertools
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import SimpleITK as sitk
from numpy.typing import ArrayLike
from numpy.typing import NDArray

TGridAttrs = TypeVar("TGridAttrs", bound="GridAttrs")


class GridAttrs(object):
    r"""Attributes of a regular data sampling grid oriented in world space."""

    __slots__ = ["_size", "origin", "spacing", "direction"]

    def __init__(
        self,
        size: Sequence[Union[float, int]],
        center: Optional[Sequence[float]] = None,
        origin: Optional[Sequence[float]] = None,
        spacing: Optional[Sequence[float]] = None,
        direction: Optional[Union[Sequence[float], Sequence[Sequence[float]]]] = None,
    ) -> None:
        r"""Initialize grid attributes.

        Note that the order of spatial dimensions of the attributes is ``(X, ...)``,
        whereas the data arrays often are given in reverse order, i.e., the shape of a
        corresponding data array is ``(..., X)``.

        Args:
            size: Size of spatial grid dimensions. The length determines the number of dimensions.
            center: Grid center point coordinates in world space along each dimension.
            origin: World coordinates of grid point with index zero along each dimension.
            spacing: Spacing between grid points along each dimension.
            direction: Direction cosines. Must be a square matrix, but can be flattened.

        """
        ndim = len(size)
        if spacing is None:
            spacing = (1.0,) * ndim
        elif not isinstance(spacing, tuple):
            spacing = tuple(spacing)
        if direction is None:
            direction = np.eye(ndim, ndim)
        direction = tuple(float(x) for x in np.asanyarray(direction).flatten())
        if origin is None:
            if center is None:
                origin = (0.0,) * ndim
            else:
                offset = np.array(0.5 * n if n > 0 else 0 for n in self._int_size(size))
                rotation = np.array(self.direction).reshape(ndim, ndim)
                scaling = np.diag(spacing)
                coords: NDArray = np.asanyarray(center) - np.matmul(rotation @ scaling, offset)
                origin = tuple(float(x) for x in coords)
        elif center is not None:
            raise ValueError("Grid() 'center' and 'origin' are mutually exclusive")
        elif not isinstance(origin, tuple):
            origin = tuple(origin)
        # store grid size as float such that grid.down().up() restores original grid
        self._size = tuple(float(n) for n in size)
        self.origin = origin
        self.spacing = spacing
        self.direction = direction

    @property
    def center(self) -> Tuple[float, ...]:
        r"""Get grid center point coordinates in world space."""
        offset = np.array(0.5 * n if n > 0 else 0 for n in self.size)
        coords: NDArray = self.origin + np.matmul(self.transform[:-1, :-1], offset)
        return tuple(float(x) for x in coords)

    @property
    def ndim(self) -> int:
        r"""Number of spatial grid dimensions."""
        return len(self.shape)

    @property
    def npoints(self) -> int:
        r"""Total number of grid points."""
        return int(np.prod(self.size))

    @staticmethod
    def _int_size(size) -> Tuple[int, ...]:
        r"""Get grid size from internal floating point representation."""
        return tuple([(int(n + 0.5) if n > 1 or n <= 0 else 1) for n in size])

    @property
    def size(self) -> Tuple[int, ...]:
        r"""Size of sampling grid with spatial dimensions in the order (X, ...)."""
        return self._int_size(self._size)

    @property
    def shape(self) -> Tuple[int, ...]:
        r"""Shape of sampling grid with spatial dimensions in the order (..., X)."""
        return tuple(reversed(self.size))

    def with_margin(self: TGridAttrs, margin: Union[int, None]) -> TGridAttrs:
        r"""Create new image grid with an additional margin along each grid axis."""
        if not margin:
            return self
        cls = type(self)
        return cls(
            size=tuple(n + 2 * margin for n in self._size),
            origin=self.index_to_physical_space((-margin,) * self.ndim),
            spacing=self.spacing,
            direction=self.direction,
        )

    def with_spacing(self: TGridAttrs, *args: float) -> TGridAttrs:
        r"""Create new image grid with specified spacing."""
        cls = type(self)
        spacing = np.asarray(*args)
        assert 0 <= spacing.ndim <= 1
        if spacing.ndim == 0:
            spacing = spacing.repeat(self.ndim)
        assert spacing.size == self.ndim
        cur_size = np.asarray(self._size)
        cur_spacing = np.asarray(self.spacing)
        size = cur_size * cur_spacing / spacing
        shift = 0.5 * (np.round(cur_size) - np.round(size) * spacing / cur_spacing)
        origin = self.index_to_physical_space(shift)
        return cls(size=size, origin=origin, spacing=spacing, direction=self.direction)

    def down(self: TGridAttrs, levels: int = 1) -> TGridAttrs:
        r"""Create new sampling grid of half the size while preserving grid extent."""
        cls = type(self)
        size = self._size
        for _ in range(levels):
            size = tuple(n / 2 for n in size)
        cur_size = self.size
        new_size = self._int_size(size)
        spacing = tuple(
            self.spacing[i] * cur_size[i] / new_size[i] if new_size[i] > 0 else self.spacing[i]
            for i in range(self.ndim)
        )
        return cls(size=size, center=self.center, spacing=spacing, direction=self.direction)

    def up(self: TGridAttrs, levels: int = 1) -> TGridAttrs:
        r"""Create new sampling grid of double the size while preserving grid extent."""
        cls = type(self)
        size = self._size
        for _ in range(levels):
            size = tuple(2 * n for n in size)
        cur_size = self.size
        new_size = self._int_size(size)
        spacing = tuple(
            self.spacing[i] * cur_size[i] / new_size[i] if new_size[i] > 0 else self.spacing[i]
            for i in range(self.ndim)
        )
        return cls(size=size, center=self.center, spacing=spacing, direction=self.direction)

    @property
    def dcm(self) -> np.ndarray:
        r"""Get direction cosine matrix."""
        return np.array(self.direction).reshape(self.ndim, self.ndim)

    @property
    def transform(self) -> np.ndarray:
        r"""Get homogeneous coordinate transformation from image grid to world space."""
        rotation = self.dcm
        scaling = np.diag(self.spacing)
        matrix = homogeneous_matrix(rotation @ scaling)
        matrix[0:-1, -1] = self.origin
        return matrix

    @property
    def inverse_transform(self) -> np.ndarray:
        r"""Get homogeneous coordinate transformation from world space to image grid."""
        rotation = self.dcm.T
        scaling = np.diag([(1 / s) for s in self.spacing])
        translation = translation_matrix([(-t) for t in self.origin])
        return homogeneous_matrix(scaling @ rotation) @ translation

    @property
    def indices(self) -> np.ndarray:
        r"""Get array of image grid point coordinates in image space."""
        return np.flip(
            np.stack(
                np.meshgrid(*[np.arange(arg) for arg in self.shape], indexing="ij"), axis=self.ndim
            ),
            axis=-1,
        )

    def axis_indices(self, axis: int) -> np.ndarray:
        r"""Get array of image grid indices along specified axis."""
        return np.arange(self.shape[axis])

    @property
    def points(self) -> np.ndarray:
        r"""Get array of image grid point coordinates in world space."""
        return self.index_to_physical_space(self.indices)

    @property
    def coords(self) -> Tuple[np.ndarray, ...]:
        r"""Get 1D arrays of grid point coordinates along each axis."""
        return tuple(self.axis_coords(axis) for axis in range(self.ndim))

    def axis_coords(self, axis: int) -> np.ndarray:
        r"""Get array of image grid point coordinates in world space along specified axis."""
        indices = [[0]] * self.ndim
        indices[axis] = [index for index in range(self.shape[axis])]
        mesh = np.stack(np.meshgrid(*indices, indexing="ij"), axis=self.ndim)
        return self.index_to_physical_space(mesh)[..., axis].flatten()

    @property
    def corners(self) -> np.ndarray:
        r"""Get corners of image domain in world space."""
        limits = []
        for axis in range(self.ndim):
            limits.append((0, self.shape[axis]))
        corners = [tuple(reversed(indices)) for indices in itertools.product(*limits)]
        return self.index_to_physical_space(corners)

    def index_to_physical_space(self, points: ArrayLike) -> np.ndarray:
        r"""Map point coordinates from image to world space.

        Args:
            points: Input voxel indices as n-dimensional array of shape ``(..., D)``.
                The input can be both integral or floating point indices.

        Returns:
            World coordinates of the indexed grid locations.

        """
        return transform_point(self.transform, points)

    def physical_space_to_index(self, points: ArrayLike) -> np.ndarray:
        r"""Map point coordinates from world to discrete image space.

        Args:
            points: World coordinates of input points given as n-dimensional array of shape ``(..., D)``.

        Returns:
            Integral indices of nearest grid points.

        """
        index: np.ndarray = np.round(self.physical_space_to_continuous_index(points))
        return index.astype(int)

    def physical_space_to_continuous_index(self, points: ArrayLike) -> np.ndarray:
        r"""Map point coordinates from world to continuous image space.

        Args:
            points: World coordinates of input points given as n-dimensional array of shape ``(..., D)``.

        Returns:
            Indices of grid locations as floating point values.

        """
        # Round to avoid 1e-15, 1e-17, -1e-14,... as representations for zero, s.t.,
        # indices == physical_space_to_continuous_index(index_to_physical_space(indices))
        return np.round(transform_point(self.inverse_transform, points), decimals=12)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, origin={self.origin}, spacing={self.spacing}, direction={self.direction})"


def image_grid_attributes(image: sitk.Image) -> GridAttrs:
    r"""Get image sampling grid attributes."""
    return GridAttrs(
        size=image.GetSize(),
        origin=image.GetOrigin(),
        spacing=image.GetSpacing(),
        direction=image.GetDirection(),
    )


def homogeneous_coords(point: np.ndarray, ndim: int = None, copy: bool = True) -> np.ndarray:
    r"""Create array of homogeneous point coordinates from D-dimensional point set.

    Args:
        point: Input point or point set as array of shape ``(..., D)``.
        ndim: Expected number of spatial dimensions. If specified, and the input ``point``
            array is of shape ``(..., ndim + 1)``, the input point array is returned
            without adding an additional dimension.
        copy: Whether to copy the input ``point`` array when ``ndim`` is specified and
            the shape of the input array is ``(..., ndim + 1)``.

    Returns:
        Array of homogeneous coordinates with shape ``(..., D + 1)``.

    """
    if ndim is not None:
        if point.shape[-1] == ndim + 1:
            return np.copy(point) if copy else point
        assert point.shape[-1] == ndim
    # Creating new array with submatrix assignment is faster than np.hstack!
    pts = np.ones(point.shape[0:-1] + (point.shape[-1] + 1,), dtype=point.dtype)
    pts[..., :-1] = point
    return pts


def homogeneous_matrix(transform: np.ndarray, ndim: int = None, copy: bool = True) -> np.ndarray:
    r"""Create homogeneous transformation matrix from affine coordinate transformation."""
    assert transform.ndim == 2
    rows, cols = transform.shape
    if ndim is None:
        ndim = rows
        assert (
            transform.shape[1] == ndim or transform.shape[1] == ndim + 1
        ), "transform.shape={}".format(transform.shape)
    elif rows == ndim + 1 and cols == ndim + 1:
        return np.copy(transform) if copy else transform
    matrix = np.eye(ndim + 1, ndim + 1, dtype=transform.dtype)
    matrix[0:rows, 0:cols] = transform
    return matrix


def translation_matrix(displacement: ArrayLike) -> np.ndarray:
    r"""Create translation matrix for homogeneous coordinates."""
    arg = np.asanyarray(displacement)
    assert arg.ndim == 1
    matrix = np.eye(arg.size + 1)
    matrix[0:-1, -1] = arg
    return matrix


def transform_point(matrix: np.ndarray, point: ArrayLike) -> np.ndarray:
    r"""Transform one or more points given a transformation matrix."""
    arg = np.asanyarray(point)
    dim = arg.shape[-1]
    mat = homogeneous_matrix(matrix, ndim=dim, copy=False)
    x = arg.reshape(-1, dim)
    y: np.ndarray = np.matmul(x, mat[0:-1, 0:-1].T) + np.expand_dims(mat[0:-1, -1], axis=0)
    return y.reshape(arg.shape)


def transform_vector(matrix: np.ndarray, vector: ArrayLike) -> np.ndarray:
    r"""Transform one or more vectors given a transformation matrix."""
    arg = np.asanyarray(vector)
    dim = arg.shape[-1]
    v = arg.reshape(-1, dim)
    u: np.ndarray = np.matmul(v, matrix[0:dim, 0:dim].T)
    return u.reshape(arg.shape)
