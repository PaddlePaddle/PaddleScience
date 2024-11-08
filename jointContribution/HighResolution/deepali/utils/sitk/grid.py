import itertools
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import SimpleITK as sitk

Coords = Union[np.ndarray, Sequence[float]]


class Grid(object):
    """Finite and discrete image sampling grid oriented in world space."""

    __slots__ = ["_size", "origin", "spacing", "direction"]

    def __init__(
        self,
        size: Tuple[Union[int, float], ...],
        origin: Optional[Tuple[float, ...]] = None,
        spacing: Optional[Tuple[float, ...]] = None,
        direction: Optional[Tuple[float, ...]] = None,
    ):
        ndim = len(size)
        if origin is None:
            origin = (0.0,) * ndim
        elif not isinstance(origin, tuple):
            origin = tuple(origin)
        if spacing is None:
            spacing = (1.0,) * ndim
        elif not isinstance(spacing, tuple):
            spacing = tuple(spacing)
        if direction is None:
            direction = np.eye(ndim, ndim)
        if isinstance(direction, np.ndarray):
            direction = direction.flatten().astype(float)
        if not isinstance(direction, tuple):
            direction = tuple(direction)
        self._size = tuple(float(n) for n in size)
        self.origin = origin
        self.spacing = spacing
        self.direction = direction

    @property
    def ndim(self) -> int:
        """Number of image grid dimensions."""
        return len(self.shape)

    @property
    def npts(self) -> int:
        """Total number of image grid points."""
        return int(np.prod(self.size))

    @staticmethod
    def _int_size(size) -> Tuple[int, ...]:
        """Get grid size from internal floating point representation."""
        return tuple([(int(n + 0.5) if n > 1 or n <= 0 else 1) for n in size])

    @property
    def size(self) -> Tuple[int, ...]:
        """Size of image data array."""
        return self._int_size(self._size)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of image data array."""
        return tuple(reversed(self.size))

    @classmethod
    def from_file(cls, path: Union[Path, str]):
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return cls.from_reader(reader)

    @classmethod
    def from_reader(cls, reader: sitk.ImageFileReader):
        return cls(
            size=reader.GetSize(),
            origin=reader.GetOrigin(),
            spacing=reader.GetSpacing(),
            direction=reader.GetDirection(),
        )

    @classmethod
    def from_image(cls, image: sitk.Image):
        return cls(
            size=image.GetSize(),
            origin=image.GetOrigin(),
            spacing=image.GetSpacing(),
            direction=image.GetDirection(),
        )

    def zeros_image(self, dtype: int = sitk.sitkInt16, channels: int = 1):
        """Create empty image from grid."""
        img = sitk.Image(self.size, dtype, channels)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction)
        img.SetSpacing(self.spacing)
        return img

    def with_margin(self, margin: int):
        """Create new image grid with an additional margin along each grid axis."""
        if not margin:
            return self
        return self.__class__(
            size=tuple(n + 2 * margin for n in self._size),
            origin=self.index_to_physical_space((-margin,) * self.ndim),
            spacing=self.spacing,
            direction=self.direction,
        )

    def with_spacing(self, *args: float):
        """Create new image grid with specified spacing."""
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
        return self.__class__(
            size=size, origin=origin, spacing=spacing, direction=self.direction
        )

    def down(self, levels: int = 1):
        """Create new image grid of half the size."""
        size = self._size
        for _ in range(levels):
            size = tuple(n / 2 for n in size)
        cur_size = self.size
        new_size = self._int_size(size)
        spacing = tuple(
            self.spacing[i] * cur_size[i] / new_size[i]
            if new_size[i] > 0
            else self.spacing[i]
            for i in range(self.ndim)
        )
        return self.__class__(size=size, spacing=spacing)

    def up(self, levels: int = 1):
        """Create new image grid of double the size."""
        size = self._size
        for _ in range(levels):
            size = tuple(2 * n for n in size)
        cur_size = self.size
        new_size = self._int_size(size)
        spacing = tuple(
            self.spacing[i] * cur_size[i] / new_size[i]
            if new_size[i] > 0
            else self.spacing[i]
            for i in range(self.ndim)
        )
        return self.__class__(size=size, spacing=spacing)

    @property
    def dcm(self) -> np.ndarray:
        """Get direction cosine matrix."""
        return np.array(self.direction).reshape(self.ndim, self.ndim)

    @property
    def transform(self) -> np.ndarray:
        """Get homogeneous coordinate transformation from image grid to world space."""
        rotation = self.dcm
        scaling = np.diag(self.spacing)
        matrix = homogeneous_matrix(rotation @ scaling)
        matrix[0:-1, (-1)] = self.origin
        return matrix

    @property
    def inverse_transform(self) -> np.ndarray:
        """Get homogeneous coordinate transformation from world space to image grid."""
        rotation = self.dcm.T
        scaling = np.diag([(1 / s) for s in self.spacing])
        translation = translation_matrix([(-t) for t in self.origin])
        return homogeneous_matrix(scaling @ rotation) @ translation

    @property
    def indices(self) -> np.ndarray:
        """Get array of image grid point coordinates in image space."""
        return np.flip(
            np.stack(
                np.meshgrid(*[np.arange(arg) for arg in self.shape], indexing="ij"),
                axis=self.ndim,
            ),
            axis=-1,
        )

    def axis_indices(self, axis: int) -> np.ndarray:
        """Get array of image grid indices along specified axis."""
        return np.arange(self.shape[axis])

    @property
    def points(self) -> np.ndarray:
        """Get array of image grid point coordinates in world space."""
        return self.index_to_physical_space(self.indices)

    @property
    def coords(self) -> Tuple[np.ndarray, ...]:
        """Get 1D arrays of grid point coordinates along each axis."""
        return tuple(self.axis_coords(axis) for axis in range(self.ndim))

    def axis_coords(self, axis: int) -> np.ndarray:
        """Get array of image grid point coordinates in world space along specified axis."""
        indices = [[0]] * self.ndim
        indices[axis] = [index for index in range(self.shape[axis])]
        mesh = np.stack(np.meshgrid(*indices, indexing="ij"), axis=self.ndim)
        return self.index_to_physical_space(mesh)[..., axis].flatten()

    @property
    def corners(self) -> np.ndarray:
        """Get corners of image domain in world space."""
        limits = []
        for axis in range(self.ndim):
            limits.append((0, self.shape[axis]))
        corners = [tuple(reversed(indices)) for indices in itertools.product(*limits)]
        return self.index_to_physical_space(corners)

    def index_to_physical_space(self, points: Coords) -> np.ndarray:
        """Map point coordinates from image to world space."""
        return transform_point(self.transform, points)

    def physical_space_to_index(self, points: Coords) -> np.ndarray:
        """Map point coordinates from world to discrete image space."""
        return np.round(self.physical_space_to_continuous_index(points)).astype(int)

    def physical_space_to_continuous_index(self, points: Coords) -> np.ndarray:
        """Map point coordinates from world to continuous image space."""
        return np.round(transform_point(self.inverse_transform, points), decimals=12)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(size={size}, origin={origin}, spacing={spacing}, direction={direction})".format(
                size=self.size,
                origin=self.origin,
                spacing=self.spacing,
                direction=self.direction,
            )
        )


def homogeneous_coords(
    point: np.ndarray, ndim: int = None, copy: bool = True
) -> np.ndarray:
    """Create array with homogeneous point coordinates."""
    if ndim is not None:
        if tuple(point.shape)[-1] == ndim + 1:
            return np.copy(point) if copy else point
        assert tuple(point.shape)[-1] == ndim
    pts = np.ones(
        tuple(point.shape)[0:-1] + (tuple(point.shape)[-1] + 1,), dtype=point.dtype
    )
    pts[(...), :-1] = point
    return pts


def homogeneous_matrix(
    transform: np.ndarray, ndim: int = None, copy: bool = True
) -> np.ndarray:
    """Create homogeneous transformation matrix from affine coordinate transformation."""
    assert transform.ndim == 2
    rows, cols = tuple(transform.shape)
    if ndim is None:
        ndim = rows
        assert (
            tuple(transform.shape)[1] == ndim or tuple(transform.shape)[1] == ndim + 1
        ), "transform.shape={}".format(tuple(transform.shape))
    elif rows == ndim + 1 and cols == ndim + 1:
        return np.copy(transform) if copy else transform
    matrix = np.eye(ndim + 1, ndim + 1, dtype=transform.dtype)
    matrix[0:rows, 0:cols] = transform
    return matrix


def translation_matrix(displacement: Sequence[float]) -> np.ndarray:
    """Create translation matrix for homogeneous coordinates."""
    displacement = np.asanyarray(displacement)
    assert displacement.ndim == 1
    matrix = np.eye(displacement.size + 1)
    matrix[0:-1, (-1)] = displacement
    return matrix


def transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Transform one or more points given a transformation matrix."""
    pts = np.asanyarray(point)
    dim = tuple(pts.shape)[-1]
    mat = homogeneous_matrix(matrix, ndim=dim, copy=False)
    x = pts.reshape(-1, dim)
    y = np.matmul(x, mat[0:-1, 0:-1].T) + np.expand_dims(mat[0:-1, (-1)], axis=0)
    return y.reshape(tuple(pts.shape))


def transform_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Transform one or more vectors given a transformation matrix."""
    vec = np.asanyarray(vector)
    dim = tuple(vec.shape)[-1]
    v = vec.reshape(-1, dim)
    u = np.matmul(v, matrix[0:dim, 0:dim].T)
    return u.reshape(tuple(vec.shape))
