from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from . import affine as A
from .flow import warp_grid
from .flow import warp_points
from .grid import ALIGN_CORNERS
from .tensor import move_dim


def bounding_box(points: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute corners of minimum axes-aligned bounding box of given points."""
    return points.amin(axis=0), points.amax(axis=0)


def distance_matrix(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """Compute squared Euclidean distances between all pairs of points.

    Args:
        x: Point set of shape ``(N, X, D)``.
        y: Point set of shape ``(N, Y, D)``.

    Returns:
        paddle.Tensor of distance matrices of shape ``(N, X, Y)``.

    """
    if not isinstance(x, paddle.Tensor) or not isinstance(y, paddle.Tensor):
        raise TypeError("distance_matrix() 'x' and 'y' must be paddle.Tensor")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("distance_matrix() 'x' and 'y' must have shape (N, X, D)")
    N, _, D = tuple(x.shape)
    if tuple(y.shape)[0] != N:
        raise ValueError("distance_matrix() 'x' and 'y' must have same batch size N")
    if tuple(y.shape)[2] != D:
        raise ValueError(
            "distance_matrix() 'x' and 'y' must have same point dimension D"
        )
    out_dtype = x.dtype
    if not out_dtype.is_floating_point:
        out_dtype = "float32"
    x = x.astype("float64")
    y = y.astype("float64")
    x_norm = x.pow(y=2).sum(axis=2).view(N, -1, 1)
    y_norm = y.pow(y=2).sum(axis=2).view(N, 1, -1)
    x = y
    perm_8 = list(range(x.ndim))
    perm_8[1] = 2
    perm_8[2] = 1
    dist = x_norm + y_norm - 2.0 * paddle.bmm(x=x, y=paddle.transpose(x=x, perm=perm_8))
    return dist.astype(out_dtype)


def closest_point_distances(
    x: paddle.Tensor, y: paddle.Tensor, split_size: int = 10000
) -> paddle.Tensor:
    """Compute minimum Euclidean distance from each point in ``x`` to point set ``y``.

    Args:
        x: Point set of shape ``(N, X, D)``.
        y: Point set of shape ``(N, Y, D)``.
        split_size: Maximum number of points in ``x`` to consider each time when computing
            the full distance matrix between these points in ``x`` and every point in ``y``.
            This is required to limit the size of the distance matrix.

    Returns:
        paddle.Tensor of shape ``(N, X)`` with minimum distances from points in ``x`` to points in ``y``.

    """
    if not isinstance(x, paddle.Tensor) or not isinstance(y, paddle.Tensor):
        raise TypeError("closest_point_distances() 'x' and 'y' must be paddle.Tensor")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(
            "closest_point_distances() 'x' and 'y' must have shape (N, X, D)"
        )
    N, _, D = tuple(x.shape)
    if tuple(y.shape)[0] != N:
        raise ValueError(
            "closest_point_distances() 'x' and 'y' must have same batch size N"
        )
    if tuple(y.shape)[2] != D:
        raise ValueError(
            "closest_point_distances() 'x' and 'y' must have same point dimension D"
        )
    x = x.astype(dtype="float32")
    y = y.astype(x.dtype)
    min_dists = paddle.empty(shape=tuple(x.shape)[0:2], dtype=x.dtype)
    for i, points in enumerate(x.split(split_size, dim=1)):
        dists = distance_matrix(points, y)
        j = slice(i * split_size, i * split_size + tuple(points.shape)[1])
        min_dists[:, (j)] = (
            paddle.min(x=dists, axis=2),
            paddle.argmin(x=dists, axis=2),
        ).values
    return min_dists


def closest_point_indices(
    x: paddle.Tensor, y: paddle.Tensor, split_size: int = 10000
) -> paddle.Tensor:
    """Determine indices of points in ``y`` with minimum Euclidean distance from each point in ``x``.

    Args:
        x: Point set of shape ``(N, X, D)``.
        y: Point set of shape ``(N, Y, D)``.
        split_size: Maximum number of points in ``x`` to consider each time when computing
            the full distance matrix between these points in ``x`` and every point in ``y``.
            This is required to limit the size of the distance matrix.

    Returns:
        paddle.Tensor of shape ``(N, X)`` with indices of closest points in ``y``.

    """
    if not isinstance(x, paddle.Tensor) or not isinstance(y, paddle.Tensor):
        raise TypeError("closest_point_indices() 'x' and 'y' must be paddle.Tensor")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(
            "closest_point_indices() 'x' and 'y' must have shape (N, X, D)"
        )
    N, _, D = tuple(x.shape)
    if tuple(y.shape)[0] != N:
        raise ValueError(
            "closest_point_indices() 'x' and 'y' must have same batch size N"
        )
    if tuple(y.shape)[2] != D:
        raise ValueError(
            "closest_point_indices() 'x' and 'y' must have same point dimension D"
        )
    x = x.astype(dtype="float32")
    y = y.astype(x.dtype)
    indices = paddle.empty(shape=tuple(x.shape)[0:2], dtype="int64")
    for i, points in enumerate(x.split(split_size, dim=1)):
        dists = distance_matrix(points, y)
        j = slice(i * split_size, i * split_size + tuple(points.shape)[1])
        indices[:, (j)] = (
            paddle.min(x=dists, axis=2),
            paddle.argmin(x=dists, axis=2),
        ).indices
    return indices


def normalize_grid(
    grid: paddle.Tensor,
    size: Optional[Union[paddle.Tensor, list]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = True,
) -> paddle.Tensor:
    """Map unnormalized grid coordinates to normalized grid coordinates."""
    if not isinstance(grid, paddle.Tensor):
        raise TypeError("normalize_grid() 'grid' must be tensors")
    if not grid.is_floating_point():
        grid = grid.astype(dtype="float32")
    if size is None:
        if channels_last:
            if grid.ndim < 4 or tuple(grid.shape)[-1] != grid.ndim - 2:
                raise ValueError(
                    "normalize_grid() 'grid' must have shape (N, ..., X, D) when 'size' not given"
                )
            size = tuple(reversed(tuple(grid.shape)[1:-1]))
        else:
            if grid.ndim < 4 or tuple(grid.shape)[1] != grid.ndim - 2:
                raise ValueError(
                    "normalize_grid() 'grid' must have shape (N, D, ..., X) when 'size' not given"
                )
            size = tuple(reversed(tuple(grid.shape)[2:]))
    zero = paddle.to_tensor(data=0, dtype=grid.dtype, place=grid.place)
    size = paddle.to_tensor(data=size, dtype=grid.dtype, place=grid.place)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        grid = move_dim(grid, 1, -1)
    if side_length != 1:
        grid = grid.mul(side_length)
    grid = paddle.where(condition=size > 1, x=grid.div(size_).sub(1), y=zero)
    if not channels_last:
        grid = move_dim(grid, -1, 1)
    return grid


def denormalize_grid(
    grid: paddle.Tensor,
    size: Optional[Union[paddle.Tensor, list]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = True,
) -> paddle.Tensor:
    """Map normalized grid coordinates to unnormalized grid coordinates."""
    if not isinstance(grid, paddle.Tensor):
        raise TypeError("denormalize_grid() 'grid' must be tensors")
    if size is None:
        if grid.ndim < 4 or tuple(grid.shape)[-1] != grid.ndim - 2:
            raise ValueError(
                "normalize_grid() 'grid' must have shape (N, ..., X, D) when 'size' not given"
            )
        size = tuple(reversed(tuple(grid.shape)[1:-1]))
    zero = paddle.to_tensor(data=0, dtype=grid.dtype, place=grid.place)
    size = paddle.to_tensor(data=size, dtype=grid.dtype, place=grid.place)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        grid = move_dim(grid, 1, -1)
    grid = paddle.where(condition=size > 1, x=grid.add(1).mul(size_), y=zero)
    if side_length != 1:
        grid = grid.div(side_length)
    if not channels_last:
        grid = move_dim(grid, -1, 1)
    return grid


def polyline_directions(
    points: paddle.Tensor, normalize: bool = False, repeat_last: bool = True
) -> paddle.Tensor:
    """Compute proximal to distal facing tangent vectors."""
    if not isinstance(points, paddle.Tensor):
        raise TypeError("polyline_directions() 'points' must be paddle.Tensor")
    if points.ndim < 2:
        raise ValueError("polyline_directions() 'points' must have shape (..., N, 3)")
    dim = points.ndim - 2
    n = tuple(points.shape)[dim]
    start_10 = points.shape[dim] + 1 if 1 < 0 else 1
    start_11 = points.shape[dim] + 0 if 0 < 0 else 0
    d = paddle.slice(points, [dim], [start_10], [start_10 + n - 1]).sub(
        paddle.slice(points, [dim], [start_11], [start_11 + (n - 1)])
    )
    if normalize:
        d = paddle.nn.functional.normalize(x=d, p=2, axis=dim)
    if repeat_last:
        start_12 = d.shape[dim] + (n - 2) if n - 2 < 0 else n - 2
        d = paddle.concat(
            x=[d, paddle.slice(d, [dim], [start_12], [start_12 + 1])], axis=dim
        )
    return d


def polyline_tangents(
    points: paddle.Tensor, normalize: bool = False, repeat_first: bool = True
) -> paddle.Tensor:
    """Compute distal to proximal facing tangent vectors."""
    if not isinstance(points, paddle.Tensor):
        raise TypeError("polyline_tangents() 'points' must be paddle.Tensor")
    if points.ndim < 2:
        raise ValueError("polyline_tangents() 'points' must have shape (..., N, 3)")
    dim = points.ndim - 2
    n = tuple(points.shape)[dim]
    start_13 = points.shape[dim] + 0 if 0 < 0 else 0
    start_14 = points.shape[dim] + 1 if 1 < 0 else 1
    d = paddle.slice(points, [dim], [start_13], [start_13 + n - 1]).sub(
        paddle.slice(points, [dim], [start_14], [start_14 + (n - 1)])
    )
    if normalize:
        d = paddle.nn.functional.normalize(x=d, p=2, axis=dim)
    if repeat_first:
        start_15 = d.shape[dim] + 0 if 0 < 0 else 0
        d = paddle.concat(
            x=[paddle.slice(d, [dim], [start_15], [start_15 + 1]), d], axis=dim
        )
    return d


def transform_grid(
    transform: paddle.Tensor, grid: paddle.Tensor, align_corners: bool = ALIGN_CORNERS
) -> paddle.Tensor:
    """Transform undeformed grid by a spatial transformation.

    This function applies a spatial transformation to map a tensor of undeformed grid points to a
    tensor of deformed grid points with the same shape as the input tensor. The input points must be
    the positions of undeformed spatial grid points, because in case of a non-rigid transformation,
    this function uses interpolation to resize the vector fields to the size of the input ``grid``.
    This assumes that input points ``x`` are the coordinates of points located on a regularly spaced
    undeformed grid which is aligned with the borders of the grid domain on which the vector fields
    of the non-rigid transformations are sampled, i.e., ``y = x + u``.

    In case of a linear transformation ``y = Ax + t``.

    If in doubt whether the input points will be sampled regularly at grid points of the domain of
    the spatial transformation, use ``transform_points()`` instead.

    Args:
        transform: paddle.Tensor representation of spatial transformation, where the shape of the tensor
            determines the type of transformation. A translation-only transformation must be given
            as tensor of shape ``(N, D, 1)``. An affine-only transformation without translation can
            be given as tensor of shape ``(N, D, D)``, and an affine transformation with translation
            as tensor of shape ``(N, D, D + 1)``. Flow fields of non-rigid transformations, on the
            other hand, are tensors of shape ``(N, D, ..., X)``, i.e., linear transformations are
            represented by 3-dimensional tensors, and non-rigid transformations by tensors of at least
            4 dimensions. If batch size is one, but the batch size of ``points`` is greater than one,
            all point sets are transformed by the same non-rigid transformation.
        grid: Coordinates of undeformed grid points as tensor of shape ``(N, ..., D)`` or ``(1, ..., D)``.
            If batch size is one, but multiple flow fields are given, this single point set is
            transformed by each non-rigid transformation to produce ``N`` output point sets.
        align_corners: Whether flow vectors in case of a non-rigid transformation are with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True). The input ``grid`` points must be
            with respect to the same spatial grid domain as the input flow fields. This option is in
            particular passed on to the ``grid_reshape()`` function used to resize the flow fields to
            the shape of the input grid.

    Returns:
        paddle.Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    if not isinstance(transform, paddle.Tensor):
        raise TypeError("transform_grid() 'transform' must be paddle.Tensor")
    if transform.ndim < 3:
        raise ValueError(
            "transform_grid() 'transform' must be at least 3-dimensional tensor"
        )
    if transform.ndim == 3:
        return A.transform_points(transform, grid)
    return warp_grid(transform, grid, align_corners=align_corners)


def transform_points(
    transform: paddle.Tensor, points: paddle.Tensor, align_corners: bool = ALIGN_CORNERS
) -> paddle.Tensor:
    """Transform set of points by a tensor of non-rigid flow fields.

    This function applies a spatial transformation to map a tensor of points to a tensor of transformed
    points of the same shape as the input tensor. Unlike ``transform_grid()``, it can be used to spatially
    transform any set of points which are defined with respect to the grid domain of the spatial transformation,
    including a tensor of shape ``(N, M, D)``, i.e., a batch of N point sets with cardianality M. It can also
    be applied to a tensor of grid points of shape ``(N, ..., X, D)`` regardless if the grid points are located
    at the undeformed grid positions or an already deformed grid. Therefore, in case of a non-rigid transformation,
    the given flow fields are sampled at the input points ``x`` using linear interpolation. The flow vectors ``u(x)``
    are then added to the input points, i.e., ``y = x + u(x)``.

    In case of a linear transformation ``y = Ax + t``.

    Args:
        transform: paddle.Tensor representation of spatial transformation, where the shape of the tensor
            determines the type of transformation. A translation-only transformation must be given
            as tensor of shape ``(N, D, 1)``. An affine-only transformation without translation can
            be given as tensor of shape ``(N, D, D)``, and an affine transformation with translation
            as tensor of shape ``(N, D, D + 1)``. Flow fields of non-rigid transformations, on the
            other hand, are tensors of shape ``(N, D, ..., X)``, i.e., linear transformations are
            represented by 3-dimensional tensors, and non-rigid transformations by tensors of at least
            4 dimensions. If batch size is one, but the batch size of ``points`` is greater than one,
            all point sets are transformed by the same non-rigid transformation.
        points: Coordinates of points given as tensor of shape ``(N, ..., D)`` or ``(1, ..., D)``.
            If batch size is one, but multiple flow fields are given, this single point set is
            transformed by each non-rigid transformation to produce ``N`` output point sets.
        align_corners: Whether flow vectors in case of a non-rigid transformation are with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True). The input ``points`` must be
            with respect to the same spatial grid domain as the input flow fields. This option is in
            particular passed on to the ``grid_sample()`` function used to sample the flow vectors at
            the input points.

    Returns:
        paddle.Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    if not isinstance(transform, paddle.Tensor):
        raise TypeError("transform_points() 'transform' must be paddle.Tensor")
    if transform.ndim < 3:
        raise ValueError(
            "transform_points() 'transform' must be at least 3-dimensional tensor"
        )
    if transform.ndim == 3:
        return A.transform_points(transform, points)
    return warp_points(transform, points, align_corners=align_corners)
