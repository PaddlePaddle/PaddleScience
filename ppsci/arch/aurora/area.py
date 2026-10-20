"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import paddle

__all__ = ["area", "compute_patch_areas", "radius_earth"]
radius_earth = 6378137 / 1000
"""float: Radius of the earth in kilometers."""


def area(polygon: paddle.Tensor) -> paddle.Tensor:
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    This function is a Paddle port of the PyPI package `area`. In particular, it is heavily
    inspired by the following file:

        https://github.com/scisco/area/blob/9d9549d6ebffcbe4bffe11b71efa2d406d1c9fe9/area/__init__.py

    Args:
        polygon (:class:`paddle.Tensor`): Polygon of the shape `(*b, n, 2)` where `b` is an optional
            multidimensional batch size, `n` is the number of points of the polygon, and 2
            concatenates first latitudes and then longitudes. The polygon does not have be closed.

    Returns:
        :class:`paddle.Tensor`: Area in square kilometers.
    """
    polygon = paddle.cat((polygon, polygon[..., -1:, :]), axis=-2)
    area = paddle.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]
    rad = paddle.deg2rad
    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n
            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]
            area = area + (rad(lon_upper) - rad(lon_lower)) * paddle.sin(
                rad(lat_middle)
            )
    area = area * radius_earth * radius_earth / 2
    return paddle.abs(area)


def expand_matrix(matrix: paddle.Tensor) -> paddle.Tensor:
    """Expand matrix by adding one row and one column to each side, using
    linear interpolation.

    Args:
        matrix (:class:`paddle.Tensor`): Matrix to expand.

    Returns:
        :class:`paddle.Tensor`: `matrix`, but with two extra rows and two extra columns.
    """
    matrix = paddle.cat(
        (2 * matrix[0:1] - matrix[1:2], matrix, 2 * matrix[-1:] - matrix[-2:-1]), axis=0
    )
    matrix = paddle.cat(
        (
            2 * matrix[:, 0:1] - matrix[:, 1:2],
            matrix,
            2 * matrix[:, -1:] - matrix[:, -2:-1],
        ),
        axis=1,
    )
    return matrix


def compute_patch_areas(lat: paddle.Tensor, lon: paddle.Tensor) -> paddle.Tensor:
    """A pair of latitude and longitude matrices defines a number non-intersecting patches on the
    Earth. For a global grid, these patches span the entire surface of the Earth. For a local grid,
    the patches might span only a country or a continent. This function computes the area of every
    specified patch.

    To divide the Earth into patches, the idea is to let a grid point be the _center_ of the
    corresponding patch. The vertices of this patch will then sit exactly inbetween the grid
    point and the grid points immediately diagonally and non-diagonally above, below, left, and
    right. For a grid point at the very top of the grid, for example, there is no immediately above
    grid point. In that case, we enlarge the grid by a row at the top by linearly interpolating the
    latitudinal progression.

    Summary of algorithm:
    1. Enlarge the latitude and longitude matrices by adding one row and one column to each side.
    2. Calculate the patch vertices by averaging every 2x2 square in the enlarged grid. We also
        call these points the midpoints.
    3. By using the vertices of the patches, i.e. the midpoints, compute the areas of the patches.

    Args:
        lat (:class:`paddle.Tensor`): Latitude matrix. Must be decreasing along rows.
        lon (:class:`paddle.Tensor`): Longitude matrix. Must be increasing along columns.

    Returns:
        :class:`paddle.Tensor`: Areas in square kilometer.
    """
    if not lat.dim() == lon.dim() == 2:
        raise ValueError("`lat` and `lon` must both be matrices.")
    if lat.shape != lon.shape:
        raise ValueError("`lat` and `lon` must have the same shape.")
    if not paddle.all(lat[1:] - lat[:-1] <= 0):
        raise ValueError("`lat` must be decreasing along rows.")
    if not paddle.all(lon[:, 1:] - lon[:, :-1] >= 0):
        raise ValueError("`lon` must be increasing along columns.")
    lat = expand_matrix(lat)
    lon = expand_matrix(lon)
    lat = paddle.clamp(lat, -90, 90)
    lat_midpoints = (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:]) / 4
    lon_midpoints = (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:]) / 4
    top_left = paddle.stack((lat_midpoints[1:, :-1], lon_midpoints[1:, :-1]), axis=-1)
    top_right = paddle.stack((lat_midpoints[1:, 1:], lon_midpoints[1:, 1:]), axis=-1)
    bottom_left = paddle.stack(
        (lat_midpoints[:-1, :-1], lon_midpoints[:-1, :-1]), axis=-1
    )
    bottom_right = paddle.stack(
        (lat_midpoints[:-1, 1:], lon_midpoints[:-1, 1:]), axis=-1
    )
    polygon = paddle.stack((top_left, top_right, bottom_right, bottom_left), axis=-2)
    return area(polygon)
