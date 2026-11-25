"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Parts of this code are inspired by

    https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/utils/pos_embed.py
"""

from __future__ import annotations

import paddle

from ppsci.arch.aurora._layers import to_2tuple
from ppsci.arch.aurora.model.fourier import FourierExpansion

__all__ = ["pos_scale_enc"]


def _patch_average(grid: paddle.Tensor, patch_dims: tuple) -> paddle.Tensor:
    """Average coordinates in the upstream CUDA kernel's row-major order.

    Paddle's pooled reduction may round differently by one float32 ULP.
    Aurora's shortest spatial wavelength is 0.01 degrees, so those coordinate
    errors noticeably change Fourier features. Keep the original accumulation
    order using native differentiable tensor operations.
    """
    ph, pw = patch_dims
    height = grid.shape[-2] // ph * ph
    width = grid.shape[-1] // pw * pw
    if height == 0 or width == 0:
        raise ValueError("The coordinate grid must contain at least one full patch")
    grid = grid[..., :height, :width]
    total = paddle.zeros_like(grid[..., ::ph, ::pw])
    for row in range(ph):
        for col in range(pw):
            total = total + grid[..., row::ph, col::pw]
    divisor = paddle.full([], ph * pw, dtype=grid.dtype, device=grid.device)
    return paddle.divide(total, divisor)


def patch_root_area(
    lat_min: paddle.Tensor,
    lon_min: paddle.Tensor,
    lat_max: paddle.Tensor,
    lon_max: paddle.Tensor,
) -> paddle.Tensor:
    """For a rectangular patch on a sphere, compute the square root of the area of the patch in
    units km^2. The root is taken to return units of km, and thus stay scalable between different
    resolutions.

    Args:
        lat_min (paddle.Tensor): Minimum latitutes of patches.
        lon_min (paddle.Tensor): Minimum longitudes of patches.
        lat_max (paddle.Tensor): Maximum latitudes of patches.
        lon_max (paddle.Tensor): Maximum longitudes of patches.

    Returns:
        paddle.Tensor: Square root of the area.
    """
    assert (
        lat_max > lat_min
    ).all(), f"lat_max - lat_min: {paddle.min(lat_max - lat_min)}."
    assert (
        lon_max > lon_min
    ).all(), f"lon_max - lon_min: {paddle.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = (
        6371**2
        * paddle.pi
        * (paddle.sin(paddle.deg2rad(lat_max)) - paddle.sin(paddle.deg2rad(lat_min)))
        * (paddle.deg2rad(lon_max) - paddle.deg2rad(lon_min))
    )
    assert (area > 0.0).all()
    return paddle.sqrt(area)


def pos_scale_enc_grid(
    encode_dim: int,
    grid: paddle.Tensor,
    patch_dims: tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute the position and scale encoding for a latitude-longitude grid.

    Requires batch dimensions in the input and returns a batch dimension.

    Args:
        encode_dim (int): Output encoding dimension `D`. Must be a multiple of four: splits
            across latitudes and longitudes and across sines and cosines.
        grid (paddle.Tensor): Latitude-longitude grid of dimensions `(B, 2, H, W)`. `grid[:, 0]`
            should be the latitudes of `grid[:, 1]` should be the longitudes.
        patch_dims (tuple): Patch dimensions. Different x-values and y-values are supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[paddle.Tensor, paddle.Tensor]: Positional encoding and scale encoding of shape
            `(B, H/patch[0] * W/patch[1], D)`.
    """
    assert encode_dim % 4 == 0
    assert grid.dim() == 4
    midpoints = _patch_average(grid, patch_dims)
    grid_h, grid_w = midpoints[:, 0], midpoints[:, 1]
    grid_lat_max = paddle.nn.functional.max_pool2d(grid[:, 0:1], patch_dims).squeeze(1)
    grid_lat_min = -paddle.nn.functional.max_pool2d(-grid[:, 0:1], patch_dims).squeeze(
        1
    )
    grid_lon_max = paddle.nn.functional.max_pool2d(grid[:, 1:2], patch_dims).squeeze(1)
    grid_lon_min = -paddle.nn.functional.max_pool2d(-grid[:, 1:2], patch_dims).squeeze(
        1
    )
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)
    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)
    pos_encode = paddle.cat((encode_h, encode_w), axis=-1)
    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)
    return pos_encode, scale_encode


def lat_lon_meshgrid(lat: paddle.Tensor, lon: paddle.Tensor) -> paddle.Tensor:
    """Construct a meshgrid of latitude and longitude coordinates.

    The returned axes are latitude then longitude. Internally the ``xy``
    meshgrid is transposed so every field follows the input grid convention.

    Args:
        lat (paddle.Tensor): Vector of latitudes.
        lon (paddle.Tensor): Vector of longitudes.

    Returns:
        paddle.Tensor: Meshgrid of shape `(2, len(lat), len(lon))`.
    """
    assert lat.dim() == 1
    assert lon.dim() == 1
    grid = paddle.meshgrid(lat, lon, indexing="xy")
    grid = paddle.stack(grid, axis=0)
    grid = grid.permute(0, 2, 1)
    return grid


def pos_scale_enc(
    encode_dim: int,
    lat: paddle.Tensor,
    lon: paddle.Tensor,
    patch_dims: (int | list | tuple),
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> paddle.Tensor:
    """Positional encoding of latitude-longitude data.

    Does not support batch dimensions in the input and does not return batch dimensions either.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        lat (paddle.Tensor): Latitudes, `H`. Can be either a vector or a matrix.
        lon (paddle.Tensor): Longitudes, `W`. Can be either a vector or a matrix.
        patch_dims (Union[list, tuple]): Patch dimensions. Different x-values and y-values are
            supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[paddle.Tensor, paddle.Tensor]: Positional encoding and scale encoding of shape
            `(H/patch[0] * W/patch[1], D)`.
    """
    if lat.dim() == lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == lon.dim() == 2:
        grid = paddle.stack((lat, lon), axis=0)
    else:
        raise ValueError(
            f"Latitudes and longitudes must either both be vectors or both be matrices, but have dimensionalities {lat.dim()} and {lon.dim()} respectively."
        )
    grid = grid[None]
    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )
    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)
