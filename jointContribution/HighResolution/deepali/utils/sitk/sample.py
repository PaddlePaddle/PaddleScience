from typing import Optional
from typing import Union

import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.spatial
import SimpleITK as sitk

from .grid import Grid


def resample_image(
    image: sitk.Image,
    reference: Union[Grid, sitk.Image],
    interpolator: int = sitk.sitkLinear,
    padding_value: float = 0,
) -> sitk.Image:
    """Interpolate image values at grid points of reference image.

    Args:
        image: Scalar or vector-valued image to evaluate at the specified points.
        reference: Sampling grid on which to evaluate interpolated image. If a path is specified,
        interpolator: Enumeration value of SimpleITK interpolator to use.
        padding_value: Output value when sampling point is outside the input image domain.

    Returns:
        output: Image interpolated at reference grid points.

    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(padding_value)
    if isinstance(reference, sitk.Image):
        resampler.SetReferenceImage(reference)
    else:
        resampler.SetSize(reference.size)
        resampler.SetOutputOrigin(reference.origin)
        resampler.SetOutputDirection(reference.direction)
        resampler.SetOutputSpacing(reference.spacing)
    return resampler.Execute(image)


def warp_image(
    image: sitk.Image,
    displacement: sitk.Image,
    reference: Optional[sitk.Image] = None,
    interpolator: int = sitk.sitkLinear,
    padding_value: float = 0,
) -> sitk.Image:
    """Interpolate image values at displaced output grid points.

    Args:
        image: Scalar or vector-valued image to evaluate at the specified points.
        displacement: Sampling grid on which to evaluate interpolated continuous image.
        interpolator: Enumeration value of SimpleITK interpolator to use.
        padding_value: Output value when sampling point is outside the input image domain.

    Returns:
        output: Image interpolated at displacement field grid points at the input image
            positions obtained by adding the given displacement to these grid coordinates.

    """
    if reference is None:
        reference = displacement
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(padding_value)
    resampler.SetReferenceImage(reference)
    assert resampler.GetSize() == reference.GetSize()
    disp_field = sitk.Cast(displacement, sitk.sitkVectorFloat64)
    transform = sitk.DisplacementFieldTransform(disp_field)
    resampler.SetTransform(transform)
    return resampler.Execute(image)


def interpolate_ndimage(
    image: sitk.Image, points: np.ndarray, padding_value: float = 0, order: int = 1
) -> np.ndarray:
    """Use ``scipy.ndimage.map_coordinates`` to interpolate image."""
    grid = Grid.from_image(image)
    idxs = np.moveaxis(grid.physical_space_to_continuous_index(points), -1, 0)
    idxs = np.flip(idxs, axis=0)
    vals = sitk.GetArrayViewFromImage(image)
    nval = image.GetNumberOfComponentsPerPixel()
    if nval > 1:
        out = np.stack(
            [
                scipy.ndimage.map_coordinates(
                    vals[..., c], idxs, cval=padding_value, order=order
                )
                for c in range(nval)
            ],
            axis=-1,
        )
    else:
        out = scipy.ndimage.map_coordinates(vals, idxs, cval=padding_value, order=order)
    return out


def interpolate_regular_grid(
    image: sitk.Image, points: np.ndarray, padding_value: float = 0
) -> np.ndarray:
    """Use ``scipy.interpolate.RegularGridInterpolator`` to interpolate image data."""
    size = tuple(points.shape)[0:-1]
    grid = Grid.from_image(image)
    vals = sitk.GetArrayViewFromImage(image)
    nval = image.GetNumberOfComponentsPerPixel()
    idxs = grid.physical_space_to_continuous_index(points.reshape(-1, grid.ndim))
    idxs = np.flip(idxs, axis=-1)
    coords = tuple(np.arange(tuple(vals.shape)[axis]) for axis in range(grid.ndim))
    if nval > 1:
        out = []
        for c in range(nval):
            func = scipy.interpolate.RegularGridInterpolator(
                coords, vals[..., c], bounds_error=False, fill_value=padding_value
            )
            out.append(func(idxs).reshape(size).astype(vals.dtype))
        out = np.stack(out, axis=-1)
    else:
        func = scipy.interpolate.RegularGridInterpolator(
            coords, vals, bounds_error=False, fill_value=padding_value
        )
        out = func(idxs).reshape(size).astype(vals.dtype)
    return out


def interpolate_griddata(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """Interpolate image similar to ``scipy.interpolate.griddata``.

    This method should only be used for comparison. The used Delaunay triangulation
    is not suited for interpolating image data sampled on a regular grid.

    Use ``warp_image`` (for regularly spaced ``points``) or ``interpolate_image``.
    """
    size = tuple(points.shape)[0:-1]
    grid = Grid.from_image(image)
    tess = scipy.spatial.qhull.Delaunay(grid.points.reshape(-1, grid.ndim))
    vals = sitk.GetArrayViewFromImage(image)
    nval = image.GetNumberOfComponentsPerPixel()
    coor = points.reshape(-1, grid.ndim)
    if nval > 1:
        out = []
        for c in range(nval):
            img = scipy.interpolate.LinearNDInterpolator(tess, vals[..., c].flatten())
            out.append(img(coor).reshape(size))
        out = np.stack(out, axis=-1)
    else:
        img = scipy.interpolate.LinearNDInterpolator(tess, vals.flatten())
        out = img(coor).reshape(size)
    return out
