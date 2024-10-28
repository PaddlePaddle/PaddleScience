from typing import Optional
from typing import Sequence

import numpy as np
import SimpleITK as sitk


def image_dtype(image: sitk.Image) -> np.dtype:
    r"""Get NumPy data type of SimpleITK image."""
    return sitk.GetArrayViewFromImage(image).dtype


def image_from_array(
    values: np.ndarray,
    origin: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
) -> sitk.Image:
    r"""Create image from sampling grid and image data array."""
    components = 1
    ndim = 0
    if origin:
        ndim = len(origin)
    elif spacing:
        ndim = len(spacing)
    elif direction:
        ndim = int(np.sqrt(len(direction)))
        assert len(direction) == ndim * ndim
    if ndim:
        if values.ndim < ndim:
            values = values.reshape((1,) * (ndim - values.ndim) + values.shape)
        elif values.ndim > ndim:
            values = values.reshape(values.shape[:ndim] + (-1,))
            components = values.shape[-1]
        assert ndim <= values.ndim <= ndim + 1
    image = sitk.GetImageFromArray(values, isVector=components > 1)
    if origin:
        image.SetOrigin(origin)
    if spacing:
        image.SetSpacing(spacing)
    if direction:
        image.SetDirection(direction)
    return image


def array_from_image(image: sitk.Image, view: bool = False) -> np.ndarray:
    r"""Get NumPy array with copy of image values."""
    return sitk.GetArrayViewFromImage(image) if view else sitk.GetArrayFromImage(image)
