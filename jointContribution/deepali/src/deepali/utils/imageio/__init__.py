r"""Auxiliary functions for reading and writing image files."""
from typing import Tuple

import paddle
from deepali.core.grid import Grid
from deepali.core.pathlib import PathUri

from .meta import has_meta_image_suffix
from .meta import read_meta_image
from .meta import write_meta_image
from .nifti import has_nifti_image_suffix
from .nifti import read_nifti_image
from .nifti import write_nifti_image


def read_image(path: PathUri) -> Tuple[paddle.Tensor, Grid]:
    r"""Read image data from specified file path.

    Args:
        path: Input file path.

    Returns:
        data: Image data tensor.
        grid: Image sampling grid.

    """
    if has_meta_image_suffix(path):
        return read_meta_image(path)
    if has_nifti_image_suffix(path):
        return read_nifti_image(path)
    try:
        from .sitk import read_sitk_image
    except ImportError:
        raise RuntimeError(
            f"Cannot read image {path}. Image file formats other than MetaImage and NIfTI require SimpleITK to be installed."
        )
    return read_sitk_image(path)


def write_image(data: paddle.Tensor, grid: Grid, path: PathUri, compress: bool = True) -> None:
    r"""Write image data to specified file path.

    Args:
        data: Image data tensor.
        grid: Image sampling grid.

    """
    if has_meta_image_suffix(path):
        return write_meta_image(data, grid, path, compress=compress)
    if has_nifti_image_suffix(path):
        return write_nifti_image(data, grid, path)
    try:
        from .sitk import write_sitk_image
    except ImportError:
        raise RuntimeError(
            f"Cannot write image {path}. Image file formats other than MetaImage and NIfTI require SimpleITK to be installed."
        )
    return write_sitk_image(data, grid, path, compress=compress)
