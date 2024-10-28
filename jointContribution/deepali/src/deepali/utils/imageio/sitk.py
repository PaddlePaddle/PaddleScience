r"""Auxiliary functions for reading and writing images using SimpleITK."""
from typing import Tuple

import paddle
from deepali.core.grid import Grid
from deepali.core.pathlib import PathUri
from deepali.utils.simpleitk.imageio import read_image as _read_image
from deepali.utils.simpleitk.imageio import write_image as _write_image
from deepali.utils.simpleitk.torch import image_from_tensor
from deepali.utils.simpleitk.torch import tensor_from_image


def read_sitk_image(path: PathUri) -> Tuple[paddle.Tensor, Grid]:
    r"""Read any image file supported by SimpleITK."""
    image = _read_image(path)
    data = tensor_from_image(image)
    grid = Grid.from_sitk(image)
    return data, grid


def write_sitk_image(data: paddle.Tensor, grid: Grid, path: PathUri, compress: bool = True) -> None:
    r"""Write image file in any format supported by SimpleITK."""
    origin = grid.origin().tolist()
    spacing = grid.spacing().tolist()
    direction = grid.direction().flatten().tolist()
    image = image_from_tensor(data, origin=origin, spacing=spacing, direction=direction)
    _write_image(image, path, compress=compress)
