from pathlib import Path
from typing import Union

import SimpleITK as sitk

PathStr = Union[Path, str]


def read_image(path: PathStr) -> sitk.Image:
    """Read image from file."""
    path = Path(path).absolute()
    if not path.exists():
        raise FileNotFoundError(f"Image file '{path}' does not exist")
    return sitk.ReadImage(str(path))


def write_image(image: sitk.Image, path: PathStr, compress: bool = True):
    """Write image to file."""
    path = Path(path).absolute()
    try:
        path.unlink()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
    return sitk.WriteImage(image, str(path), compress)
