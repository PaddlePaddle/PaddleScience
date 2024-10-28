r"""Auxiliary functions for reading and writing NIfTI image files."""
import sys
from typing import Tuple

import paddle

try:
    import nibabel as nib
except ImportError:
    nib = None
import numpy as np
from deepali.core.grid import Grid
from deepali.core.pathlib import PathUri
from deepali.core.pathlib import path_suffix
from deepali.core.pathlib import unlink_or_mkdir
from deepali.core.storage import StorageObject

NIFTI_IMAGE_SUFFIXES = (".nia", ".nii", ".nii.gz", ".hdr", ".hdr.gz", ".img", ".img.gz")


def has_nifti_image_suffix(path: PathUri) -> bool:
    r"""Check whether filename ends in a NIfTI image extension."""
    return path_suffix(path).lower() in NIFTI_IMAGE_SUFFIXES


def read_nifti_image(path: PathUri) -> Tuple[paddle.Tensor, Grid]:
    r"""Read image data and grid attributes from NIfTI file."""
    if not has_nifti_image_suffix(path):
        raise ValueError(f"NIfTI image filename suffix must be one of {NIFTI_IMAGE_SUFFIXES}")
    if nib is None:
        try:
            from .sitk import read_sitk_image
        except ImportError:
            raise RuntimeError("nibabel or SimpleITK is required for reading NIfTI images")
        return read_sitk_image(path)
    # Load image
    with StorageObject.from_path(path) as obj:
        obj.pull(force=True)
        image = nib.load(obj.path)
    # Image sampling grid attributes
    dim = np.asarray(image.header["dim"])
    ndim = int(dim[0])
    D = min(ndim, 3)
    size = dim[1 : D + 1]
    spacing = np.asarray(image.header["pixdim"][1 : D + 1])
    affine = np.asarray(image.affine)
    origin = affine[:D, 3]
    direction = np.divide(affine[:D, :D], spacing)
    # Convert to ITK LPS convention
    origin[:2] *= -1
    direction[:2] *= -1
    # Replace small values and -0 by 0
    epsilon = sys.float_info.epsilon
    origin[np.abs(origin) < epsilon] = 0
    direction[np.abs(direction) < epsilon] = 0
    # Image data array
    slope = image.dataobj.slope
    inter = image.dataobj.inter
    if abs(slope) > epsilon and (abs(slope - 1) > epsilon or abs(inter) > epsilon):
        data: np.ndarray = image.get_fdata()
    else:
        data: np.ndarray = image.dataobj.get_unscaled()
    # Squeeze unused dimensions
    # https://github.com/InsightSoftwareConsortium/ITK/blob/3454d857dc46e4333ad1178be8c186547fba87ef/Modules/IO/NIFTI/src/itkNiftiImageIO.cxx#L1112-L1156
    intent_code = int(image.header["intent_code"])
    if intent_code in (1005, 1006, 1007):
        # Vector or matrix valued image
        for realdim in range(4, 1, -1):
            if dim[realdim] > 1:
                break
        else:
            realdim = 1
    elif intent_code == 1004:
        raise NotImplementedError(
            f"{path} has an intent code of NIFTI_INTENT_GENMATRIX which is not yet implemented"
        )
    else:
        # Scalar image
        realdim = ndim
        while realdim > 3 and dim[realdim] == 1:
            realdim -= 1
    data = np.reshape(data, data.shape[:realdim] + data.shape[5:])
    # Reverse order of axes
    data = np.transpose(data, axes=tuple(reversed(range(data.ndim))))
    # Add leading channel dimension
    grid = Grid(size=size, origin=origin, spacing=spacing, direction=direction)
    if data.ndim == grid.ndim:
        data = np.expand_dims(data, 0)
    if data.dtype == np.uint16:
        data = data.astype(np.int32)
    elif data.dtype == np.uint32:
        data = data.astype(np.int64)
    return paddle.to_tensor(data=data), grid


def write_nifti_image(data: paddle.Tensor, grid: Grid, path: PathUri) -> None:
    r"""Write image data and grid attributes to NIfTI image file."""
    if not has_nifti_image_suffix(path):
        raise ValueError(f"NIfTI image filename suffix must be one of {NIFTI_IMAGE_SUFFIXES}")
    if nib is None:
        try:
            from .sitk import write_sitk_image
        except ImportError:
            raise RuntimeError("nibabel or SimpleITK is required for writing NIfTI images")
        return write_sitk_image(data, grid, path, compress=str(path).lower().endswith(".gz"))
    data = data.detach().cpu()
    if data.ndim == grid.ndim:
        data = data.unsqueeze(axis=0)
    if data.ndim != grid.ndim + 1:
        raise ValueError("write_image() data.ndim must be equal to grid.ndim or grid.ndim + 1")
    # Reverse order of axes
    dataobj = np.transpose(data.numpy(), axes=tuple(reversed(range(data.ndim))))
    # Convert to NIfTI RAS convention
    affine = grid.affine().cpu().numpy()
    affine[:2] *= -1
    with StorageObject.from_path(path) as obj:
        local_path = unlink_or_mkdir(obj.path)
        nib.save(nib.Nifti1Image(dataobj, affine), str(local_path))
        obj.push(force=True)
