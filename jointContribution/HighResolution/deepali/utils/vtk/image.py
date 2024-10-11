"""Auxiliary functions for working with vtkImageData."""
from typing import Optional

import numpy as np
from vtk import vtkImageData
from vtk import vtkImageStencilData
from vtk import vtkImageStencilToImage
from vtk import vtkMatrixToLinearTransform
from vtk import vtkPolyData
from vtk import vtkPolyDataToImageStencil
from vtk import vtkTransformPolyDataFilter

from ..sitk.grid import Grid
from .numpy import numpy_to_vtk_matrix4x4


def surface_mesh_grid(*mesh: vtkPolyData, resolution: Optional[float] = None) -> Grid:
    """Compute image grid for given surface mesh discretization with specified resolution."""
    bounds = np.zeros((6,), dtype=float)
    for pointset in mesh:
        _bounds = np.asarray(pointset.GetBounds())
        bounds[0::2] = np.minimum(bounds[0::2], _bounds[0::2])
        bounds[1::2] = np.maximum(bounds[1::2], _bounds[1::2])
    if resolution is None or resolution <= 0 or np.isnan(resolution):
        resolution = np.sqrt(np.sum(np.square(bounds[1::2] - bounds[0::2]))) / 256
    return Grid(
        size=np.ceil((bounds[1::2] - bounds[0::2]) / resolution).astype(int),
        origin=bounds[0::2] + 0.5 * resolution,
        spacing=np.asarray([resolution] * 3),
    )


def surface_image_stencil(mesh: vtkPolyData, grid: Grid) -> vtkImageStencilData:
    """Convert vtkPolyData surface mesh to image stencil."""
    rot = np.eye(4, dtype=np.float)
    rot[:3, :3] = np.array(grid.direction).reshape(3, 3)
    rot = numpy_to_vtk_matrix4x4(rot)
    transform = vtkMatrixToLinearTransform()
    transform.SetInput(rot)
    transformer = vtkTransformPolyDataFilter()
    transformer.SetInputData(mesh)
    transformer.SetTransform(transform)
    converter = vtkPolyDataToImageStencil()
    converter.SetInputConnection(transformer.GetOutputPort())
    converter.SetOutputOrigin(grid.origin)
    converter.SetOutputSpacing(grid.spacing)
    converter.SetOutputWholeExtent(
        [0, grid.size[0] - 1, 0, grid.size[1] - 1, 0, grid.size[2] - 1]
    )
    converter.Update()
    stencil = vtkImageStencilData()
    stencil.DeepCopy(converter.GetOutput())
    return stencil


def binary_image_stencil_mask(stencil: vtkImageStencilData) -> vtkImageData:
    """Set values inside image stencil to specified value."""
    converter = vtkImageStencilToImage()
    converter.SetInsideValue(1)
    converter.SetOutsideValue(0)
    converter.SetInputData(stencil)
    converter.SetOutputScalarTypeToUnsignedChar()
    converter.Update()
    mask = vtkImageData()
    mask.DeepCopy(converter.GetOutput())
    return mask
