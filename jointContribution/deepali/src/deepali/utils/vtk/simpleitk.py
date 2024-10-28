r"""Bridge between VTK and SimpleITK."""
from typing import Optional
from typing import Sequence
from typing import Union

import SimpleITK as sitk
from deepali.utils.simpleitk.grid import GridAttrs
from deepali.utils.simpleitk.sample import interpolate_ndimage
from vtk import VTK_CHAR
from vtk import VTK_DOUBLE
from vtk import VTK_FLOAT
from vtk import VTK_LONG
from vtk import VTK_LONG_LONG
from vtk import VTK_SHORT
from vtk import VTK_UNSIGNED_CHAR
from vtk import VTK_UNSIGNED_LONG
from vtk import VTK_UNSIGNED_LONG_LONG
from vtk import VTK_UNSIGNED_SHORT
from vtk import vtkImageData
from vtk import vtkImageImport
from vtk import vtkPoints
from vtk import vtkPointSet

from .numpy import numpy_to_vtk_array
from .numpy import vtk_to_numpy_array
from .numpy import vtk_to_numpy_points

VTK_DATA_TYPE_FROM_SITK_PIXEL_ID = {
    sitk.sitkInt8: VTK_CHAR,
    sitk.sitkInt16: VTK_SHORT,
    sitk.sitkInt32: VTK_LONG,
    sitk.sitkInt64: VTK_LONG_LONG,
    sitk.sitkUInt8: VTK_UNSIGNED_CHAR,
    sitk.sitkUInt16: VTK_UNSIGNED_SHORT,
    sitk.sitkUInt32: VTK_UNSIGNED_LONG,
    sitk.sitkUInt64: VTK_UNSIGNED_LONG_LONG,
    sitk.sitkFloat32: VTK_FLOAT,
    sitk.sitkFloat64: VTK_DOUBLE,
    sitk.sitkVectorInt8: VTK_CHAR,
    sitk.sitkVectorInt16: VTK_SHORT,
    sitk.sitkVectorInt32: VTK_LONG,
    sitk.sitkVectorInt64: VTK_LONG_LONG,
    sitk.sitkVectorUInt8: VTK_UNSIGNED_CHAR,
    sitk.sitkVectorUInt16: VTK_UNSIGNED_SHORT,
    sitk.sitkVectorUInt32: VTK_UNSIGNED_LONG,
    sitk.sitkVectorUInt64: VTK_UNSIGNED_LONG_LONG,
    sitk.sitkVectorFloat32: VTK_FLOAT,
    sitk.sitkVectorFloat64: VTK_DOUBLE,
}


def apply_warp_field_to_points(
    warp_field: sitk.Image, points: vtkPoints, is_def_field: bool = False
) -> vtkPoints:
    r"""Transform vtkPoints by linearly interpolated dense vector field.

    Args:
        warp_field: Vector field that acts on the given points.
            If ``None``, an identity deformation is assumed.
        points: Input points in physical space of warp field.
        is_def_field: If ``True``, the input ``warp_field`` must be a vector field
            of output coordinates. Otherwise, ``warp_field`` must contain displacements
            in physical image space.

    Returns:
        Transformed output points.

    """
    out = points.NewInstance()

    if warp_field is None:
        out.DeepCopy(points)
        return out

    x = vtk_to_numpy_points(points)
    y = interpolate_ndimage(warp_field, x)
    if not is_def_field:
        y += x
    out.SetData(numpy_to_vtk_array(y))
    return out


def apply_warp_field_to_pointset(
    warp_field: sitk.Image, pointset: vtkPointSet, is_def_field: bool = False
) -> vtkPointSet:
    r"""Transform vtkPoints of vtkPointSet by linearly interpolated dense displacement field.

    Args:
        warp_field: Vector field that acts on the given points.
            If ``None``, an identity deformation is assumed.
        pointset: Input point set (e.g., vtkPolyData surface mesh).
        is_def_field: If ``True``, the input ``warp_field`` must be a vector field
            of output coordinates. Otherwise, ``warp_field`` must contain displacements
            in physical image space.

    Returns:
        Deep copy of input ``pointset`` with transformed points.

    """
    output = pointset.NewInstance()
    output.DeepCopy(pointset)
    if warp_field is not None:
        points = apply_warp_field_to_points(
            warp_field, pointset.GetPoints(), is_def_field=is_def_field
        )
        output.SetPoints(points)
    return output


def image_data_grid(data: vtkImageData) -> GridAttrs:
    r"""Create image grid from vtkImageData object."""
    extent = data.GetExtent()
    return GridAttrs(
        size=(extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1),
        origin=data.GetOrigin(),
        spacing=data.GetSpacing(),
    )


def vtk_image_from_sitk_image(
    image: sitk.Image, spacing: Optional[Union[float, Sequence[float]]] = None
) -> vtkImageData:
    r"""Create vtkImageData from SimpleITK image.

    VTK versions before 9.0 do not support orientation information. Therefore, use only the pixel spacing
    information for index to physical space transformations when working with VTK data structures. These
    can be mapped into the original physical space by applying the rotation and translation from the
    original image grid.

    Args:
        image: SimpleITK image.
        spacing: Data spacing to use instead of ``image.GetSpacing()``.

    Returns:
        vtk_image: vtkImageData instance with spacing set, but origin equal to (0, 0, 0).

    """
    data = sitk.GetArrayFromImage(image).tobytes()
    pixel_id = image.GetPixelIDValue()
    data_type = VTK_DATA_TYPE_FROM_SITK_PIXEL_ID[pixel_id]
    size = list(image.GetSize())
    if spacing is None:
        spacing = image.GetSpacing()
    elif isinstance(spacing, (int, float)):
        spacing = (spacing,) * len(size)
    spacing = list(spacing)
    while len(size) < 3:
        size.append(1)
        spacing.append(1.0)
    importer = vtkImageImport()
    importer.CopyImportVoidPointer(data, len(data))
    importer.SetDataScalarType(data_type)
    importer.SetNumberOfScalarComponents(image.GetNumberOfComponentsPerPixel())
    importer.SetDataSpacing(spacing)
    importer.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    importer.SetDataExtentToWholeExtent()
    importer.UpdateWholeExtent()
    output = vtkImageData()
    output.DeepCopy(importer.GetOutput())
    return output


def sitk_image_from_vtk_image(image: vtkImageData, grid: Optional[GridAttrs] = None) -> sitk.Image:
    r"""Create SimpleITK image from vtkImageData."""
    if image.GetNumberOfScalarComponents() != 1:
        raise NotImplementedError("sitk_image_from_vtk_image() only supports scalar 'image'")
    data = image.GetPointData().GetScalars()
    data = vtk_to_numpy_array(data)
    data = data.reshape(tuple(reversed(grid.size)))
    output = sitk.GetImageFromArray(data)
    output.SetOrigin(grid.origin)
    output.SetSpacing(grid.spacing)
    output.SetDirection(grid.direction)
    return output
