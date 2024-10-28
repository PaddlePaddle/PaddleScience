r"""Bridge between VTK and NumPy."""
import warnings
from typing import Optional
from typing import Union

import numpy as np
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
from vtk import vtkCellArray
from vtk import vtkDataArray
from vtk import vtkMatrix4x4
from vtk import vtkPoints
from vtk import vtkPointSet
from vtk.util import numpy_support  # type: ignore

VTK_DATA_TYPE_FROM_NUMPY_DTYPE = {
    np.dtype("int8"): VTK_CHAR,
    np.dtype("int16"): VTK_SHORT,
    np.dtype("int32"): VTK_LONG,
    np.dtype("int64"): VTK_LONG_LONG,
    np.dtype("uint8"): VTK_UNSIGNED_CHAR,
    np.dtype("uint16"): VTK_UNSIGNED_SHORT,
    np.dtype("uint32"): VTK_UNSIGNED_LONG,
    np.dtype("uint64"): VTK_UNSIGNED_LONG_LONG,
    np.dtype("float32"): VTK_FLOAT,
    np.dtype("float64"): VTK_DOUBLE,
}


NUMPY_DTYPE_FROM_VTK_DATA_TYPE = {
    VTK_CHAR: np.dtype("int8"),
    VTK_SHORT: np.dtype("int16"),
    VTK_LONG: np.dtype("int32"),
    VTK_LONG_LONG: np.dtype("int64"),
    VTK_UNSIGNED_CHAR: np.dtype("uint8"),
    VTK_UNSIGNED_SHORT: np.dtype("uint16"),
    VTK_UNSIGNED_LONG: np.dtype("uint32"),
    VTK_UNSIGNED_LONG_LONG: np.dtype("uint64"),
    VTK_FLOAT: np.dtype("float32"),
    VTK_DOUBLE: np.dtype("float64"),
}


def numpy_to_vtk_matrix4x4(arr: np.ndarray) -> vtkMatrix4x4:
    r"""Create vtkMatrix4x4 from NumPy array."""
    assert arr.shape == (4, 4)
    matrix = vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, arr[i, j])
    return matrix


def numpy_to_vtk_array(*args, name: Optional[str] = None, **kwargs) -> vtkDataArray:
    r"""Convert NumPy array to vtkDataArray."""
    with warnings.catch_warnings():
        # vtk/util/numpy_support.py:137: FutureWarning: Conversion of the second argument of
        # issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will
        # be treated as `np.complex128 == np.dtype(complex).type`.
        warnings.filterwarnings("ignore", category=FutureWarning)
        data_array = numpy_support.numpy_to_vtk(*args, **kwargs)
    if name is not None:
        data_array.SetName(name)
    return data_array


def numpy_to_vtk_points(arr: np.ndarray) -> vtkPoints:
    r"""Convert NumPy array to vtkPoints."""
    data = numpy_to_vtk_array(arr)
    points = vtkPoints()
    points.SetData(data)
    return points


def numpy_to_vtk_cell_array(arr: np.ndarray) -> vtkCellArray:
    r"""Convert NumPy array to vtkCellArray."""
    if arr.ndim != 2:
        raise ValueError("numpy_to_vtk_cell_array() 'arr' must be 2-dimensional")
    if arr.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("numpy_to_vtk_cell_array() 'arr' must have dtype int32 or int64")
    cells = vtkCellArray()
    use_set_data = False
    if use_set_data:
        # FIXME: causes segfault when loading
        dtype = arr.dtype
        offsets = np.repeat(np.array(arr.shape[1], dtype=dtype), arr.shape[0])
        offsets = np.concatenate([[0], np.cumsum(offsets)])
        offsets = numpy_to_vtk_array(offsets.astype(dtype))
        connectivity = numpy_to_vtk_array(arr.flatten().astype(dtype))
        if not cells.SetData(offsets, connectivity):
            raise RuntimeError(
                "numpy_to_vtk_cell_array() failed to convert NumPy array to vtkCellArray"
            )
    else:
        cells.AllocateExact(arr.shape[0], np.prod(arr.shape))
        for cell in arr:
            cells.InsertNextCell(len(cell))
            for ptId in cell:
                cells.InsertCellPoint(ptId)
    assert cells.IsValid()
    return cells


def vtk_to_numpy_array(data: vtkDataArray) -> np.ndarray:
    r"""Convert vtkDataArray to NumPy array."""
    return numpy_support.vtk_to_numpy(data)


def vtk_to_numpy_points(points: Union[vtkPoints, vtkPointSet]) -> np.ndarray:
    r"""Convert vtkPoints to NumPy array."""
    if isinstance(points, vtkPointSet):
        points = points.GetPoints()
    return vtk_to_numpy_array(points.GetData())
