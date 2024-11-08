from io import StringIO
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from vtk import vtkCellArray
from vtk import vtkPLYReader
from vtk import vtkPLYWriter
from vtk import vtkPoints
from vtk import vtkPolyData
from vtk import vtkPolyDataReader
from vtk import vtkPolyDataWriter
from vtk import vtkXMLPolyDataReader
from vtk import vtkXMLPolyDataWriter

from .numpy import vtk_to_numpy_array
from .numpy import vtk_to_numpy_points

PathStr = Union[Path, str]


def read_polydata(path: PathStr) -> vtkPolyData:
    """Read vtkPolyData from specified file."""
    path = Path(path).absolute()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix == ".off":
        return read_polydata_off(path)
    elif suffix == ".ply":
        reader = vtkPLYReader()
    elif suffix == ".vtp":
        reader = vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtkPolyDataReader()
    else:
        raise ValueError("Unsupported file name extension: {}".format(suffix))
    reader.SetFileName(str(path))
    reader.Update()
    polydata = vtkPolyData()
    polydata.DeepCopy(reader.GetOutput())
    return polydata


def read_off(path: PathStr) -> Tuple[List[List[float]], List[List[int]]]:
    """Read values from .off file."""
    data = Path(path).read_text()
    stream = StringIO(data)
    magic = stream.readline().strip()
    if magic not in ("OFF", "CNOFF"):
        raise ValueError(f"Invalid OFF file header: {path}")
    header = tuple([int(s) for s in stream.readline().strip().split(" ")])
    n_verts, n_faces = header[:2]
    verts = [
        [float(s) for s in stream.readline().strip().split(" ")] for _ in range(n_verts)
    ]
    faces = [
        [int(s) for s in stream.readline().strip().split(" ")] for _ in range(n_faces)
    ]
    assert (
        len(verts) == n_verts
    ), f"Expected {n_verts} vertices, found only {len(verts)}"
    assert (
        len(faces) == n_faces
    ), f"Expected {n_faces} vertices, found only {len(faces)}"
    return verts, faces


def read_polydata_off(path: PathStr) -> vtkPolyData:
    """Read vtkPolyData from .off file."""
    verts, faces = read_off(path)
    points = vtkPoints()
    polys = vtkCellArray()
    for vert in verts:
        points.InsertNextPoint(vert[:3])
    for poly in faces:
        polys.InsertNextCell(poly[0], poly[1 : 1 + poly[0]])
    output = vtkPolyData()
    output.SetPoints(points)
    output.SetPolys(polys)
    return output


def write_polydata(polydata: vtkPolyData, path: PathStr):
    """Write vtkPolyData to specified file in XML format."""
    path = Path(path).absolute()
    suffix = path.suffix.lower()
    if suffix == ".off":
        write_polydata_off(polydata, path)
        return
    if suffix == ".ply":
        writer = vtkPLYWriter()
        writer.SetFileTypeToBinary()
    elif suffix == ".vtp":
        writer = vtkXMLPolyDataWriter()
    elif suffix == ".vtk":
        writer = vtkPolyDataWriter()
    else:
        raise ValueError("Unsupported file name extension: {}".format(suffix))
    try:
        path.unlink()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Update()


def write_polydata_off(polydata: vtkPolyData, path: PathStr):
    """Write vtkPolyData to specified file in OFF format."""
    path = Path(path).absolute()
    try:
        path.unlink()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
    verts = vtk_to_numpy_points(polydata)
    F = polydata.GetPolys().GetNumberOfCells()
    faces = vtk_to_numpy_array(polydata.GetPolys().GetData())
    assert faces.ndim == 1
    if len(faces) / F != 4:
        raise ValueError(
            "write_polydata_off() only supports triangulated surface meshes"
        )
    faces = faces.reshape(-1, 4)
    with path.open(mode="wt") as fp:
        fp.write("OFF\n")
        fp.write(f"{len(verts)} {len(faces)} 0\n")
        np.savetxt(fp, verts, delimiter=" ", newline="\n", header="", footer="")
        np.savetxt(
            fp, faces, delimiter=" ", newline="\n", header="", footer="", fmt="%d"
        )
