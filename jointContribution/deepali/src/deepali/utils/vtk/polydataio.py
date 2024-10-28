r"""Auxiliary functions for working with vtkPolyData."""

from io import StringIO
from typing import List
from typing import Tuple

import numpy as np
from deepali.core.pathlib import PathUri
from deepali.core.storage import StorageObject
from vtk import vtkCellArray
from vtk import vtkOBJReader
from vtk import vtkOBJWriter
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


def read_polydata(path: PathUri) -> vtkPolyData:
    r"""Read vtkPolyData from specified file."""
    with StorageObject.from_path(path) as obj:
        if not obj.is_file():
            raise FileNotFoundError(f"File not found: {obj.uri}")
        obj = obj.pull(force=True)
        if not obj.path.is_file():
            raise FileNotFoundError(f"File not found: {obj.path}")
        suffix = obj.path.suffix.lower()
        if suffix == ".off":
            return read_polydata_off(obj.path)
        if suffix == ".obj":
            reader = vtkOBJReader()
        elif suffix == ".ply":
            reader = vtkPLYReader()
        elif suffix == ".vtp":
            reader = vtkXMLPolyDataReader()
        elif suffix == ".vtk":
            reader = vtkPolyDataReader()
        else:
            raise ValueError("Unsupported file name extension: {}".format(suffix))
        reader.SetFileName(str(obj.path))
        reader.Update()
        polydata = vtkPolyData()
        polydata.DeepCopy(reader.GetOutput())
    return polydata


def read_off(path: PathUri) -> Tuple[List[List[float]], List[List[int]]]:
    r"""Read values from .off file."""
    with StorageObject.from_path(path) as obj:
        data = obj.read_text()
    stream = StringIO(data)
    magic = stream.readline().strip()
    if magic not in ("OFF", "CNOFF"):
        raise ValueError(f"Invalid OFF file header: {path}")
    header = tuple([int(s) for s in stream.readline().strip().split(" ")])
    n_verts, n_faces = header[:2]
    verts = [[float(s) for s in stream.readline().strip().split(" ")] for _ in range(n_verts)]
    faces = [[int(s) for s in stream.readline().strip().split(" ")] for _ in range(n_faces)]
    assert len(verts) == n_verts, f"Expected {n_verts} vertices, found only {len(verts)}"
    assert len(faces) == n_faces, f"Expected {n_faces} vertices, found only {len(faces)}"
    return verts, faces


def read_polydata_off(path: PathUri) -> vtkPolyData:
    r"""Read vtkPolyData from .off file."""
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


def write_polydata(polydata: vtkPolyData, path: PathUri):
    r"""Write vtkPolyData to specified file in XML format."""
    with StorageObject.from_path(path) as obj:
        suffix = obj.path.suffix.lower()
        if suffix == ".off":
            write_polydata_off(polydata, obj.path)
            return
        if suffix == ".obj":
            writer = vtkOBJWriter()
        elif suffix == ".ply":
            writer = vtkPLYWriter()
            writer.SetFileTypeToBinary()
        elif suffix == ".vtp":
            writer = vtkXMLPolyDataWriter()
        elif suffix == ".vtk":
            writer = vtkPolyDataWriter()
        else:
            raise ValueError("Unsupported file name extension: {}".format(suffix))
        try:
            obj.path.unlink()  # in case of protected symlink to DVC cache
        except FileNotFoundError:
            obj.path.parent.mkdir(parents=True, exist_ok=True)
        writer.SetFileName(str(obj.path))
        writer.SetInputData(polydata)
        writer.Update()
        obj.push(force=True)


def write_polydata_off(polydata: vtkPolyData, path: PathUri):
    r"""Write vtkPolyData to specified file in OFF format."""
    verts = vtk_to_numpy_points(polydata)

    F = polydata.GetPolys().GetNumberOfCells()
    faces = vtk_to_numpy_array(polydata.GetPolys().GetData())
    assert faces.ndim == 1
    if len(faces) / F != 4:
        raise ValueError("write_polydata_off() only supports triangulated surface meshes")
    faces = faces.reshape(-1, 4)

    stream = StringIO()
    stream.write("OFF\n")
    stream.write(f"{len(verts)} {len(faces)} 0\n")
    np.savetxt(stream, verts, delimiter=" ", newline="\n", header="", footer="")
    np.savetxt(stream, faces, delimiter=" ", newline="\n", header="", footer="", fmt="%d")

    with StorageObject.from_path(path) as obj:
        obj.write_text(stream.getvalue())
