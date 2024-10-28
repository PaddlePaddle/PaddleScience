r"""Common auxiliary functions for working with VTK data structures."""

from typing import Generator

from vtk import vtkIdList
from vtk import vtkPolyData


def iter_id_list(ids: vtkIdList, start: int = 0, stop: int = None) -> Generator[int, None, None]:
    r"""Iterate over IDs in vtkIdList."""
    if stop is None:
        stop = ids.GetNumberOfIds()
    elif stop < 0:
        stop = stop % (ids.GetNumberOfIds() + 1)
    for idx in range(start, stop):
        yield ids.GetId(idx)


def iter_cell_point_ids(
    polydata: vtkPolyData, cell_id: int, start: int = 0, stop: int = None
) -> Generator[int, None, None]:
    r"""Iterate over IDs of points that a given vtkPolyData cell is made up of."""
    point_ids = vtkIdList()
    polydata.GetCellPoints(cell_id, point_ids)
    yield from iter_id_list(point_ids, start, stop)


def iter_point_cell_ids(
    polydata: vtkPolyData, point_id: int, start: int = 0, stop: int = None
) -> Generator[int, None, None]:
    r"""Iterate over IDs of vtkPolyData cells that contain a specified point."""
    cell_ids = vtkIdList()
    polydata.GetPointCells(point_id, cell_ids)
    yield from iter_id_list(cell_ids, start, stop)
