r"""Auxiliary functions for working with `VTK <https://vtk.org/>`_ data structures."""

from .idlist import iter_cell_point_ids
from .idlist import iter_id_list
from .idlist import iter_point_cell_ids
from .polydataio import read_polydata
from .polydataio import write_polydata

# fmt: off
__all__ = (
    "iter_id_list",
    "iter_cell_point_ids",
    "iter_point_cell_ids",
    "read_polydata",
    "write_polydata",
)
# fmt: on
