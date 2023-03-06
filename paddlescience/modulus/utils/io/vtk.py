"""Helper functions for generating vtk files
"""

import time
import scipy
import numpy as np
import matplotlib
import sympy as sp
import logging

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pathlib import Path
import pathlib
from typing import List, Dict, Union, Tuple

logger = logging.getLogger(__name__)


class VTKBase:
    # Only supports working with point data
    def __init__(self, file_name: str, file_dir: str):
        self.file_name = file_name
        self.file_dir = file_dir
        self.ext = ".vtk"

        self.vtk_obj = None
        self.writer = None
        self.export_map = {}

    def save_vtk(self):
        raise NotImplementedError("Implement in VTK subclass")

    def get_points(self):
        raise NotImplementedError("Implement in VTK subclass")

    def get_cells(self):
        raise NotImplementedError("Implement in VTK subclass")

    def set_points(self):
        raise NotImplementedError("Implement in VTK subclass")

    def set_cells(self):
        raise NotImplementedError("Implement in VTK subclass")

    def get_array_names(self):
        narrays = self.vtk_obj.GetPointData().GetNumberOfArrays()
        names = []
        for i in range(narrays):
            names.append(self.vtk_obj.GetPointData().GetArrayName(i))
        return names

    def get_array(self, name: str, dim: Union[None, int] = None):
        if name not in self.get_array_names():
            logger.warn(f"{name} not found in data arrays")
            return None

        data_array = vtk_to_numpy(self.vtk_obj.GetPointData().GetArray(name))

        # Expand last dim for scalars for consistency
        if data_array.ndim == 1:
            data_array = data_array[:, np.newaxis]
        elif dim is not None:
            # Get component of data array
            if dim > data_array.shape[1]:
                raise ValueError(
                    f"Dimension requested of VTK dataarray {name}:{dim} is too large. Data-array size: {data_array.shape}"
                )
            data_array = data_array[:, dim : dim + 1]
        return data_array

    def get_data_from_map(self, vtk_data_map: Dict[str, List[str]]):

        data_dict = {}
        coord_map = {"x": 0, "y": 1, "z": 2}
        # Loop through input map values
        for input_key, vtk_keys in vtk_data_map.items():
            input_array = []
            for vtk_key in vtk_keys:
                # Check if coordinate array
                if vtk_key in coord_map:
                    input_array0 = self.get_points(dims=[coord_map[vtk_key]])
                    input_array.append(input_array0)
                # Check if data array
                elif vtk_key.split(":")[0] in self.get_array_names():
                    if len(vtk_key.split(":")) > 1:
                        input_array0 = self.get_array(
                            name=vtk_key.split(":")[0], dim=int(vtk_key.split(":")[1])
                        )
                    else:
                        input_array0 = self.get_array(name=vtk_key)
                    input_array.append(input_array0)

            data_dict[input_key] = np.concatenate(input_array, axis=1)

        return data_dict

    def var_to_vtk(
        self,
        data_vars: Dict[str, np.array],
        file_name: str = None,
        file_dir: str = None,
        step: int = None,
    ):
        if file_name is None:
            file_name = self.file_name

        if file_dir is None:
            file_dir = self.file_dir

        if step is not None:
            file_name = file_name + f"{step:06}"

        # Convert any non list values in input map to lists
        for input_key, vtk_keys in self.export_map.items():
            if isinstance(vtk_keys, str):
                self.export_map[input_key] = [vtk_keys]

        # Apply vtk mask, to compose multidim variables
        out_var = {}
        for key, data_keys in self.export_map.items():
            vtk_array = []
            for data_key in data_keys:
                if data_key in data_vars:
                    if data_vars[data_key].ndim == 1:
                        vtk_array.append(data_vars[data_key][:, np.newaxis])
                    else:
                        vtk_array.append(data_vars[data_key])
                elif data_key is None:
                    vtk_array.append(
                        np.zeros((self.vtk_obj.GetNumberOfPoints(), 1), dtype=np.short)
                    )

            # If we recieved any data that fits the map
            if len(vtk_array) > 0:
                out_var[key] = np.squeeze(np.concatenate(vtk_array, axis=1))

        # Add data to vtk file
        # TODO: Only save points inside class and create vtk obj on save call
        for key, data in out_var.items():
            self.add_point_array(key, data.astype(np.float32))

        self.save_vtk(file_name, file_dir)

    def save_vtk(
        self,
        file_name: str = None,
        file_dir: str = None,
        compression: int = 1,
        data_mode: int = 1,
    ):
        # Compression level: 1 (worst compression, fastest) ... 9 (best compression, slowest).
        # https://vtk.org/doc/nightly/html/classvtkXMLWriterBase.html
        # Data mode: 0 = ascii, 1 = binary
        if file_name is None:
            file_name = self.file_name

        if file_dir is None:
            file_dir = self.file_dir

        Path(file_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(file_dir) / Path(file_name + self.ext)

        self.writer.SetFileName(file_path)
        self.writer.SetCompressorTypeToZLib()
        self.writer.SetCompressionLevel(compression)
        self.writer.SetDataMode(data_mode)
        self.writer.SetInputData(self.vtk_obj)
        self.writer.Write()

    def __concatenate_cord(cordinates):
        x = []
        for cord in cordinates:
            x.append(cord)
        points = np.concatenate(x, axis=0)

        # to pointsToVTK input format
        points_vtk = list()
        for i in range(3):
            points_vtk.append(points[:, i].copy())

        return points_vtk

    # concatenate data
    def __concatenate_data(outs, nt=None):

        vtkname = ["u1", "u2", "u3", "u4", "u5"]

        data = dict()

        # to numpy
        npouts = list()
        if nt is None:
            nouts = len(outs)
        else:
            nouts = len(outs) - 1

        for i in range(nouts):
            out = outs[i]
            if type(out) != np.ndarray:
                npouts.append(out.numpy())  # tenor to array
            else:
                npouts.append(out)

        # concatenate data
        ndata = outs[0].shape[1]
        data_vtk = list()

        n = 1 if (nt is None) else nt
        for t in range(n):
            for i in range(ndata):
                x = list()
                for out in npouts:
                    s = int(len(out) / n) * t
                    e = int(len(out) / n) * (t + 1)
                    x.append(out[s:e, i])
                data[vtkname[i]] = np.concatenate(x, axis=0)

            data_vtk.append(data)

        return data_vtk

    def save_vtk_cord(self, filename="output", time_array=None, cord=None, data=None):

        # concatenate data and cordiante 
        points_vtk = self.__concatenate_cord(cord)

        # points's shape is [ndims][npoints]
        npoints = len(points_vtk[0])
        ndims = len(points_vtk)

        # data
        if data is None:
            data_vtk = {"placeholder": np.ones(npoints, dtype=config._dtype)}
        elif type(data) == types.LambdaType:
            data_vtk = dict()
            if ndims == 3:
                data_vtk["data"] = data(points_vtk[0], points_vtk[1],
                                        points_vtk[2])
            elif ndims == 2:
                data_vtk["data"] = data(points_vtk[0], points_vtk[1])
        else:
            data_vtk = __concatenate_data(data, nt)

        n = 1 if (nt is None) else nt
        if ndims == 3:
            axis_x = points_vtk[0]
            axis_y = points_vtk[1]
            axis_z = points_vtk[2]
            for t in range(n):
                fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
                pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])
        elif ndims == 2:
            axis_x = points_vtk[0]
            axis_y = points_vtk[1]
            axis_z = np.zeros(npoints, dtype=config._dtype)
            for t in range(n):
                fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
                pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])


    def add_point_array(self, name: str, data: np.array):
        """Adds point array data into VTK file

        Parameters
        ----------
        name : str
            data array name
        data : np.array
            1D or 2D numpy data array
        """
        assert (
            data.shape[0] == self.vtk_obj.GetNumberOfPoints()
        ), f"Input array incorrect size. Got {data.shape[0]} instead of {self.vtk_obj.GetNumberOfPoints()}"
        assert data.ndim < 3, "1D and 2D arrays supported"

        data_array = numpy_to_vtk(data, deep=True)
        if data.ndim == 2:
            data_array.SetNumberOfComponents(data.shape[1])
        data_array.SetName(name)
        self.vtk_obj.GetPointData().AddArray(data_array)

    def remove_point_array(self, name: str):
        if name in self.get_array_names():
            self.vtk_obj.GetPointData().RemoveArray(name)
        else:
            logger.warn(f"Point data {name} not present in VTK object")





class VTKUniformGrid(VTKBase):
    """vtkUniformGrid wrapper class

    Parameters
    ----------
    bounds : List[List[int]]
        Domain bounds of each dimension
    npoints : List[int]
        List of number of points in each dimension
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1 to 1 mapping if none is provided, by default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    init_vtk : bool, optional
        Initialize new VTK object from parameters (used by VTKFromFile), by default True
    """

    def __init__(
        self,
        bounds: List[List[int]],
        npoints: List[int],
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
        init_vtk: bool = True,
    ):
        super().__init__(file_name, file_dir)

        self.vtk_obj = vtk.vtkUniformGrid()
        self.writer = vtk.vtkXMLImageDataWriter()
        self.ext = ".vti"
        self.export_map = export_map

        if init_vtk:
            self.init_points(bounds, npoints)

    def init_points(
        self,
        bounds: List[List[int]],
        npoints: List[int],
    ):
        assert len(bounds) == len(
            npoints
        ), f"Bounds and npoints must be same length {len(bounds)}, {len(npoints)}"
        assert (
            len(bounds) > 0 and len(bounds) < 4
        ), "Only 1, 2, 3 grid dimensionality allowed"
        # Padd for missing dimensions
        npoints = np.array(npoints + [1, 1])
        bounds = np.array(bounds + [[0, 0], [0, 0]])
        dx = abs(bounds[:, 0] - bounds[:, 1]) / np.maximum(
            np.ones_like(npoints), npoints - 1
        )

        # This is unique to uniform grid since it uses the imgdata backend
        self.vtk_obj.SetOrigin(
            bounds[0][0], bounds[1][0], bounds[2][0]
        )  # default values
        self.vtk_obj.SetSpacing(dx[0], dx[1], dx[2])
        self.vtk_obj.SetDimensions(npoints[0], npoints[1], npoints[2])

    def get_points(self, dims: List[int] = [0, 1, 2]):

        # Slow but VTK Image data does not explicitly store point coords
        points = []
        for i in range(self.vtk_obj.GetNumberOfPoints()):
            points.append(self.vtk_obj.GetPoint(i))
        points = np.array(points)
        return np.concatenate([points[:, i : i + 1] for i in dims], axis=1)

    def set_points(self, points: np.array):
        raise NotImplementedError("Cannot set points on vtkUniformGrid")

    def set_cells(self):
        raise AttributeError("Cannot set the cells of a vtkStructuredPoints")

    @classmethod
    def init_from_obj(
        cls,
        vtk_obj,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
    ):
        vtk_wrapper = VTKUniformGrid(
            None,
            None,
            export_map=export_map,
            file_name=file_name,
            file_dir=file_dir,
            init_vtk=False,
        )
        vtk_wrapper.vtk_obj = vtk_obj
        return vtk_wrapper


class VTKRectilinearGrid(VTKBase):
    """vtkRectilinearGrid wrapper class

    Parameters
    ----------
    axis_coords : List[np.array]
        List of arrays that define points on each axis
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1 to 1 mapping if none is provided, by default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    init_vtk : bool, optional
        Initialize new VTK object from parameters (used by VTKFromFile), by default True
    """

    def __init__(
        self,
        axis_coords: List[np.array],
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
        init_vtk: bool = True,
    ):
        super().__init__(file_name, file_dir)

        self.vtk_obj = vtk.vtkRectilinearGrid()
        self.writer = vtk.vtkXMLRectilinearGridWriter()
        self.ext = ".vtr"
        self.export_map = export_map
        if init_vtk:
            self.init_points(axis_coords)

    def init_points(self, coords: List[np.array]):
        assert len(coords) < 4, "Maximum  of 3 spacial coordinate arrays accepted"
        # Padd for missing dimensions
        coords = coords + [np.array([0]), np.array([0])]

        # This is unique to vtkRectilinearGrid since points are not explicit
        self.vtk_obj.SetDimensions(
            coords[0].shape[0], coords[1].shape[0], coords[2].shape[0]
        )
        self.vtk_obj.SetXCoordinates(numpy_to_vtk(coords[0]))
        self.vtk_obj.SetYCoordinates(numpy_to_vtk(coords[1]))
        self.vtk_obj.SetZCoordinates(numpy_to_vtk(coords[2]))

    def get_points(self, dims: List[int] = [0, 1, 2]):
        # GetPoint in vtkRectilinearGrid takes in point container to populate since
        # it does not have one internally
        # https://vtk.org/doc/nightly/html/classvtkRectilinearGrid.html
        points = vtk.vtkPoints()
        self.vtk_obj.GetPoints(points)
        # Now we can convert to numpy
        points = vtk_to_numpy(points.GetData())
        return np.concatenate([points[:, i : i + 1] for i in dims], axis=1)

    def set_points(self, points: np.array):
        raise AttributeError("Cannot set the points of a vtkRectilinearGrid explicitly")

    def set_cells(self):
        raise AttributeError("Cannot set the cells of a vtkRectilinearGrid explicitly")

    @classmethod
    def init_from_obj(
        cls,
        vtk_obj,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
    ):
        vtk_wrapper = VTKRectilinearGrid(
            None,
            export_map=export_map,
            file_name=file_name,
            file_dir=file_dir,
            init_vtk=False,
        )
        vtk_wrapper.vtk_obj = vtk_obj
        return vtk_wrapper


class VTKStructuredGrid(VTKBase):
    """vtkStructuredGrid wrapper class

    Parameters
    ----------
    points : np.array
        Mesh grid of points in 'ij' format
    dims : List[int]
        Number of points in each dimension
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1 to 1 mapping if none is provided, by default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    init_vtk : bool, optional
        Initialize new VTK object from parameters (used by VTKFromFile), by default True
    """

    def __init__(
        self,
        points: np.array,
        dims: List[int],
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
        init_vtk: bool = True,
    ):
        super().__init__(file_name, file_dir)

        self.vtk_obj = vtk.vtkStructuredGrid()
        self.writer = vtk.vtkXMLStructuredGridWriter()
        self.ext = ".vts"
        self.export_map = export_map

        if init_vtk:
            self.init_points(points, dims)

    def init_points(self, points: np.array, dims: List[int]):
        assert points.ndim == 2, "Points array must have 2 dimensions [npoints, dim]"
        assert points.shape[1] < 4, "Maximum  of 3 spacial point arrays accepted"
        assert len(dims) == points.shape[1], "Domain dimension must match dim of points"
        # Padd for missing dimensions
        points = np.concatenate(
            [points, np.zeros((points.shape[0], 2), dtype=np.short)], axis=1
        )
        dims = dims + [1, 1]
        assert (
            dims[0] * dims[1] * dims[2] == points.shape[0]
        ), "Number of points do not match provided dimensions"

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(points.shape[0])
        pts.SetData(numpy_to_vtk(points[:, :3]))

        self.vtk_obj.SetDimensions(dims[:3])
        self.vtk_obj.SetPoints(pts)

    def get_points(self, dims: List[int] = [0, 1, 2]):
        points = vtk_to_numpy(self.vtk_obj.GetPoints().GetData())
        return np.concatenate([points[:, i : i + 1] for i in dims], axis=1)

    def set_points(self, points: np.array, dims: List[int]):
        points = np.concatenate(
            [points, np.zeros((points.shape[0], 2), dtype=np.short)], axis=1
        )
        dims = dims + [1, 1]
        assert (
            dims[0] * dims[1] * dims[2] == points.shape[0]
        ), "Number of points do not match provided dimensions"

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(points.shape[0])
        pts.SetData(numpy_to_vtk(points[:, :3]))

        self.vtk_obj.SetDimensions(dims[:3])
        self.vtk_obj.SetPoints(pts)

    def set_cells(self):
        raise AttributeError("Cannot set the cells of a vtkStructuredGrid explicitly")

    @classmethod
    def init_from_obj(
        cls,
        vtk_obj,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
    ):
        vtk_wrapper = VTKStructuredGrid(
            None,
            None,
            export_map=export_map,
            file_name=file_name,
            file_dir=file_dir,
            init_vtk=False,
        )
        vtk_wrapper.vtk_obj = vtk_obj
        return vtk_wrapper


# ===================
# VTK Unstructured Grid
# ===================
class VTKUnstructuredGrid(VTKBase):
    """vtkUnstructuredGrid wrapper class

    Parameters
    ----------
    points : np.array
         Array of point locations [npoints, (1,2 or 3)]
    cell_index : Tuple[ np.array, np.array ]
        Tuple of (cell_offsets, cell_connectivity) arrays.
        Cell offsets is a 1D array denoting how many points make up a face for each cell.
        Cell connectivity is a 1D array that contains verticies of each cell face in order
    cell_types : np.array
        Array of cell vtk types:
        https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1 to 1 mapping if none is provided, by default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    init_vtk : bool, optional
        Initialize new VTK object from parameters (used by VTKFromFile), by default True
    """

    def __init__(
        self,
        points: np.array,
        cell_index: Tuple[np.array, np.array],
        cell_types: np.array,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
        init_vtk: bool = True,
    ):
        super().__init__(file_name, file_dir)

        self.vtk_obj = vtk.vtkUnstructuredGrid()
        self.writer = vtk.vtkXMLUnstructuredGridWriter()
        self.ext = ".vtu"
        self.export_map = export_map

        if init_vtk:
            self.init_points(points, cell_index, cell_types)

    def init_points(
        self,
        points: np.array,
        cell_index: Tuple[np.array, np.array],
        cell_types: np.array,
    ):
        assert points.ndim == 2, "Points array must have 2 dimensions [npoints, dim]"
        assert points.shape[1] < 4, "Maximum  of 3 spacial point arrays accepted"
        assert (
            len(cell_index) == 2
        ), "Cell index must be tuple of numpy arrays containing [offsets, connectivity]"
        # Could check cell type and cell index are consistent, but we assume the user
        # knows what they are doing

        # Padd for missing dimensions
        points = np.concatenate(
            [points, np.zeros((points.shape[0], 2), dtype=np.short)], axis=1
        )

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(points.shape[0])
        pts.SetData(numpy_to_vtk(points[:, :3]))
        self.vtk_obj.SetPoints(pts)

        vtk_celltypes = vtk.vtkIntArray()
        vtk_celltypes.SetNumberOfComponents(1)
        vtk_celltypes = numpy_to_vtk(
            cell_types.astype(int), array_type=vtk.vtkUnsignedCharArray().GetDataType()
        )

        vtk_cells = vtk.vtkCellArray()
        vtk_offsets = numpy_to_vtk(
            cell_index[0], array_type=vtk.vtkTypeInt64Array().GetDataType()
        )
        vtk_connectivity = numpy_to_vtk(
            cell_index[1], array_type=vtk.vtkTypeInt64Array().GetDataType()
        )
        vtk_cells.SetData(vtk_offsets, vtk_connectivity)

        self.vtk_obj.SetCells(vtk_celltypes, vtk_cells)

    def get_points(self, dims: List[int] = [0, 1, 2]):
        points = vtk_to_numpy(self.vtk_obj.GetPoints().GetData())
        return np.concatenate([points[:, i : i + 1] for i in dims], axis=1)
        # points = vtk_to_numpy(self.vtk_obj.GetPoints().GetData())
        # points = [points[:, 0:1], points[:, 1:2], points[:, 2:3]]
        # return [points[i] for i in dims]

    def get_cells(self):
        cells = self.vtk_obj.GetCells()
        # Get cells data contains array [nedges, v1, v2, v3, ..., nedges, v1, v2, v3,...]
        # Need to seperate offset and connectivity array for practical use
        cell_connectivity = vtk_to_numpy(cells.GetConnectivityArray())
        cell_offsets = vtk_to_numpy(cells.GetOffsetsArray())
        return cell_offsets, cell_connectivity

    def get_celltypes(self):
        cell_types = vtk_to_numpy(self.vtk_obj.GetCellTypesArray())
        return cell_types

    def set_points(self, points: np.array):
        assert points.ndim == 2, "Points array must have 2 dimensions [npoints, dim]"
        assert points.shape[1] < 4, "Maximum  of 3 spacial point arrays accepted"

        points = np.concatenate(
            [points, np.zeros((points.shape[0], 2), dtype=np.short)], axis=1
        )

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(points.shape[0])
        pts.SetData(numpy_to_vtk(points[:, :3]))
        self.vtk_obj.SetPoints(pts)

    def set_cells(self, cell_index: Tuple[np.array, np.array], cell_types: np.array):
        assert (
            len(cell_index) == 2
        ), "Cell index must be tuple of numpy arrays containing [offsets, connectivity]"

        vtk_celltypes = vtk.vtkIntArray()
        vtk_celltypes.SetNumberOfComponents(1)
        vtk_celltypes = numpy_to_vtk(
            cell_types.astype(int), array_type=vtk.vtkUnsignedCharArray().GetDataType()
        )

        vtk_cells = vtk.vtkCellArray()
        vtk_offsets = numpy_to_vtk(
            cell_index[0], array_type=vtk.vtkTypeInt64Array().GetDataType()
        )
        vtk_connectivity = numpy_to_vtk(
            cell_index[1], array_type=vtk.vtkTypeInt64Array().GetDataType()
        )
        vtk_cells.SetData(vtk_offsets, vtk_connectivity)

        self.vtk_obj.SetCells(vtk_celltypes, vtk_cells)

    @classmethod
    def init_from_obj(
        cls,
        vtk_obj,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
    ):
        vtk_wrapper = VTKUnstructuredGrid(
            None,
            None,
            None,
            export_map=export_map,
            file_name=file_name,
            file_dir=file_dir,
            init_vtk=False,
        )
        vtk_wrapper.vtk_obj = vtk_obj
        return vtk_wrapper


# ===================
# VTK Polydata
# ===================
class VTKPolyData(VTKBase):
    """vtkPolyData wrapper class

    Parameters
    ----------
    points : np.array
        Array of point locations [npoints, (1,2 or 3)]
    line_index : np.array, optional
        Array of line connections [nedges, 2], by default None
    poly_index : Tuple[poly_offsets, poly_connectivity]
        Tuple of polygon offsets and polygon connectivity arrays.
        Polygon offsets is a 1D array denoting how many points make up a face for each polygon.
        Polygon connectivity is a 1D array that contains verticies of each polygon face in order, by default None
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1 to 1 mapping if none is provided, by default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    init_vtk : bool, optional
        Initialize new VTK object from parameters (used by VTKFromFile), by default True
    """

    def __init__(
        self,
        points: np.array,
        line_index: np.array = None,
        poly_index: Tuple[
            np.array, np.array
        ] = None,  # Tuple[poly_offsets, poly_connectivity]
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
        init_vtk: bool = True,
    ):
        super().__init__(file_name, file_dir)

        self.vtk_obj = vtk.vtkPolyData()
        self.writer = vtk.vtkXMLPolyDataWriter()
        self.ext = ".vtp"
        self.export_map = export_map

        if init_vtk:
            self.init_points(points, line_index, poly_index)

    def init_points(
        self,
        points: np.array,
        line_index: np.array = None,
        poly_index: Tuple[np.array, np.array] = None,
    ):
        assert points.ndim == 2, "Points array must have 2 dimensions [npoints, dim]"
        assert points.shape[1] < 4, "Maximum  of 3 spacial point arrays accepted"
        # Padd for missing dimensions
        points = np.concatenate(
            [points, np.zeros((points.shape[0], 2), dtype=np.short)], axis=1
        )

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(int(points.shape[0]))
        pts.SetData(numpy_to_vtk(points[:, :3]))
        self.vtk_obj.SetPoints(pts)

        # Add cell array for verts
        vert_cells = vtk.vtkCellArray()
        for i in range(points.shape[0]):
            vert_cells.InsertNextCell(1)
            vert_cells.InsertCellPoint(i)
        self.vtk_obj.SetVerts(vert_cells)

        if line_index is not None:
            self.set_lines(line_index)
        if poly_index is not None:
            self.set_polys(poly_index)

    def get_points(self, dims: List[int] = [0, 1, 2]):
        points = vtk_to_numpy(self.vtk_obj.GetPoints().GetData())
        return np.concatenate([points[:, i : i + 1] for i in dims], axis=1)

    def get_lines(self):
        lines = vtk_to_numpy(self.vtk_obj.GetLines().GetData())
        line_index = np.stack([lines[1::3], lines[2::3]], axis=1)
        return line_index

    def get_polys(self):
        polys = self.vtk_obj.GetPolys()
        # Poly data contains array [nedges, v1, v2, v3, ..., nedges, v1, v2, v3,...]
        # Need to seperate offset and connectivity array for practical use
        poly_connectivity = vtk_to_numpy(polys.GetConnectivityArray())
        poly_offsets = vtk_to_numpy(polys.GetOffsetsArray())
        return poly_offsets, poly_connectivity

    def get_cells(self):
        raise AttributeError("vtkPolyData has polys not cells, call get_polys instead")

    def set_points(self, points: np.array):
        assert points.ndim == 2, "Points array must have 2 dimensions [npoints, dim]"
        assert points.shape[1] < 4, "Maximum  of 3 spacial point arrays accepted"

        points = np.concatenate(
            [points, np.zeros((points.shape[0], 2), dtype=np.short)], axis=1
        )

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(points.shape[0])
        pts.SetData(numpy_to_vtk(points[:, :3]))
        self.vtk_obj.SetPoints(pts)

        # Add cell array for verts
        vert_cells = vtk.vtkCellArray()
        for i in range(points.shape[0]):
            vert_cells.InsertNextCell(1)
            vert_cells.InsertCellPoint(i)
        self.vtk_obj.SetVerts(vert_cells)

    def set_lines(self, edge_index: np.array):
        assert (
            edge_index.ndim == 2 and edge_index.shape[1] == 2
        ), "Edge index array must have 2 dimensions [npoints, 2]"

        lines = vtk.vtkCellArray()
        for i in range(edge_index.shape[0]):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(edge_index[i, 0])
            lines.InsertCellPoint(edge_index[i, 1])

        self.vtk_obj.SetLines(lines)

    def set_polys(self, poly_index: Tuple[np.array, np.array]):
        assert (
            len(poly_index) == 2
        ), "poly_index should be tuple of (poly_offsets, poly_connectivity)"

        vtk_polys = vtk.vtkCellArray()
        vtk_offsets = numpy_to_vtk(
            poly_index[0], array_type=vtk.vtkTypeInt64Array().GetDataType()
        )
        vtk_connectivity = numpy_to_vtk(
            poly_index[1], array_type=vtk.vtkTypeInt64Array().GetDataType()
        )
        vtk_polys.SetData(vtk_offsets, vtk_connectivity)

        self.vtk_obj.SetPolys(vtk_polys)

    def set_cells(self):
        raise AttributeError("vtkPolyData has polys not cells, call set_polys instead")

    @classmethod
    def init_from_obj(
        cls,
        vtk_obj,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
    ):
        vtk_wrapper = VTKPolyData(
            None,
            export_map=export_map,
            file_name=file_name,
            file_dir=file_dir,
            init_vtk=False,
        )
        vtk_wrapper.vtk_obj = vtk_obj
        return vtk_wrapper


class VTKFromFile(object):
    """Reads VTK file into memory and constructs corresponding VTK object

    Parameters
    ----------
    file_path : str
        File directory/name of input vtk file
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1 to 1 mapping if none is provided, by default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    force_legacy : bool, optional
        Force a legacy only read, by default False
    """

    def __new__(
        cls,
        file_path: str,
        export_map: Dict[str, List[str]] = {},
        file_name: str = "vtk_output",
        file_dir: str = ".",
        force_legacy: bool = False,
    ) -> None:
        assert Path(file_path).is_file(), f"Provided VTK file {file_path} not found"

        read_success = False
        # Attempt to create XML reader
        if not force_legacy:
            try:
                vtk_reader = cls.readXMLVTK(file_path)
                read_success = True
            except:
                logger.warn("VTK file not valid XML format, will attempt legacy load")
        # If failed or legacy force, create VTK Reader
        if not read_success:
            try:
                vtk_reader = cls.readLegacyVTK(file_path)
                read_success = True
            except:
                logger.warn("VTK file not valid VTK format")
        # Hopefully VTK reader is loaded
        assert read_success, "Failed to load VTK file in either XML or Legacy format"
        logger.info(f"Read {Path(file_path).name} file successfully")

        return cls.extractVTKObject(
            vtk_reader=vtk_reader,
            export_map=export_map,
            file_name=file_name,
            file_dir=file_dir,
        )

    @classmethod
    def extractVTKObject(cls, vtk_reader, **kwargs) -> VTKBase:
        # Get vtk object from reader
        vtk_obj = vtk_reader.GetOutput()
        # Create modulus VTK wrapper
        if vtk_obj.__vtkname__ == "vtkImageData":
            vtk_wrapper = VTKUniformGrid.init_from_obj(vtk_obj, **kwargs)
        elif vtk_obj.__vtkname__ == "vtkRectilinearGrid":
            vtk_wrapper = VTKRectilinearGrid.init_from_obj(vtk_obj, **kwargs)
        elif vtk_obj.__vtkname__ == "vtkStructuredGrid":
            vtk_wrapper = VTKStructuredGrid.init_from_obj(vtk_obj, **kwargs)
        elif vtk_obj.__vtkname__ == "vtkUnstructuredGrid":
            vtk_wrapper = VTKUnstructuredGrid.init_from_obj(vtk_obj, **kwargs)
        elif vtk_obj.__vtkname__ == "vtkPolyData":
            vtk_wrapper = VTKPolyData.init_from_obj(vtk_obj, **kwargs)
        else:
            raise ValueError("Unsupported vtk data type read")

        logger.info(f"Loaded {vtk_obj.__vtkname__} object from file")
        return vtk_wrapper

    @classmethod
    def readXMLVTK(cls, file_path: str):
        # vtk.vtkXMLGenericDataObjectReader does not seem to work
        # Could read first like of XML and check VTKFile type=...
        file_path = Path(file_path)
        if file_path.suffix == ".vti":
            vtk_reader = vtk.vtkXMLImageDataReader()
        elif file_path.suffix == ".vtr":
            vtk_reader = vtk.vtkXMLRectilinearGridReader()
        elif file_path.suffix == ".vts":
            vtk_reader = vtk.vtkXMLStructuredGridReader()
        elif file_path.suffix == ".vtu":
            vtk_reader = vtk.vtkXMLUnstructuredGridReader()
        elif file_path.suffix == ".vtp":
            vtk_reader = vtk.vtkXMLPolyDataReader()
        else:
            raise ValueError("Unsupported XML VTK format")

        vtk_reader.SetFileName(file_path)
        vtk_reader.Update()

        return vtk_reader

    @classmethod
    def readLegacyVTK(cls, file_path: str):
        vtk_reader = vtk.vtkGenericDataObjectReader()
        vtk_reader.SetFileName(file_path)
        vtk_reader.ReadAllScalarsOn()
        vtk_reader.ReadAllVectorsOn()
        vtk_reader.Update()

        return vtk_reader


def var_to_polyvtk(
    var_dict: Dict[str, np.array], file_path: str, coordinates=["x", "y", "z"]
):
    """Helper method for nodes to export thier variables to a vtkPolyData file
    Should be avoided when possible as other VTK formats can save on memory.

    Parameters
    ----------
    var_dict : Dict[str, np.array]
        Dictionary of variables in the array format [nstates, dim]
    file_path : str
        File directory/name of output vtk file
    coordinates : list, optional
        Variable names that corresponds to point positions, by default ["x", "y", "z"]
    """
    # Extract point locations
    points = []
    for axis in coordinates:
        if axis not in var_dict.keys():
            data0 = next(iter(var_dict.values()))
            points.append(np.zeros((data0.shape[0], 1), dtype=np.short))
        else:
            points.append(var_dict[axis])
            del var_dict[axis]
    points = np.concatenate(points, axis=1)
    # Create 1:1 export map
    export_map = {}
    for key in var_dict.keys():
        export_map[key] = [key]

    file_path = Path(file_path)
    vtk_obj = VTKPolyData(
        points=points,
        export_map=export_map,
        file_name=file_path.stem,
        file_dir=file_path.parents[0],
    )

    vtk_obj.var_to_vtk(data_vars=var_dict)


def grid_to_vtk(var_dict: Dict[str, np.array], file_path: str, batch_index: int = 0):
    """Helper method for nodes to export image/grid data to vtkUniformData file.
    Arrays should be in the numpy 'ij' layout (element [0,0] is origin)

    Parameters
    ----------
    var_dict : Dict[str, np.array]
        Dictionary of variables in the array format [batch, dim, xdim, ydim, zdim]
    file_path : str
        File directory/name of output vtk file
    batch_index : int, optional
        Batch index to write to file, by default 0
    """
    # convert keys to strings
    var = {str(key): value for key, value in var_dict.items()}
    shape = np.shape(next(iter(var.values())))
    assert len(shape) > 2 and len(shape) < 6, "Input variables must be dim 3, 4, 5"

    # Padd for any missing dims
    bsize = shape[0]
    cdim = shape[1]
    grid_shape = list(shape[2:])
    bounds = [[0, i - 1] for i in grid_shape]

    # Flatten data and select batch
    shaped_dict = {}
    for key in var_dict.keys():
        shaped_dict[key] = var_dict[key][batch_index]
        cdim = shaped_dict[key].shape[0]
        shaped_dict[key] = shaped_dict[key].reshape(cdim, -1).T

    # Create 1:1 export map
    export_map = {}
    for key in shaped_dict.keys():
        export_map[key] = [key]

    file_path = Path(file_path)
    vtk_obj = VTKUniformGrid(
        bounds=bounds,
        npoints=grid_shape,
        export_map=export_map,
        file_name=file_path.stem,
        file_dir=file_path.parents[0],
    )

    vtk_obj.var_to_vtk(data_vars=shaped_dict)
