# sys.path.append("./PaddleScience/")
# sys.path.append("/home/aistudio/3rd_lib")

from pathlib import Path

import numpy as np
import paddle
import vtk
from src.data.base_datamodule import BaseDataModule
from vtk.util.numpy_support import vtk_to_numpy


def read_ply(file_path):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return reader, polydata


def read_obj(file_path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return reader, polydata


def read_vtk(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()

    # point_data_keys = [polydata.GetPointData().GetArrayName(i)for i in range(polydata.GetPointData().GetNumberOfArrays())]
    # cell_data_keys = [polydata.GetCellData().GetArrayName(i)for i in range(polydata.GetCellData().GetNumberOfArrays())]
    # print("Point Data Keys:", point_data_keys)
    # print("Cell Data Keys:", cell_data_keys)
    return reader, polydata


def normals(polydata):
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.FlipNormalsOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.Update()
    numpy_cell_normals = vtk_to_numpy(
        normals_filter.GetOutput().GetCellData().GetNormals()
    ).astype(np.float32)
    return numpy_cell_normals


def areas(polydata):
    cell_size_filter = vtk.vtkCellSizeFilter()
    cell_size_filter.SetInputData(polydata)
    cell_size_filter.ComputeAreaOn()
    cell_size_filter.Update()
    numpy_cell_areas = vtk_to_numpy(
        cell_size_filter.GetOutput().GetCellData().GetArray("Area")
    ).astype(np.float32)
    return numpy_cell_areas


def centoirds(polydata):
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(polydata)
    cell_centers.Update()
    numpy_cell_centers = vtk_to_numpy(
        cell_centers.GetOutput().GetPoints().GetData()
    ).astype(np.float32)
    return numpy_cell_centers


def nodes(polydata):
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    return points


def load_velocity(polydata):
    point_data_keys = [
        polydata.GetPointData().GetArrayName(i)
        for i in range(polydata.GetPointData().GetNumberOfArrays())
    ]
    if "point_vectors" in point_data_keys:
        vel = vtk_to_numpy(polydata.GetPointData().GetArray("point_vectors")).astype(
            np.float32
        )
        return vel
    else:
        return "no data"


def load_sdf_queries():
    tx = np.linspace(0, 1, 64)
    ty = np.linspace(0, 1, 64)
    tz = np.linspace(0, 1, 64)
    sdf_q = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
    sdf_q = np.transpose(sdf_q, (3, 0, 1, 2))
    return sdf_q


def load_sdf():
    sdf = np.ones([64, 64, 64]).astype(np.float32)
    return sdf


def read(file_path):
    if file_path.suffix == ".ply":
        _, polydata = read_ply(file_path)
        data_dict = {
            "centroids": centoirds(polydata),
            "areas": areas(polydata),
            "normal": normals(polydata),
            "sdf": load_sdf(),
            "sdf_query_points": load_sdf_queries(),
        }
    elif file_path.suffix == ".obj":
        _, polydata = read_obj(file_path)
        data_dict = {
            "centroids": centoirds(polydata),
            "vertices": nodes(polydata),
            "areas": areas(polydata),
            "normal": normals(polydata),
            "sdf": load_sdf(),
            "sdf_query_points": load_sdf_queries(),
        }
    elif file_path.suffix == ".vtk":
        _, polydata = read_vtk(file_path)
        load_sdf()
        data_dict = {
            "vertices": nodes(polydata),
            "velocity": load_velocity(polydata),
            "sdf": load_sdf(),
            "sdf_query_points": load_sdf_queries(),
        }
    else:
        raise NotImplementedError

    return data_dict


class VelocityDataset(paddle.io.Dataset):
    def __init__(self, dir, index_list):
        self.dir = dir
        self.index_list = index_list
        self.len = len(index_list)

    def __getitem__(self, index):
        index = self.index_list[index]
        index = str(index).zfill(3)
        data_dict = read(self.dir / f"vel_{index}.vtk")
        return data_dict

    def __len__(self):
        return self.len


class VelocityDataModule(BaseDataModule):
    def __init__(
        self, train_data_dir, test_data_dir, train_index_list, test_index_list
    ):
        BaseDataModule.__init__(self)
        self.train_data = VelocityDataset(Path(train_data_dir), train_index_list)
        self.test_data = VelocityDataset(Path(test_data_dir), test_index_list)
        self.train_indices = train_index_list
        self.test_indices = test_index_list

    def decode(self, x):
        return x


# incase some one like sdf
# def normalization(locations,min_bounds, max_bounds):
#     locations = (locations - min_bounds) / (max_bounds - min_bounds)
#     locations = 2 * locations - 1
#     return locations

# def load_bound(data_dir, filename):
#     with open(data_dir / filename, "r") as fp:
#         min_bounds = fp.readline().split(" ")
#         max_bounds = fp.readline().split(" ")
#         min_bounds = [float(a) - 1e-6 for a in min_bounds]
#         max_bounds = [float(a) + 1e-6 for a in max_bounds]
#     return min_bounds, max_bounds

# def location_normalization(locations, min_bounds, max_bounds):
#     min_bounds = paddle.to_tensor(min_bounds)
#     max_bounds = paddle.to_tensor(max_bounds)
#     locations = (locations - min_bounds) / (max_bounds - min_bounds)
#     locations = 2 * locations - 1
#     return locations

# def load_sdf():
#     data_dir = Path("/home/aistudio/txt")
#     min_bounds, max_bounds = load_bound(data_dir, filename="global_bounds.txt")
#     tx = np.linspace(min_bounds[0], max_bounds[0], 64)
#     ty = np.linspace(min_bounds[1], max_bounds[1], 64)
#     tz = np.linspace(min_bounds[2], max_bounds[2], 64)
#     sdf_query_points = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
#     mesh = open3d.io.read_triangle_mesh(str(file_path))
#     mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
#     scene = open3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(mesh)
#     sdf = scene.compute_distance(sdf_query_points).numpy()
#     sdf_query_points = paddle.to_tensor(sdf_query_points)
#     data_dict["sdf_query_points"] = location_normalization(sdf_query_points, min_bounds, max_bounds).unsqueeze(axis=0).transpose(perm=[0, 4, 1, 2, 3])
#     data_dict["df"] = paddle.to_tensor(sdf).unsqueeze(axis=0)
#     data_dict["info"] = [{"load_velocity": 33.33}]
#     return data_dict

if __name__ == "__main__":
    file_path = "/home/aistudio/1a0bc9ab92c915167ae33d942430658c.obj"
    _, polydata = read_obj(file_path)
    data_dict = {
        "centroids": centoirds(polydata),
        # "areas":        areas(polydata),
        # "normal":       normals(polydata),
    }
    print(centoirds(polydata))
