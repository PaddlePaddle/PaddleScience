import os

import h5py
import numpy as np
import paddle
import pyvista as pv
import trimesh
import trimesh.sample as sample
import vtk


def load_mesh_ply_vtk(file_path):
    mesh = pv.read(file_path)
    points = mesh.points
    cells_vtk = list(mesh.cell)
    cells = []
    for cell_vtk in cells_vtk:
        cell = []
        for id in range(cell_vtk.GetNumberOfPoints()):
            cell.append(cell_vtk.GetPointId(id))
        cells.append(cell)
    points = np.array(points)
    cells = np.array(cells)
    return points, cells


def read_vtk(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def convert_quads_to_tris(unstructured_grid):
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()
    poly_data = geometry_filter.GetOutput()
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(poly_data)
    triangle_filter.Update()
    return triangle_filter.GetOutput()


def compute_and_add_normals(poly_data):
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(poly_data)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOff()
    normal_generator.Update()
    normals = normal_generator.GetOutput().GetPointData().GetNormals()
    return normals


def get_points(poly_data):
    points = poly_data.GetPoints()
    num_points = points.GetNumberOfPoints()
    points_array = np.zeros((num_points, 3))
    for i in range(num_points):
        points_array[i, :] = points.GetPoint(i)
    return points_array


def get_pressure_data(poly_data):
    pressure_array = poly_data.GetPointData().GetArray("point_scalars")
    num_points = poly_data.GetNumberOfPoints()
    if pressure_array is None:
        raise ValueError("Pressure data not found in the input VTK file.")
    pressure = np.zeros((num_points, 1))
    for i in range(num_points):
        pressure[i, 0] = pressure_array.GetValue(i)
    return pressure


def extract_triangle_indices(poly_data):
    poly_data.BuildLinks()
    num_cells = poly_data.GetNumberOfCells()
    triangle_indices = []
    for cell_id in range(num_cells):
        cell = poly_data.GetCell(cell_id)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            point_ids = cell.GetPointIds()
            indices = [point_ids.GetId(i) for i in range(3)]
            triangle_indices.append(indices)
    return np.array(triangle_indices)


def write_to_vtk(data: dict, write_file_path):
    grid = vtk.vtkUnstructuredGrid()
    points = data["node|pos"]
    points_vtk = vtk.vtkPoints()
    [points_vtk.InsertNextPoint(point) for point in points]
    grid.SetPoints(points_vtk)
    point_data = grid.GetPointData()
    for key in data.keys():
        if not key.startswith("node"):
            continue
        if key == "node|pos":
            continue
        array_data = data[key]
        vtk_data_array = vtk.vtkFloatArray()
        k = (
            1
            if type(array_data[0]) is np.float64 or type(array_data[0]) is np.float32
            else len(array_data[0])
        )
        vtk_data_array.SetNumberOfComponents(k)
        if k == 1:
            [vtk_data_array.InsertNextTuple([value]) for value in array_data]
        else:
            [vtk_data_array.InsertNextTuple(value) for value in array_data]
        vtk_data_array.SetName(key)
        point_data.AddArray(vtk_data_array)
    cells = data["cells_node"].reshape(-1, 3)
    cell_array = vtk.vtkCellArray()
    for cell in cells:
        triangle = vtk.vtkTriangle()
        for i, id in enumerate(cell):
            triangle.GetPointIds().SetId(i, id)
        cell_array.InsertNextCell(triangle)
    grid.SetCells(vtk.vtkTriangle().GetCellType(), cell_array)
    cell_data = grid.GetCellData()
    for key in data.keys():
        if not key.startswith("cell|"):
            continue
        if key == "cell|cells_node":
            continue
        array_data = data[key]
        vtk_data_array = vtk.vtkFloatArray()
        k = (
            1
            if type(array_data[0]) is np.float64 or type(array_data[0]) is np.float32
            else len(array_data[0])
        )
        vtk_data_array.SetNumberOfComponents(k)
        if k == 1:
            [vtk_data_array.InsertNextTuple([value]) for value in array_data]
        else:
            [vtk_data_array.InsertNextTuple(value) for value in array_data]
        vtk_data_array.SetName(key)
        cell_data.AddArray(vtk_data_array)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(write_file_path)
    writer.SetInputData(grid)
    writer.Write()
    print(f"vtu file saved:{write_file_path}")


def write_point_cloud_to_vtk(data: dict, write_file_path):
    grid = vtk.vtkUnstructuredGrid()
    points = data["node|pos"]
    points_vtk = vtk.vtkPoints()
    [points_vtk.InsertNextPoint(point) for point in points]
    grid.SetPoints(points_vtk)
    point_data = grid.GetPointData()
    for key in data.keys():
        if not key.startswith("node"):
            continue
        if key == "node|pos":
            continue
        array_data = data[key]
        vtk_data_array = vtk.vtkFloatArray()
        k = (
            1
            if type(array_data[0]) is np.float64 or type(array_data[0]) is np.float32
            else len(array_data[0])
        )
        vtk_data_array.SetNumberOfComponents(k)
        if k == 1:
            [vtk_data_array.InsertNextTuple([value]) for value in array_data]
        else:
            [vtk_data_array.InsertNextTuple(value) for value in array_data]
        vtk_data_array.SetName(key)
        point_data.AddArray(vtk_data_array)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(write_file_path)
    writer.SetInputData(grid)
    writer.Write()
    print(f"vtu file saved:[{write_file_path}]")


def compute_sdf_query_points(points, cells, query_points) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=points, faces=cells)
    sds = mesh.nearest.signed_distance(query_points)
    return sds


def compute_sdf_grid(points, cells, bounds, resolution: list, eq_res=False):
    res = resolution
    x, y, z = [
        np.linspace(bounds[0][i], bounds[1][i], res_i) for i, res_i in enumerate(res)
    ]
    xx, yy, zz = np.meshgrid(x, y, z)
    grids = np.stack([xx, yy, zz], axis=-1)
    query_points = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
    return grids, compute_sdf_query_points(points, cells, query_points)


def normalize_points(points, bounds):
    """
    Normalize points to a cube defined by [-1, 1] in each dimension.

    Parameters:
    points (numpy.ndarray): An array of shape (N, 3) representing the points.
    bounds (numpy.ndarray): An array of shape (2, 3) representing the min and max bounds for x, y, z.

    Returns:
    numpy.ndarray: The normalized points.
    """
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    center = (min_bounds + max_bounds) / 2.0
    half_range = (max_bounds - min_bounds) / 2.0
    normalized_points = (points - center) / half_range
    return normalized_points


def compute_mean_std(data):
    mean = 0.0
    std = 0.0
    n_samples = 0
    for x in data:
        x = x.reshape(-1, 1)
        n_samples += x.shape[0]
        mean += x.sum(axis=0)
    mean /= n_samples
    for x in data:
        x = x.reshape(-1, 1)
        std += ((x - mean) ** 2).sum(axis=0)
    std = paddle.sqrt(x=std / n_samples)
    return mean.to("float32"), std.to("float32")


def compute_mean_std_3dvector(data):
    normals = np.concatenate(data, axis=0)
    mean_vector = np.mean(normals, axis=0)
    variance_vector = np.var(normals, axis=0)
    return paddle.to_tensor(data=mean_vector).to("float32"), paddle.to_tensor(
        data=variance_vector
    ).to("float32")


def dict2Device(data: dict, device):
    for key, v in data.items():
        data[key] = v.to("float32").to(device)
    return data


def compute_sdf_for_h5_file(h5_file_path):
    with h5py.File(h5_file_path, "r+") as h5file:
        for key in h5file.keys():
            dataset = h5file[key]
            pos = dataset["node|pos"][:]
            cells_node = dataset["cells_node"][:].reshape(-1, 3)
            bounds = np.array([1, 1, 1])
            bounds = np.loadtxt(
                os.path.join(
                    os.path.dirname(h5_file_path), "watertight_global_bounds.txt"
                )
            )
            if "voxel|sdf" in dataset.keys():
                del dataset["voxel|sdf"]
            if "voxel|grid" in dataset.keys():
                del dataset["voxel|grid"]
            grid, sdf = compute_sdf_grid(pos, cells_node, bounds, [64, 64, 64])
            dataset.create_dataset("voxel|sdf", data=sdf)
            dataset.create_dataset("voxel|grid", data=grid)
            print(f"process {key} done")


def compute_ao(ply_file, n_samples=64):
    model = trimesh.load(ply_file, force="mesh")
    assert isinstance(model, trimesh.Trimesh)
    NDIRS = n_samples
    RELSIZE = 0.05
    sphere_pts, _ = sample.sample_surface_even(trimesh.primitives.Sphere(), count=NDIRS)
    normal_dir_similarities = model.vertex_normals @ sphere_pts.T
    assert tuple(normal_dir_similarities.shape)[0] == len(model.vertex_normals)
    assert tuple(normal_dir_similarities.shape)[1] == len(sphere_pts)
    normal_dir_similarities[normal_dir_similarities <= 0] = 0
    normal_dir_similarities[normal_dir_similarities > 0] = 1
    vert_idxs, dir_idxs = np.where(normal_dir_similarities)
    del normal_dir_similarities
    normals = model.vertex_normals[vert_idxs]
    origins = model.vertices[vert_idxs] + normals * model.scale * 0.0005
    directions = sphere_pts[dir_idxs]
    assert len(origins) == len(directions)
    hit_pts, idxs_rays, _ = model.ray.intersects_location(
        ray_origins=origins, ray_directions=directions
    )
    succ_origs = origins[idxs_rays]
    distances = np.linalg.norm(succ_origs - hit_pts, axis=1)
    idxs_rays = idxs_rays[distances < RELSIZE * model.scale]
    idxs_orig = vert_idxs[idxs_rays]
    uidxs, uidxscounts = np.unique(idxs_orig, return_counts=True)
    assert len(uidxs) == len(uidxscounts)
    counts_verts = np.zeros(len(model.vertices))
    counts_verts[uidxs] = uidxscounts
    counts_verts = counts_verts / np.max(counts_verts) * 255
    counts_verts = 255 - counts_verts.astype(int).reshape(-1, 1)
    AO = counts_verts / np.full_like(counts_verts, 255.0)
    return AO
