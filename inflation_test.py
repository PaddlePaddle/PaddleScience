import numpy as np
import pymesh
from pyevtk.hl import pointsToVTK

from paddlescience.neo_geometry import Disk, Mesh, Rectangle


def __save_vtk_raw(filename="output", cordinate=None, data=None):

    npoints = len(cordinate)
    ndims = len(cordinate[0])

    if data is None:
        data = np.ones((npoints, 1), dtype=type(cordinate[0, 0]))

    data_vtk = dict()

    for i in range(len(data[0, :])):
        data_vtk[str(i + 1)] = data[:, i].copy()

    if ndims == 3:
        axis_x = cordinate[:, 0].copy()
        axis_y = cordinate[:, 1].copy()
        axis_z = cordinate[:, 2].copy()
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)
    elif ndims == 2:
        axis_x = cordinate[:, 0].copy()
        axis_y = cordinate[:, 1].copy()
        axis_z = np.zeros(npoints).astype(axis_x.dtype)
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)


nrs = [1000, 10000, 20000]
dis = [0.0, 5, -5]
val = [1.0, 20.0, 40.0]
random = "Sobol"
box_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/box.stl")
original_boundary_points = box_mesh.random_boundary_points(10000, random=random)
# print(original_boundary_points.shape, original_boundary_points.min(), original_boundary_points.max())
inflated_boundary_points = box_mesh.inflated_random_boundary_points(nrs, dis, random=random)
# print(inflated_boundary_points.shape, inflated_boundary_points.min(), inflated_boundary_points.max())
__save_vtk_raw(f"original_boundary_points_{random}", original_boundary_points, np.full([len(original_boundary_points), 1], 10.0))
__save_vtk_raw(f"inflated_boundary_in_out_points_{random}", inflated_boundary_points, np.concatenate([np.full([nrs[i], 1], val[i]) for i in range(len(nrs))], axis=0))
