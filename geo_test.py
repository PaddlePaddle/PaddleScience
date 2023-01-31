import numpy as np
# import paddle
import pymesh
from pyevtk.hl import pointsToVTK
# import open3d
from paddlescience.geometry_new import Mesh, Rectangle, Disk

# import pysdf
# vertices = np.array([
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0]
# ], dtype="float32")
# faces = np.array([0, 1, 2], dtype="uint32")
# print(vertices.shape)
# print(faces.shape)
# sdf = pysdf.SDF(vertices, faces)
# dist = sdf(np.array([
#     [0.0, 0.0, 0.0]
# ]))
# print(dist)
# exit()
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



# vertices = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 1],
# ])
# faces = np.array([
#     [0, 2, 1],
#     [1, 2, 3]
# ])
# mesh = pymesh.form_mesh(vertices, faces)
# print(type())
# mesh.add_attribute("vertex_normal")
# normals = mesh.get_attribute("vertex_normal").reshape([-1, 3])
# print(normals)
# exit()

# closed_mesh = Mesh("/workspace/hesensen/PaddleScience_Aneurysm/examples/BloodFlow/aneurysm_closed.stl")
# closed_mesh_o3d = open3d.io.read_triangle_mesh("/workspace/hesensen/PaddleScience_Aneurysm/examples/BloodFlow/aneurysm_closed.stl")
# print(np.asarray(closed_mesh_o3d.triangles).shape)
# print(closed_mesh.py_mesh.faces.shape)
# exit()
# print(closed_mesh.py_mesh.vertices.shape)
# tri = closed_mesh.py_mesh.faces[302]
# p = closed_mesh.py_mesh.vertices[tri]
# p0, p1, p2 = np.split(p, 3, axis=0)
# # print(p0, p1, p2)
# p01 = (p1 - p0).reshape([-1, ])
# p12 = (p2 - p2).reshape([-1, ])
# p02 = (p2 - p0).reshape([-1, ])
# closed_mesh.py_mesh.add_attribute("vertex_normal")
# normals = closed_mesh.py_mesh.get_attribute("vertex_normal").reshape([-1, 3])
# p0_normal = normals[tri[0]]
# p1_normal = normals[tri[1]]
# p2_normal = normals[tri[2]]
# print(p0_normal.shape, p01.shape)
# print(p1_normal.shape, p12.shape)
# print(p2_normal.shape, p02.shape)
# print(p01.dot(p0_normal))
# print(p12.dot(p1_normal))
# print(p02.dot(p2_normal))
# exit()
# print(normals.shape, normals.sum(), normals.dtype)
# print((normals**2).sum())
# exit()
# box_mesh = Mesh("./box_zyb.stl")
# difference_mesh.add_sample_config("interior", batch_size=50000)
# difference_mesh.add_sample_config("boundary", batch_size=50000)
# difference_mesh.add_sample_config("bc1", batch_size=50000, criteria=lambda x, y, z: (-10 < x - 0.0) & (x - 0.0 < 10))
# points_dict = difference_mesh.fetch_batch_data()
# bc1_points = points_dict["bc1"]
# interior_points = points_dict["interior"]
# boundary_points = points_dict["boundary"]
# for k, v in points_dict.items():
#     print(f"{k} {v.shape}")
# exit()
# __save_vtk_raw("difference_interior", interior_points, np.full([len(interior_points), 1], 10.0))
# __save_vtk_raw("difference_boundary", boundary_points, np.full([len(boundary_points), 1], 100.0))
# __save_vtk_raw("difference_bc1", bc1_points, np.full([len(bc1_points), 1], 200.0))
# exit()
# inlet_mesh = Mesh("/workspace/hesensen/PaddleScience_Aneurysm/examples/BloodFlow/aneurysm_inlet.stl")
# integral_mesh = Mesh("/workspace/hesensen/PaddleScience_Aneurysm/examples/BloodFlow/aneurysm_integral.stl")
# noslip_mesh = Mesh("/workspace/hesensen/PaddleScience_Aneurysm/examples/BloodFlow/aneurysm_noslip.stl")
# outlet_mesh = Mesh("/workspace/hesensen/PaddleScience_Aneurysm/examples/BloodFlow/aneurysm_outlet.stl")

# __save_vtk_raw("closed_mesh", closed_mesh.points, np.full([len(closed_mesh.points), 1], 1.0))
# __save_vtk_raw("inlet_mesh", inlet_mesh.points, np.full([len(inlet_mesh.points), 1], 5.0))
# __save_vtk_raw("integral_mesh", integral_mesh.points, np.full([len(integral_mesh.points), 1], 10.0))
# __save_vtk_raw("noslip_mesh", noslip_mesh.points, np.full([len(noslip_mesh.points), 1], 15.0))
# __save_vtk_raw("outlet_mesh", outlet_mesh.points, np.full([len(outlet_mesh.points), 1], 20.0))

# union_mesh = (inlet_mesh | integral_mesh) | outlet_mesh


# __save_vtk_raw("union_mesh", union_mesh.points, np.full([len(union_mesh.points), 1], 25.0))

# plate_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/plate.ply")
# pymesh_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/pymesh.ply")
# ball_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/ball.stl")
# __save_vtk_raw("ball", ball_mesh.points, np.full([len(ball_mesh.points), 1], 10.0))
# ball_random_points = ball_mesh.random_boundary_points(1000)
# print(ball_random_points.shape)
# __save_vtk_raw("ball_boundary", ball_random_points, np.full([len(ball_random_points), 1], 5.0))
# exit()

# and_mesh = plate_mesh & pymesh_mesh
# __save_vtk_raw("and_mesh", and_mesh.points, np.full([len(and_mesh.points), 1], 25.0))

# x_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/x.stl")
# one = open3d.io.readst
# # print(x_mesh.num_points)
# points = x_mesh.random_boundary_points(10000)
# x_inter_points = x_mesh.random_points(10000)
# __save_vtk_raw("x_random_points", points, np.full([len(points), 1], 10.0))
# __save_vtk_raw("x_interior_random_points", x_inter_points, np.full([len(x_inter_points), 1], 10.0))
# y_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/y.stl")
# z_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/z.stl")
# # xy_mesh = pymesh.boolean(x_mesh.py_mesh, y_mesh.py_mesh, "union")
# # print(xy_mesh.num_vertices)
# # xyz_mesh = pymesh.boolean(xy_mesh, z_mesh.py_mesh, "union")
# # print(xyz_mesh.num_vertices)
# # __save_vtk_raw("xyz_mesh", xyz_mesh.vertices, np.full([len(xyz_mesh.vertices), 1], 20.0))
# # exit()
# ball_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/ball.stl")
box_mesh = Mesh("/workspace/hesensen/PaddleScience_geometry_dev/vtus/box.stl")
# left_mesh = ball_mesh & box_mesh
# right_mesh = y_mesh | z_mesh
# exit()
# up_mesh = left_mesh - right_mesh
# up_points = up_mesh.random_boundary_points(10000)
# up_inter_points = up_mesh.random_points(10000)
# up_inter_points_boundary = up_points[up_mesh.on_boundary(up_points)]
# up_inter_points_real_inter = up_inter_points[up_mesh.inside(up_inter_points)]
# print(up_inter_points.shape)
# print(up_inter_points_boundary.shape)
# print(up_inter_points_real_inter.shape)
# __save_vtk_raw("new_up_random_points", up_points, np.full([len(up_points), 1], 20.0))
# __save_vtk_raw("new_up_interior_random_points", up_inter_points, np.full([len(up_inter_points), 1], 10.0))
# __save_vtk_raw("new_up_interior_random_points_boundary", up_inter_points_boundary, np.full([len(up_inter_points_boundary), 1], 10.0))
# __save_vtk_raw("new_up_interior_random_points_real_inter", up_inter_points_real_inter, np.full([len(up_inter_points_real_inter), 1], 10.0))
# exit()
__save_vtk_raw("new_left_mesh", left_mesh.points, np.full([len(left_mesh.points), 1], 10.0))
__save_vtk_raw("new_right_mesh", right_mesh.points, np.full([len(right_mesh.points), 1], 20.0))
__save_vtk_raw("new_up_mesh", up_mesh.points, np.full([len(up_mesh.points), 1], 30.0))
__save_vtk_raw("new_x_mesh", x_mesh.points, np.full([len(x_mesh.points), 1], 10.0))
__save_vtk_raw("new_y_mesh", y_mesh.points, np.full([len(y_mesh.points), 1], 20.0))
__save_vtk_raw("new_z_mesh", z_mesh.points, np.full([len(z_mesh.points), 1], 30.0))


# rect = Rectangle([0, 0], [10, 10])
# # disk = Disk([1, 1], 3)

# # rect_points = rect.random_points(10000)
# # disk_points = disk.random_points(10000)
# # rect_diff_disk = rect - disk

# for i in range(5):
#     batch_data_dict = rect.fetch_batch_data(
#         10000,
#         {"x":0, "y":1},
#         fixed=False
#     )
#     for k, v in batch_data_dict.items():
#         print(f"{k} {v.shape}")
# print(rect_points.shape)
# print(disk_points.shape)
# rect_diff_disk_points = rect_diff_disk.random_points(10000)
# # print(type(rect_diff_disk))
# # print(rect_diff_disk_points.shape)
# # exit()
# __save_vtk_raw("rect", rect_points, np.full([len(rect_points), 1], 10.0))
# __save_vtk_raw("disk", disk_points, np.full([len(disk_points), 1], 100.0))
# __save_vtk_raw("rect_diff_disk", rect_diff_disk_points, np.full([len(rect_diff_disk_points), 1], 1000.0))
