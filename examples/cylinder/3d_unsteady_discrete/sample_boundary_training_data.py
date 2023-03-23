# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import numpy as np
# import paddlescience as psci
# from paddlescience.modulus.geometry.primitives_3d import Box, Sphere, Cylinder
# from paddlescience.modulus.utils.io.vtk import var_to_polyvtk

# domain_coordinate_interval_dict = {1:[0,800], 2:[0,400], 3:[0,300]}
# def normalize(max_domain, min_domain, array, index):
#     #array_min = min(array[:,index])
#     #array_max = max(array[:,index])
#     diff = max_domain - min_domain
#     if abs(diff) < 0.00001:
#         array[:,index] = 0.0
#     else:
#         array[:,index] = (array[:, index] - min_domain)/diff

# def sample_data(t_step=50, nr_points = 4000):
#     # make standard constructive solid geometry example
#     # make primitives
#     # box = Box(point_1=(-20, -20, -15), point_2=(60, 20, 15))
#     # box2 = Box(point_1=(-10, -10, -15), point_2=(40, 10, 15))

#     box = Box(point_1=(0, 0, 0), point_2=(800, 400, 300))
#     box2 = Box(point_1=(120, 120, 0), point_2=(400, 280, 300))
#     # sphere = Sphere(center=(0, 0, 0), radius=1.2)
#     cylinder_1 = Cylinder(center=(200, 200, 150), radius=40, height=300)
#     # cylinder_1 = Cylinder(center=(0, 0, 0), radius=4, height=30)
#     # cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 1.0), axis="x")
#     # cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 1.0), axis="y")

#     # combine with boolean operations
#     # all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
#     # box_minus_sphere = box & sphere
#     # geo = box_minus_sphere - all_cylinders
#     # all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
#     # geo = box + box2 - cylinder_1
#     geo = box - cylinder_1
#     geo1 = box2 - cylinder_1

#     print("Sampling Boundary data ......")
#     # sample geometry for plotting in Paraview
#     # 5: Inlet, 4:Outlet, 12:cylinder
#     boundaries, s  = geo.sample_boundary(nr_points=nr_points, curve_index_filters=[4, 5, 6])
#     var_to_polyvtk(s, "boundary")
#     print("Surface Area: {:.3f}".format(np.sum(s["area"])))

#     # inlet = boundaries[0]
#     inlet = boundaries[1]   # inlet不是boundaries[0],应该是boundaries[1]
#     inlet = convert_float64_to_float32(inlet)
#     inlet_xyz = np.concatenate((inlet['x'], inlet['y'], inlet['z']), axis=1)
#     inlet_txyz = replicate_t(t_step, inlet_xyz)

#     # outlet = boundaries[1]
#     outlet = boundaries[0]  # outlet不是boundaries[1],应该是boundaries[0]
#     outlet = convert_float64_to_float32(outlet)
#     outlet_xyz = np.concatenate((outlet['x'], outlet['y'], outlet['z']), axis=1)
#     outlet_txyz = replicate_t(t_step, outlet_xyz)

#     cylinder = boundaries[2]
#     cylinder = convert_float64_to_float32(cylinder)
#     cylinder_xyz = np.concatenate((cylinder['x'], cylinder['y'], cylinder['z']), axis=1)
#     cylinder_txyz = replicate_t(t_step, cylinder_xyz)

#     print("Sampling Domain data ......")
#     interior = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
#     interior = convert_float64_to_float32(interior)
#     interior_xyz = np.concatenate((interior['x'], interior['y'], interior['z']), axis=1)
#     interior1 = geo1.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
#     interior1 = convert_float64_to_float32(interior1)
#     interior1_xyz = np.concatenate((interior1['x'], interior1['y'], interior1['z']), axis=1)
#     interior2_xyz = np.concatenate((interior_xyz, interior1_xyz), axis=0)
#     # interior_txyz = replicate_t(t_step, interior_xyz)
#     interior2_txyz = replicate_t(t_step, interior2_xyz)
#     var_to_polyvtk(interior, "interior")
#     print("Volume: {:.3f}".format(np.sum(s["area"])))

#     for item in [inlet_txyz, outlet_txyz, cylinder_txyz, interior2_txyz]:
#     # Normalize x,y,z to [0,1]
#         for coordinate, interval in domain_coordinate_interval_dict.items():
#             min_domain = interval[0]
#             max_domain = interval[1]
#             normalize(min_domain, max_domain, item, coordinate)

#     return inlet_txyz, outlet_txyz, cylinder_txyz, interior2_txyz

# def convert_float64_to_float32(dict_a):
#     for k in dict_a:
#         dict_a[k]=dict_a[k].astype(np.float32)
#     return dict_a

# def replicate_t(t_step, data):
#     full_data = None
#     for time in range(1, (t_step+1)):
#         t_len = data.shape[0]
#         t_extended = np.array([time] * t_len, dtype=np.float32).reshape((-1, 1))
#         t_data = np.concatenate((t_extended, data), axis=1)
#         if full_data is None:
#             full_data = t_data
#         else:
#             full_data = np.concatenate((full_data, t_data))

#     return full_data
"""
Created in Oct. 2022
@author: Hui Xiang, Yanbo Zhang
"""
import numpy as np
import paddlescience as psci
from paddlescience.modulus.geometry.primitives_3d import Box, Cylinder, Sphere
from paddlescience.modulus.utils.io.vtk import var_to_polyvtk
from paddlescience.visu.visu_vtk import __save_vtk_raw

domain_coordinate_interval_dict = {1: [0, 1600], 2: [0, 800], 3: [0, 320]}


def normalize(max_domain, min_domain, array, index):
    diff = max_domain - min_domain
    if abs(diff) < 0.0000001:
        array[:, index] = 0.0
    else:
        array[:, index] = (array[:, index] - min_domain) / diff


def sample_data(t_num, t_index, t_step=50, nr_points=4000):
    # make standard constructive solid geometry example
    # make primitives
    box = Box(point_1=(10, 10, 60), point_2=(1590, 798, 250))
    # box = Box(point_1=(0, 0, 0), point_2=(1200, 400, 300))

    box2 = Box(point_1=(200, 100, 60), point_2=(1590, 700, 250))
    cylinder_1 = Cylinder(center=(400, 400, 160), radius=40, height=190)

    # combine with boolean operations
    geo = box - cylinder_1
    geo1 = box2 - cylinder_1

    print("Sampling Boundary data ......")
    # sample geometry for plotting in Paraview
    # 5: Inlet, 4:Outlet, 6:cylinder, 0: Front, 1:Back, 2:Top, 3:Bottom
    # boundaries, s  = geo.sample_boundary(nr_points=nr_points, curve_index_filters=[4, 5, 6])
    boundaries, s = geo.sample_boundary(nr_points=nr_points)

    var_to_polyvtk(s, "boundary")
    print("Surface Area: {:.3f}".format(np.sum(s["area"])))

    inlet = boundaries[5]
    inlet = convert_float64_to_float32(inlet)
    inlet_xyz = np.concatenate((inlet['x'], inlet['y'], inlet['z']), axis=1)
    inlet_txyz = replicate_t(t_num, t_index, t_step, inlet_xyz)

    # outlet = boundaries[1]
    outlet = boundaries[4]
    outlet = convert_float64_to_float32(outlet)
    outlet_xyz = np.concatenate(
        (outlet['x'], outlet['y'], outlet['z']), axis=1)
    outlet_txyz = replicate_t(t_num, t_index, t_step, outlet_xyz)

    cylinder = boundaries[6]
    cylinder = convert_float64_to_float32(cylinder)
    cylinder_xyz = np.concatenate(
        (cylinder['x'], cylinder['y'], cylinder['z']), axis=1)
    cylinder_txyz = replicate_t(t_num, t_index, t_step, cylinder_xyz)

    front = boundaries[0]
    front = convert_float64_to_float32(front)
    front_xyz = np.concatenate((front['x'], front['y'], front['z']), axis=1)
    front_txyz = replicate_t(t_num, t_index, t_step, front_xyz)

    back = boundaries[1]
    back = convert_float64_to_float32(back)
    back_xyz = np.concatenate((back['x'], back['y'], back['z']), axis=1)
    back_txyz = replicate_t(t_num, t_index, t_step, back_xyz)

    top = boundaries[2]
    top = convert_float64_to_float32(top)
    top_xyz = np.concatenate((top['x'], top['y'], top['z']), axis=1)
    top_txyz = replicate_t(t_num, t_index, t_step, top_xyz)

    bottom = boundaries[3]
    bottom = convert_float64_to_float32(bottom)
    bottom_xyz = np.concatenate(
        (bottom['x'], bottom['y'], bottom['z']), axis=1)
    bottom_txyz = replicate_t(t_num, t_index, t_step, bottom_xyz)

    print("Sampling Domain data ......")
    interior = geo.sample_interior(
        nr_points=nr_points, compute_sdf_derivatives=True)
    interior = convert_float64_to_float32(interior)
    interior_xyz = np.concatenate(
        (interior['x'], interior['y'], interior['z']), axis=1)
    interior1 = geo1.sample_interior(
        nr_points=nr_points, compute_sdf_derivatives=True)
    interior1 = convert_float64_to_float32(interior1)
    interior1_xyz = np.concatenate(
        (interior1['x'], interior1['y'], interior1['z']), axis=1)
    interior2_xyz = np.concatenate((interior_xyz, interior1_xyz), axis=0)
    # interior_txyz = replicate_t(t_num, t_step, interior_xyz)
    interior2_txyz = replicate_t(t_num, t_index, t_step, interior2_xyz)
    var_to_polyvtk(interior, "interior")
    print("Volume: {:.3f}".format(np.sum(s["area"])))

    # for item in [inlet_txyz, outlet_txyz, front_txyz, back_txyz, top_txyz, bottom_txyz, cylinder_txyz, interior2_txyz]:
    #     # Normalize x,y,z to [0,1]
    #     for coordinate, interval in domain_coordinate_interval_dict.items():
    #         min_domain = interval[0]
    #         max_domain = interval[1]
    #         normalize(max_domain, min_domain, item, coordinate)

    return inlet_txyz, outlet_txyz, front_txyz, back_txyz, top_txyz, bottom_txyz, cylinder_txyz, interior2_txyz


def sample_data_mesh(t_num, t_index, t_step=50, nr_points=4000):
    # make standard constructive solid geometry example
    # make primitives
    box = psci.neo_geometry.Mesh("Box1.stl")
    box2 = psci.neo_geometry.Mesh("Box2.stl")

    cylinder = psci.neo_geometry.Mesh("cylinder.stl")
    # combine with boolean operations
    geo = box - cylinder
    geo1 = box2 - cylinder
    # inlet (630, 3)
    # outlet (618, 3)
    # front (, 3)
    # back (, 3)
    # top (1265, 3)
    # bottom (1255, 3)
    # cylinder (202, 3)
    geo.add_sample_config("interior", nr_points)
    geo.add_sample_config(
        "boundary_inlet", 630, criteria=lambda x, y, z: np.isclose(x, 0))
    geo.add_sample_config(
        "boundary_outlet", 618, criteria=lambda x, y, z: np.isclose(x, 1600))
    geo.add_sample_config(
        "boundary_front", 2969, criteria=lambda x, y, z: np.isclose(z, 320))
    geo.add_sample_config(
        "boundary_back", 3061, criteria=lambda x, y, z: np.isclose(z, 0))
    geo.add_sample_config(
        "boundary_top", 1265, criteria=lambda x, y, z: np.isclose(y, 800))
    geo.add_sample_config(
        "boundary_bottom", 1255, criteria=lambda x, y, z: np.isclose(y, 0))
    geo.add_sample_config(
        "boundary_cylinder",
        202,
        criteria=lambda x, y, z: np.isclose((x - 400)**2 + (y - 400)**2, 1600))

    geo1.add_sample_config("interior", nr_points)

    print("Sampling Boundary data ......")
    geo_points_dict = geo.fetch_batch_data()
    geo1_points_dict = geo1.fetch_batch_data()

    inlet = geo_points_dict["boundary_inlet"]
    inlet_txyz = replicate_t(t_num, t_index, t_step, inlet)

    outlet = geo_points_dict["boundary_outlet"]
    outlet_txyz = replicate_t(t_num, t_index, t_step, outlet)

    cylinder = geo_points_dict["boundary_cylinder"]
    cylinder_txyz = replicate_t(t_num, t_index, t_step, cylinder)

    front = geo_points_dict["boundary_front"]
    front_txyz = replicate_t(t_num, t_index, t_step, front)

    back = geo_points_dict["boundary_back"]
    back_txyz = replicate_t(t_num, t_index, t_step, back)

    top = geo_points_dict["boundary_top"]
    top_txyz = replicate_t(t_num, t_index, t_step, top)

    bottom = geo_points_dict["boundary_bottom"]
    bottom_txyz = replicate_t(t_num, t_index, t_step, bottom)

    print("Sampling Domain data ......")
    interior = geo_points_dict["interior"]
    interior1 = geo1_points_dict["interior"]
    interior2_xyz = np.concatenate((interior, interior1), axis=0)
    interior2_txyz = replicate_t(t_num, t_index, t_step, interior2_xyz)

    return inlet_txyz, outlet_txyz, front_txyz, back_txyz, top_txyz, bottom_txyz, cylinder_txyz, interior2_txyz


def convert_float64_to_float32(dict_a):
    for k in dict_a:
        dict_a[k] = dict_a[k].astype(np.float32)
    return dict_a


def replicate_t(t_num, t_index, t_step, data):
    full_data = None
    # for time in range(1, t_num+1):
    for time in t_index:
        t_len = data.shape[0]
        t_extended = np.array(
            [time * t_step] * t_len, dtype=np.float32).reshape((-1, 1))
        t_data = np.concatenate((t_extended, data), axis=1)
        if full_data is None:
            full_data = t_data
        else:
            full_data = np.concatenate((full_data, t_data))

    return full_data


def get_boundary_data_and_training_data(time_num, num_points):
    inlet_txyz, outlet_txyz, front_txyz, back_txyz, top_txyz, bottom_txyz, cylinder_txyz, interior_txyz = \
        sample_data(t_step=time_num, nr_points=num_points)

    t_step = 1

    inlet_uvwp = np.random.rand(inlet_txyz.shape[0], 4)
    inlet_uvwp[:, 0] = 0.1
    inlet_uvwp[:, 1] = 0
    inlet_uvwp[:, 2] = 0
    inlet_uvwp[:, 3] = 0
    inlet_txyz_uvwp = np.concatenate((inlet_txyz, inlet_uvwp), axis=1)
    inlet_t = inlet_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    inlet_x = inlet_txyz_uvwp[:, 1].reshape((-1, 1))
    inlet_y = inlet_txyz_uvwp[:, 2].reshape((-1, 1))
    inlet_z = inlet_txyz_uvwp[:, 3].reshape((-1, 1))
    inlet_u = inlet_txyz_uvwp[:, 4].reshape((-1, 1))
    inlet_v = inlet_txyz_uvwp[:, 5].reshape((-1, 1))
    inlet_w = inlet_txyz_uvwp[:, 6].reshape((-1, 1))
    inlet_p = inlet_txyz_uvwp[:, 7].reshape((-1, 1))

    front_uvwp = np.random.rand(front_txyz.shape[0], 4)
    front_uvwp[:, 0] = 0.1
    front_uvwp[:, 1] = 0
    front_uvwp[:, 2] = 0
    front_uvwp[:, 3] = 0
    front_txyz_uvwp = np.concatenate((front_txyz, front_uvwp), axis=1)
    front_t = front_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    front_x = front_txyz_uvwp[:, 1].reshape((-1, 1))
    front_y = front_txyz_uvwp[:, 2].reshape((-1, 1))
    front_z = front_txyz_uvwp[:, 3].reshape((-1, 1))
    front_u = front_txyz_uvwp[:, 4].reshape((-1, 1))
    front_v = front_txyz_uvwp[:, 5].reshape((-1, 1))
    front_w = front_txyz_uvwp[:, 6].reshape((-1, 1))
    front_p = front_txyz_uvwp[:, 7].reshape((-1, 1))

    back_uvwp = np.random.rand(back_txyz.shape[0], 4)
    back_uvwp[:, 0] = 0.1
    back_uvwp[:, 1] = 0
    back_uvwp[:, 2] = 0
    back_uvwp[:, 3] = 0
    back_txyz_uvwp = np.concatenate((back_txyz, back_uvwp), axis=1)
    back_t = back_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    back_x = back_txyz_uvwp[:, 1].reshape((-1, 1))
    back_y = back_txyz_uvwp[:, 2].reshape((-1, 1))
    back_z = back_txyz_uvwp[:, 3].reshape((-1, 1))
    back_u = back_txyz_uvwp[:, 4].reshape((-1, 1))
    back_v = back_txyz_uvwp[:, 5].reshape((-1, 1))
    back_w = back_txyz_uvwp[:, 6].reshape((-1, 1))
    back_p = back_txyz_uvwp[:, 7].reshape((-1, 1))

    top_uvwp = np.random.rand(top_txyz.shape[0], 4)
    top_uvwp[:, 0] = 0.1
    top_uvwp[:, 1] = 0
    top_uvwp[:, 2] = 0
    top_uvwp[:, 3] = 0
    top_txyz_uvwp = np.concatenate((top_txyz, top_uvwp), axis=1)
    top_t = top_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    top_x = top_txyz_uvwp[:, 1].reshape((-1, 1))
    top_y = top_txyz_uvwp[:, 2].reshape((-1, 1))
    top_z = top_txyz_uvwp[:, 3].reshape((-1, 1))
    top_u = top_txyz_uvwp[:, 4].reshape((-1, 1))
    top_v = top_txyz_uvwp[:, 5].reshape((-1, 1))
    top_w = top_txyz_uvwp[:, 6].reshape((-1, 1))
    top_p = top_txyz_uvwp[:, 7].reshape((-1, 1))

    bottom_uvwp = np.random.rand(bottom_txyz.shape[0], 4)
    bottom_uvwp[:, 0] = 0.1
    bottom_uvwp[:, 1] = 0
    bottom_uvwp[:, 2] = 0
    bottom_uvwp[:, 3] = 0
    bottom_txyz_uvwp = np.concatenate((bottom_txyz, bottom_uvwp), axis=1)
    bottom_t = bottom_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    bottom_x = bottom_txyz_uvwp[:, 1].reshape((-1, 1))
    bottom_y = bottom_txyz_uvwp[:, 2].reshape((-1, 1))
    bottom_z = bottom_txyz_uvwp[:, 3].reshape((-1, 1))
    bottom_u = bottom_txyz_uvwp[:, 4].reshape((-1, 1))
    bottom_v = bottom_txyz_uvwp[:, 5].reshape((-1, 1))
    bottom_w = bottom_txyz_uvwp[:, 6].reshape((-1, 1))
    bottom_p = bottom_txyz_uvwp[:, 7].reshape((-1, 1))

    outlet_uvwp = np.random.rand(outlet_txyz.shape[0], 4)
    outlet_uvwp[:, 0] = 0
    outlet_uvwp[:, 1] = 0
    outlet_uvwp[:, 2] = 0
    outlet_uvwp[:, 3] = 0
    outlet_txyz_uvwp = np.concatenate((outlet_txyz, outlet_uvwp), axis=1)
    outlet_t = outlet_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    outlet_x = outlet_txyz_uvwp[:, 1].reshape((-1, 1))
    outlet_y = outlet_txyz_uvwp[:, 2].reshape((-1, 1))
    outlet_z = outlet_txyz_uvwp[:, 3].reshape((-1, 1))
    outlet_u = outlet_txyz_uvwp[:, 4].reshape((-1, 1))
    outlet_v = outlet_txyz_uvwp[:, 5].reshape((-1, 1))
    outlet_w = outlet_txyz_uvwp[:, 6].reshape((-1, 1))
    outlet_p = outlet_txyz_uvwp[:, 7].reshape((-1, 1))

    cylinder_uvwp = np.random.rand(cylinder_txyz.shape[0], 4)
    cylinder_uvwp[:, 0] = 0
    cylinder_uvwp[:, 1] = 0
    cylinder_uvwp[:, 2] = 0
    cylinder_uvwp[:, 3] = 0
    cylinder_txyz_uvwp = np.concatenate((cylinder_txyz, cylinder_uvwp), axis=1)
    cylinder_t = cylinder_txyz_uvwp[:, 0].reshape((-1, 1)) * t_step
    cylinder_x = cylinder_txyz_uvwp[:, 1].reshape((-1, 1))
    cylinder_y = cylinder_txyz_uvwp[:, 2].reshape((-1, 1))
    cylinder_z = cylinder_txyz_uvwp[:, 3].reshape((-1, 1))
    cylinder_u = cylinder_txyz_uvwp[:, 4].reshape((-1, 1))
    cylinder_v = cylinder_txyz_uvwp[:, 5].reshape((-1, 1))
    cylinder_w = cylinder_txyz_uvwp[:, 6].reshape((-1, 1))
    cylinder_p = cylinder_txyz_uvwp[:, 7].reshape((-1, 1))

    interior_t = interior_txyz[:, 0].reshape((-1, 1)) * t_step
    interior_x = interior_txyz[:, 1].reshape((-1, 1))
    interior_y = interior_txyz[:, 2].reshape((-1, 1))
    interior_z = interior_txyz[:, 3].reshape((-1, 1))

    inlet_data = (inlet_t, inlet_x, inlet_y, inlet_z, inlet_u, inlet_v,
                  inlet_w, inlet_p)
    front_data = (front_t, front_x, front_y, front_z, front_u, front_v,
                  front_w, front_p)
    back_data = (back_t, back_x, back_y, back_z, back_u, back_v, back_w,
                 back_p)
    top_data = (top_t, top_x, top_y, top_z, top_u, top_v, top_w, top_p)
    bottom_data = (bottom_t, bottom_x, bottom_y, bottom_z, bottom_u, bottom_v,
                   bottom_w, bottom_p)
    cylinder_data = (cylinder_t, cylinder_x, cylinder_y, cylinder_z,
                     cylinder_u, cylinder_v, cylinder_w, cylinder_p)
    outlet_data = (outlet_t, outlet_x, outlet_y, outlet_z, outlet_u, outlet_v,
                   outlet_w, outlet_p)
    interior_data = (interior_t, interior_x, interior_y, interior_z)
    return inlet_data, front_data, back_data, top_data, bottom_data, cylinder_data, outlet_data, interior_data
