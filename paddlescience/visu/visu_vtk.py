# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import copy
import types
import vtk
from pyevtk.hl import pointsToVTK
import paddle
from .. import config


# Save geometry pointwise
def save_vtk(filename="output", time_array=None, geo_disc=None, data=None):
    """
    Visualization. Save data to vtk format.

    Parameters:
        filename(string): file name.
        time_array(list or numpy array, optional): time steps list / array.
        geo_disc (GeometryDiscrete): discrete geometry.
        data (numpy array, optional): data to be visualized.


    Example:
        >>> import paddlescience as psci
        >>> psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)
    """

    nt = 1 if (time_array is None) else len(time_array) - 1
    nprocs = paddle.distributed.get_world_size()
    nrank = paddle.distributed.get_rank()
    if nprocs == 1:
        geo_disc_sub = geo_disc
    else:
        geo_disc_sub = geo_disc.sub(nprocs, nrank)

    # concatenate data and cordiante
    points_vtk = __concatenate_geo(geo_disc_sub)

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

    if ndims == 3:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = points_vtk[2]
    elif ndims == 2:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = np.zeros(npoints, dtype=config._dtype)

    if data is not None:
        for t in range(nt):
            fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])
    else:
        for t in range(nt):
            fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
            __save_vtk_raw(filename=fpname, cordinate=np.array(points_vtk).T)


def test_save(filename='output', time_array=None, cord=None, data=None):

    # concatenate data and cordiante
    points_vtk = __concatenate_cord(cord)

    # points's shape is [ndims][npoints]
    npoints = len(points_vtk[0])
    ndims = len(points_vtk)
    nt = 1 if (time_array is None) else len(time_array) - 1
    # nprocs = paddle.distributed.get_world_size()
    nrank = paddle.distributed.get_rank()
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

    if ndims == 3:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = points_vtk[2]
        for t in range(nt):
            fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])
    elif ndims == 2:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = np.zeros(npoints, dtype=config._dtype)
        for t in range(nt):
            fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])


def save_vtk_cord(filename="./visual/output", time_array=None, cord=None, data=None):
    npoints = len(cord)
    # ndims = len(points_vtk)
    ndims = len(cord[0])
    # concatenate data and cordiante
    points_vtk = __concatenate_cord(cordinates=cord, nd=ndims) # Tuple[shape[30000,], shape[30000,], shape[30000,]]
    # points_vtk[0] *= 80
    # points_vtk[1] *= 40
    # points_vtk[2] *= 30
    # print(points_vtk[0].shape, points_vtk[0].min(), points_vtk[0].max())
    # print(points_vtk[1].shape, points_vtk[1].min(), points_vtk[1].max())
    # print(points_vtk[2].shape, points_vtk[2].min(), points_vtk[2].max())
    # exit()
    # points's shape is [ndims][npoints]
    # npoints = len(points_vtk[0])

    nt = 1 if (time_array is None) else len(time_array)
    print(f"nt = {nt}")

    # nprocs = paddle.distributed.get_world_size()
    nrank = paddle.distributed.get_rank()
    # data
    if data is None:
        data_vtk = {"placeholder": np.ones(npoints, dtype=config._dtype)}
    elif type(data) == types.LambdaType:
        data_vtk = dict()
        if ndims == 3:
            data_vtk["data"] = data(points_vtk[0], points_vtk[1], points_vtk[2])
        elif ndims == 2:
            data_vtk["data"] = data(points_vtk[0], points_vtk[1])
    else:
        data_vtk = __concatenate_data(data, nt)
        # for t in range(nt):
            # print(data_vtk[t].keys())
            # for k in data_vtk[t]:
                # print(points_vtk[0].shape)
                # print(points_vtk[1].shape)
                # print(points_vtk[2].shape)
                # print(data_vtk[t][k].shape)
            # print("\n")
            # exit()
        # [(t1){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}
        #  (t2){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}
        #  (t3){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}
        #  (t4){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}]

    if ndims == 3:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = points_vtk[2]
        for t in range(nt):
            fpname = filename + "-p" + str(nrank) + "-t" + str(t + 1); print(fpname)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])
    elif ndims == 2:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = np.zeros(npoints, dtype=config._dtype)
        for t in range(nt):
            fpname = filename + "-p" + str(nrank) + "-t" + str(t + 1); print(fpname)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])


# def save_vtk_cord(filename="./visual/output", time_array=None, cord=None, data=None):

#     npoints = len(cord)
#     # ndims = len(points_vtk)
#     ndims = len(cord[0])
#     # concatenate data and cordiante
#     points_vtk = __concatenate_cord(cordinates=cord, nd=ndims)

#     points_vtk[0] *= 80
#     points_vtk[1] *= 40
#     points_vtk[2] *= 30
#     # print(points_vtk[0].shape, points_vtk[0].min(), points_vtk[0].max())
#     # print(points_vtk[1].shape, points_vtk[1].min(), points_vtk[1].max())
#     # print(points_vtk[2].shape, points_vtk[2].min(), points_vtk[2].max())
#     # exit()
#     # points's shape is [ndims][npoints]
#     # npoints = len(points_vtk[0])

#     nt = 1 if (time_array is None) else len(time_array) - 1
#     # nprocs = paddle.distributed.get_world_size()
#     nrank = paddle.distributed.get_rank()
#     # data
#     if data is None:
#         data_vtk = {"placeholder": np.ones(npoints, dtype=config._dtype)}
#     elif type(data) == types.LambdaType:
#         data_vtk = dict()
#         if ndims == 3:
#             data_vtk["data"] = data(points_vtk[0], points_vtk[1],
#                                     points_vtk[2])
#         elif ndims == 2:
#             data_vtk["data"] = data(points_vtk[0], points_vtk[1])
#     else:
#         data_vtk = __concatenate_data(data, nt)

#     if ndims == 3:
#         axis_x = points_vtk[0]
#         axis_y = points_vtk[1]
#         axis_z = points_vtk[2]
#         for t in range(nt):
#             fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
#             pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])
#     elif ndims == 2:
#         axis_x = points_vtk[0]
#         axis_y = points_vtk[1]
#         axis_z = np.zeros(npoints, dtype=config._dtype)
#         for t in range(nt):
#             fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
#             pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])


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
        axis_x = cordinate[:, 0].copy().astype(config._dtype)
        axis_y = cordinate[:, 1].copy().astype(config._dtype)
        axis_z = np.zeros(npoints, dtype=config._dtype)
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)


# concatenate cordinates of interior points and boundary points
def __concatenate_geo(geo_disc):

    # concatenate interior and bounday points
    x = [geo_disc.interior]
    for value in geo_disc.boundary.values():
        x.append(value)
    points = np.concatenate(x, axis=0)

    ndims = len(points[0])

    # to pointsToVTK input format
    points_vtk = list()
    for i in range(ndims):
        points_vtk.append(points[:, i].copy())

    return points_vtk


def __concatenate_cord(cordinates=None, nd=None):
    x = []
    # print('cordinates.shape: ', cordinates.shape) # (150000, 3)
    for cord in cordinates:
        x.append(cord)
    points = np.concatenate(x, axis=0)
    points = points.reshape(cordinates.shape[0],cordinates.shape[1])
    # print(points.shape)
    # print(np.allclose(cordinates, points))
    # exit(0)

    # to pointsToVTK input format
    # 变成[[x1,x2,....,xn], [y1,y2,....,yn], [z1,z2,....,zn]]的list，类型为Tuple[shape[N,], shape[N,], shape[N,]]
    points_vtk = list()
    for i in range(nd):
        points_vtk.append(points[:, i].copy())
    return points_vtk


# concatenate data
def __concatenate_data(outs, nt=1):
    # print([x.shape for x in outs])
    # outs: [(150000, 4), (4695, 4), (1570, 4), (4600, 4), (179260, 4)]
    # outs: [(150000, 4), (4515, 4), (1320, 4), (4725, 4), (9280, 4), (9220, 4), (179260, 4), (82200, 4)]
    vtkname = ["u1", "u2", "u3", "u4", "u5"]

    # to numpy
    npouts = []
    for out in outs:
        if type(out) != np.ndarray:
            npouts.append(out.numpy())  # tenor to array
        else:
            npouts.append(out)

    # concatenate data
    ndata = outs[0].shape[1] # 4
    data_vtk = []  # list[dict[str, np.ndarray]]

    # 1~nt solutions
    for t in range(nt):
        data_t = {}
        for i in range(ndata):
            x = []
            for out in npouts:
                s = int(len(out) / nt) * t
                e = int(len(out) / nt) * (t + 1)
                x.append(out[s:e, i])
            data_t[vtkname[i]] = np.concatenate(x, axis=0)
        data_vtk.append(data_t)
    # for u in vtkname[:-1]:
    #     print(f"{np.allclose(data_vtk[0][u], data_vtk[1][u])} {(data_vtk[0][u] - data_vtk[1][u]).mean().item():.5f}")
    #     print(f"{np.allclose(data_vtk[0][u], data_vtk[2][u])} {(data_vtk[0][u] - data_vtk[2][u]).mean().item():.5f}")
    #     print(f"{np.allclose(data_vtk[0][u], data_vtk[3][u])} {(data_vtk[0][u] - data_vtk[3][u]).mean().item():.5f}")
    #     print(f"{np.allclose(data_vtk[0][u], data_vtk[4][u])} {(data_vtk[0][u] - data_vtk[4][u]).mean().item():.5f}")
    # exit()
    # [(t1){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}
    #  (t2){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}
    #  (t3){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}
    #  (t4){"u1": ndarray, "u2": ndarray, ..., "u4": ndarray}]
    return data_vtk