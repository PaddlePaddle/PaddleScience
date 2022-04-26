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


# Save geometry pointwise
def save_vtk(filename="output", geo_disc=None, data=None):

    nprocs = paddle.distributed.get_world_size()
    nrank = paddle.distributed.get_rank()
    fpname = filename + str(nrank)
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
        data_vtk = {"placeholder": np.ones(npoints, dtype="float32")}
    elif type(data) == types.LambdaType:
        data_vtk = dict()
        if ndims == 3:
            data_vtk["data"] = data(points_vtk[0], points_vtk[1],
                                    points_vtk[2])
        elif ndims == 2:
            data_vtk["data"] = data(points_vtk[0], points_vtk[1])
    else:
        data_vtk = __concatenate_data(data)

    if ndims == 3:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = points_vtk[2]
        pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk)
    elif ndims == 2:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = np.zeros(npoints, dtype="float32")
        pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk)


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


# concatenate data
def __concatenate_data(outs):

    vtkname = ["u", "v", "p", "w"]

    data = dict()

    # to numpy
    npouts = list()
    for out in outs:
        if type(out) != np.ndarray:
            npouts.append(out.numpy())  # tenor to array
        else:
            npouts.append(out)

    # concatenate data
    ndata = outs[0].shape[1]
    for i in range(ndata):
        x = list()
        for out in npouts:
            x.append(out[:, i])
        data[vtkname[i]] = np.concatenate(x, axis=0)

    return data


# def save_vtk(geo, data, filename="output"):
#     """
#     Save geometry and data to vtk file for visualisation

#     Parameters:
#         geo: geometry

#         data: data to save

#     Example:
#         >>> import paddlescience as psci
#         >>> pde = psci.visu.save_vtk(geo, data, filename="output")

#     """
#     # plane obj
#     vtkobjname, vtkobj, nPoints = geo.get_vtk_obj()
#     # data
#     data_vtk = vtk.vtkFloatArray()
#     data_vtk.SetNumberOfValues(nPoints)
#     for i in range(nPoints):
#         data_vtk.SetValue(i, data[i])

#     if vtkobjname == "vtkPlanceSource":
#         # set data
#         vtkobj.GetOutput().GetPointData().SetScalars(data_vtk)
#         # writer
#         writer = vtk.vtkXMLPolyDataWriter()
#         writer.SetFileName(filename + '.vtp')
#         writer.SetInputConnection(vtkobj.GetOutputPort())
#         writer.Write()
#     elif vtkobjname == "vtkImageData":
#         # set data
#         vtkobj.GetPointData().SetScalars(data_vtk)
#         # writer
#         writer = vtk.vtkXMLImageDataWriter()
#         writer.SetFileName(filename + ".vti")
#         writer.SetInputData(vtkobj)
#         writer.Write()
