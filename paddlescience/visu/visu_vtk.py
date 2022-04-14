# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import vtk
from pyevtk.hl import pointsToVTK
import copy


# Save geometry pointwise
def save_vtk(filename="output", geo_disc=None, data=None):

    # concatenate data and cordiante 
    points_vtk = __concatenate_geo(geo_disc)
    data_vtk = __concatenate_data(data)

    # points's shape is [ndims][npoints]
    npoints = len(points_vtk[0])
    ndims = len(points_vtk)

    if ndims == 3:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = points_vtk[2]
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)
    elif ndims == 2:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = np.zeros(npoints, dtype="float32")
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


# concatenate data
def __concatenate_data(outs):

    varname = ["u", "v", "w", "p"]

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
        data[varname[i]] = np.concatenate(x, axis=0)

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
