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

    # geo to numpy
    geonp = geo.space_domain.numpy()

    # copy for vtk
    for key in data:
        data[key] = data[key].copy()

    if geo.space_dims == 3:
        # pointsToVTK requires continuity in memory
        axis_x = geonp[:, 0].copy()
        axis_y = geonp[:, 1].copy()
        axis_z = geonp[:, 2].copy()
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data)
    elif geo.space_dims == 2:
        axis_x = geonp[:, 0].copy()
        axis_y = geonp[:, 1].copy()
        axis_z = np.zeros(len(geonp), dtype="float32")
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data)


# concatenate cordinates of interior points and boundary points
def __concatenate_geo(geo_disc):

    x = [geo_disc.interior]
    for value in geo_disc.boundary.values():
        x.append(value)
    points = np.concatenate(x, axis=0)

    return points


def __concatenate_data(outs):

    vname = ["u", "v", "w"]

    data = dict()
    ndata = outs[0].shape[1]
    for i in range(ndata):
        x = list()
        for out in outs:
            x.append(out[:, i])
        data[vname[i]] = np.concatenate(x, axis=0)

    return data


def save_vtk(geo, data, filename="output"):
    """
    Save geometry and data to vtk file for visualisation

    Parameters:
        geo: geometry
        
        data: data to save

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.visu.save_vtk(geo, data, filename="output")

    """
    # plane obj
    vtkobjname, vtkobj, nPoints = geo.get_vtk_obj()
    # data
    data_vtk = vtk.vtkFloatArray()
    data_vtk.SetNumberOfValues(nPoints)
    for i in range(nPoints):
        data_vtk.SetValue(i, data[i])

    if vtkobjname == "vtkPlanceSource":
        # set data
        vtkobj.GetOutput().GetPointData().SetScalars(data_vtk)
        # writer
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + '.vtp')
        writer.SetInputConnection(vtkobj.GetOutputPort())
        writer.Write()
    elif vtkobjname == "vtkImageData":
        # set data
        vtkobj.GetPointData().SetScalars(data_vtk)
        # writer
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename + ".vti")
        writer.SetInputData(vtkobj)
        writer.Write()
