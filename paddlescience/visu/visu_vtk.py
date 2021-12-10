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

import vtk


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
