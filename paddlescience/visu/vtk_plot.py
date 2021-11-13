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

def Rectangular2D(geo, data, filename="output"):
    # plane obj
    nPoints, plane = geo.visu_vtk()
    # set data
    data_vtk = vtk.vtkFloatArray()
    data_vtk.SetNumberOfValues(nPoints)
    for i in range(nPoints):
        data_vtk.SetValue(i, data[i])
    plane.GetOutput().GetPointData().SetScalars(data_vtk)
    # writer
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + '.vtp')
    writer.SetInputConnection(plane.GetOutputPort())
    writer.Write()