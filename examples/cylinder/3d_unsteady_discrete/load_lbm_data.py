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
"""
Created in Mar. 2023
@author: Guan Wang
"""
import meshio
import numpy as np


def load_vtk(time_list,
             t_step,
             load_uvwp=False,
             load_txyz=False,
             name_wt_time=None):
    """load LBM(traditional methodology) points coordinates, use these points as interior points for tranning_summary_
    Args:
        time_list (list): time step index
        t_step (int): the time that one step cost
        load_uvwp (bool, optional): choose to load uvwp data or not. Defaults to False.
        load_txyz (bool, optional): choose to load txyz data or not. Defaults to False.
        name_wt_time (str, optional): input file directory minus its tail. Defaults to None.
    Returns:
        list: txyzuvwp
    """
    return_list = []
    for i in time_list:
        file = name_wt_time + f'{i}.vtu'
        mesh = meshio.read(file)
        n = mesh.points.shape[0]
        part1, part2 = np.zeros((n, 4)), np.zeros((n, 4))
        if load_txyz == True:
            t = np.full((n, 1), int(i * t_step))
            x = mesh.points[:, 0].reshape(n, 1)
            y = mesh.points[:, 1].reshape(n, 1)
            z = mesh.points[:, 2].reshape(n, 1)
            part1 = np.concatenate((t, x, y, z), axis=1).astype(np.float32)
        if load_uvwp == True:
            u = np.array(mesh.point_data['1'])
            v = np.array(mesh.point_data['2'])
            w = np.array(mesh.point_data['3'])
            p = np.array(mesh.point_data['4'])
            part2 = np.concatenate((u, v, w, p), axis=1).astype(np.float32)
        return_list.append(np.concatenate((part1, part2), axis=1))
    return return_list


def load_msh(file):
    """load mesh file and read the mesh information
    Args:
        file (str): input file directory
    Returns:
        np.array : mesh coordinates
        mesh : return the mesh object
    """
    mesh = meshio.read(file)
    n = mesh.points.shape[0]
    cord = np.zeros((n, 4))
    t = np.full((n, 1), int(0))
    x = mesh.points[:, 0].reshape(n, 1)
    y = mesh.points[:, 1].reshape(n, 1)
    z = mesh.points[:, 2].reshape(n, 1)
    cord = np.concatenate((t, x, y, z), axis=1).astype(np.float32)
    return cord, mesh


def write_vtu(file, mesh, solution):
    """Write vtk file by concatenate mesh and solution
    Args:
        file (str): output directory
        mesh (mesh): mesh object
        solution (np.array): results matrix compose of result vectors like : velocity, pressure ...
    """
    point_data_dic = {}
    point_data_dic['u'] = solution[:, 0]
    point_data_dic['v'] = solution[:, 1]
    point_data_dic['w'] = solution[:, 2]
    point_data_dic['p'] = solution[:, 3]
    mesh.point_data = point_data_dic
    mesh.write(file)
    print(f"vtk_raw file saved at [{file}]")
