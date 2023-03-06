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
Created in Oct. 2022
@author: Hui Xiang, Yanbo Zhang
"""

import os
import numpy as np


def load_vtk(time_list, t_step, load_uvwp=False, load_txyz=False, name_wt_time=None):
    """_summary_
    load LBM(traditional methodology) points coordinates, use these points as interior points for tranning
    """
    import sys
    import meshio
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
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
            part1 = np.concatenate((t, x, y, z), axis = 1).astype(np.float32)
            #part1.append((np.concatenate((t, x, y, z), axis = 1)).tolist())
        if load_uvwp == True:
            u = np.array(mesh.point_data['1'])
            v = np.array(mesh.point_data['2'])
            w = np.array(mesh.point_data['3'])
            p = np.array(mesh.point_data['4'])
            part2 = np.concatenate((u, v, w, p), axis = 1).astype(np.float32)
            # part2.append((np.concatenate((u, v, w, p), axis = 1)).tolist())
        return_list.append(np.concatenate((part1, part2), axis = 1))
    return return_list
