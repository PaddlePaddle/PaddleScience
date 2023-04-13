# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List

import meshio
import numpy as np

import ppsci.data.dataset as dataset


class Reader:
    def __init__(
        self,
        time_index: List[int],
        time_step: int,
    ):
        self.time_index = time_index
        self.time_step = time_step

    def vtk(
        self,
        filename_without_timeid: str,
        time_point=None,
        read_input: bool = True,
        read_label: bool = True,
        dim=3,
    ):
        if time_point is None:
            time_index = self.time_index
        else:
            time_index = [time_point]
        time_step = self.time_step
        for i, t in enumerate(time_index):
            file = filename_without_timeid + f"{t}.vtu"
            mesh = meshio.read(file)
            if i == 0:
                n = mesh.points.shape[0]
                input_dict = {
                    var: np.zeros((len(time_index) * n, 1)).astype(np.float32)
                    for var in dataset.Input
                }
                label_dict = {
                    var: np.zeros((len(time_index) * n, 1)).astype(np.float32)
                    for var in dataset.Label
                }
            if read_input == True:
                input_dict[dataset.Input.t][i * n : (i + 1) * n] = np.full(
                    (n, 1), int(t * time_step)
                )
                input_dict[dataset.Input.x][i * n : (i + 1) * n] = mesh.points[
                    :, 0
                ].reshape(n, 1)
                input_dict[dataset.Input.y][i * n : (i + 1) * n] = mesh.points[
                    :, 1
                ].reshape(n, 1)
                if dim == 3:
                    input_dict[dataset.Input.z][i * n : (i + 1) * n] = mesh.points[
                        :, 2
                    ].reshape(n, 1)
            if read_label == True:
                label_dict[dataset.Label.u][i * n : (i + 1) * n] = np.array(
                    mesh.point_data["1"]
                )
                label_dict[dataset.Label.v][i * n : (i + 1) * n] = np.array(
                    mesh.point_data["2"]
                )
                if dim == 3:
                    label_dict[dataset.Label.w][i * n : (i + 1) * n] = np.array(
                        mesh.point_data["3"]
                    )
                label_dict[dataset.Label.p][i * n : (i + 1) * n] = np.array(
                    mesh.point_data["4"]
                )
        return input_dict, label_dict

    def vtk_samples_with_time(self, file: str):
        mesh = meshio.read(file)
        n = mesh.points.shape[0]
        t = np.array(mesh.point_data["time"])
        x = mesh.points[:, 0].reshape(n, 1)
        y = mesh.points[:, 1].reshape(n, 1)
        z = mesh.points[:, 2].reshape(n, 1)
        txyz = np.concatenate((t, x, y, z), axis=1).astype(np.float32).reshape(n, 4, 1)
        return txyz


class Writer:
    def __init__(self):
        pass

    def vtk(self, filename, label, coordinates):
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)

        n = len(next(iter(coordinates.values())))
        m = len(coordinates)
        # get the list variable transposed
        points = np.stack((x for x in iter(coordinates.values()))).reshape(m, n)
        mesh = meshio.Mesh(
            points=points.T, cells=[("vertex", np.arange(n).reshape(n, 1))]
        )
        mesh.point_data = label
        mesh.write(filename)
        print(f"vtk_raw file saved at [{filename}]")
