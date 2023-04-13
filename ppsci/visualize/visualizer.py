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

import os.path as osp
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np

import ppsci.data.dataset as dataset
from ppsci.visualize import base
from ppsci.visualize import plot
from ppsci.visualize import vtu


class VisualizerScatter1D(base.Visualizer):
    """Visualizer for 1d scatter data.

    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        num_timestamps (int): Number of timestamps
        prefix (str): Prefix for output file.
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        num_timestamps: int = 1,
        prefix: str = "plot",
    ):
        super().__init__(input_dict, output_expr, num_timestamps, prefix)

    def save(self, data_dict, filename):
        plot.save_plot_from_1d_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )


class VisualizerVtu(base.Visualizer):
    """Visualizer for 2D points data.

    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        num_timestamps (int): Number of timestamps
        prefix (str): Prefix for output file.
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        num_timestamps: int = 1,
        prefix: str = "vtu",
    ):
        super().__init__(input_dict, output_expr, num_timestamps, prefix)

    def save(self, filename, data_dict):
        vtu.save_vtu_from_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )


class Visualizer3D(base.Visualizer):
    """Visualizer for 3D plot data.

    Args:
        time_step (Union[int, float, None]): Time step to predict.
        time_list (List[int]): Time index list to predict
        factor_dict (Dict): Factors dict for scaling
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        ref_file (str): Baseline file dir
        visualizer_batch_size (int): Split input dict to fit GPU cache
        num_timestamps (int): Number of timestamps
        prefix (str): Prefix for output file.
    """

    def __init__(
        self,
        time_step: Union[int, float, None],
        time_list: List[int],
        factor_dict: Dict,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        ref_file: str,
        visualizer_batch_size: int = 4e5,
        num_timestamps: int = 1,
        prefix: str = "vtu",
    ):

        super().__init__(input_dict, output_expr, num_timestamps, prefix)
        self.time_list = time_list
        self.time_step = time_step
        self.factor_dict = factor_dict
        self.ref_file = ref_file
        self.visualizer_batch_size = visualizer_batch_size

    def construct_input(self):
        """construct input dic by baseline file"""
        time_list = self.time_list
        time_step = self.time_step
        time_tmp = time_step * time_list
        # Data reader
        reader = dataset.Reader(time_index=time_list, time_step=time_step)
        # Construct Input for prediction
        _, label = reader.vtk(
            read_input=False, filename_without_timeid=self.ref_file
        )  # using referece sampling points coordinates[t\x\y\z] as input
        one_input, _ = reader.vtk(time_point=0, filename_without_timeid=self.ref_file)
        n = len(next(iter(one_input.values())))
        self.data_len_for_onestep = n
        input = {key: np.zeros((n * len(time_tmp), 1)) for key in one_input.keys()}
        for i, time in enumerate(time_tmp):
            input[dataset.Input.t][i * n : (i + 1) * n] = np.full(
                (n, 1), int(time)
            ).astype(np.float32)
            input[dataset.Input.x][i * n : (i + 1) * n] = one_input[dataset.Input.x]
            input[dataset.Input.y][i * n : (i + 1) * n] = one_input[dataset.Input.y]
            input[dataset.Input.z][i * n : (i + 1) * n] = one_input[dataset.Input.z]
        input = dataset.normalization(input, self.factor_dict)
        onestep_xyz = {
            dataset.Input.x: one_input[dataset.Input.x],
            dataset.Input.y: one_input[dataset.Input.y],
            dataset.Input.z: one_input[dataset.Input.z],
        }

        return input, label, onestep_xyz

    def quantitive_error(self, label: Dict, solution: Dict):
        """Caculate quantitive error

        Args:
            label (Dict): reference baseline result
            solution (Dict): predicted result
        """
        # LBM baseline, output Error
        n = self.data_len_for_onestep
        err_dict = {key: [] for key in dataset.Label}
        for i in range(len(self.time_list)):
            for key in solution.keys():
                err_dict[key].append(
                    label[key][i * n : (i + 1) * n]  # n : nodes number per time step
                    - solution[key][
                        i * n : (i + 1) * n
                    ]  # n : nodes number per time step
                )
            print(
                f"{self.time_list[i]} \
                time = {self.time_step * self.time_list[i]} s, \
                sum = {(np.absolute(err_dict[dataset.Label.u][i])).sum(axis=0)}，\
                mean = {(np.absolute(err_dict[dataset.Label.u][i])).mean(axis=0)}, \
                median = {np.median(np.absolute(err_dict[dataset.Label.u][i]), axis=0)}"
            )
            # psci.visu.__save_vtk_raw(filename = dirname + f"/vtk/0302_error_{i+1}", cordinate=cord, data=temp_list)  # output error being displayed in paraview

    def save(self, dirname: str, cord: Dict, solution: Dict):
        """Save points result

        Args:
            dirname (str): Output file name with directory
            cord (Dict): points coordinates
            solution (Dict): predicted result
        """
        writer = dataset.Writer()
        n = self.data_len_for_onestep
        for i in range(len(self.time_list)):
            writer.vtk(
                filename=osp.join(dirname, f"predict_{i+1}.vtu"),
                label={
                    key.value: solution[key][i * n : (i + 1) * n]
                    for key in solution.keys()
                },  # n : nodes number per time step
                coordinates=cord,
            )
