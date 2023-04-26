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

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np

from ppsci.visualize import base
from ppsci.visualize import plot
from ppsci.visualize import vtu


class VisualizerScatter1D(base.Visualizer):
    """Visualizer for 1d scatter data.

    Examples:
        ``` python
        >>> visualizer = {
        ...     "visulzie_u": ppsci.visualize.VisualizerScatter1D(
        ...         visu_mat,
        ...         ("t_f",),
        ...         {
        ...             "eta": lambda d: d["eta"],  # plot with latex title
        ...             "eta_gt": lambda d: d["eta_gt"],  # plot with latex title
        ...         },
        ...         num_timestamps=1,
        ...         prefix="viv_pred",
        ...     )
        ... }
        ```
    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        coord_keys (Tuple[str, ...]): Coordinate keys, such as ("x", "y").
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int, optional): Number of timestamps. Defaults to 1.
        prefix (str, optional): Prefix for output file. Defaults to "plot".
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        coord_keys: Tuple[str, ...],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        prefix: str = "plot",
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)
        self.coord_keys = coord_keys

    def save(self, filename, data_dict):
        plot.save_plot_from_1d_dict(
            filename, data_dict, self.coord_keys, self.output_keys, self.num_timestamps
        )


class VisualizerScatter3D(base.Visualizer):
    """Visualizer for 3d scatter data.

    Examples:
        ``` python
        >>> visualizer = {
        ...     "visulzie_states": ppsci.visualize.VisualizerScatter3D(
        ...         vis_datas,
        ...         {
        ...             "pred_states": lambda d: output_transform(d),
        ...             "states": lambda d: d["states"],
        ...         },
        ...         num_timestamps=1,
        ...         prefix="result_states",
        ...     )
        ... }
        ```
    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int): Number of timestamps. Defaults to 1.
        prefix (str): Prefix for output file. Defaults to "plot3d_scatter".
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        prefix: str = "plot3d_scatter",
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)

    def save(self, filename, data_dict):
        data_dict = {
            key: value for key, value in data_dict.items() if key in self.output_keys
        }
        value = data_dict[self.output_keys[0]]
        dim = len(value.shape)
        if dim == 3:
            # value.shape=(B, T, 3)
            for i in range(value.shape[0]):
                cur_data_dict = {key: value[i] for key, value in data_dict.items()}
                plot.save_plot_from_3d_dict(
                    filename + str(i),
                    cur_data_dict,
                    self.output_keys,
                    self.num_timestamps,
                )
        else:
            # value.shape=(T, 3)
            plot.save_plot_from_3d_dict(
                filename, data_dict, self.output_keys, self.num_timestamps
            )


class VisualizerVtu(base.Visualizer):
    """Visualizer for 2D points data.

    Examples:
        ``` python
        >>> visualizer = {
        ...     "visulzie_u_v": ppsci.visualize.VisualizerVtu(
        ...         vis_points,
        ...         {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
        ...         num_timestamps=ntime_all,
        ...         prefix="result_u_v",
        ...     )
        ... }
        ```
    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int): Number of timestamps
        prefix (str): Prefix for output file.
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        prefix: str = "vtu",
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)

    def save(self, filename, data_dict):
        vtu.save_vtu_from_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )


class Visualizer2D(base.Visualizer):
    """Visualizer for 2D data.

    Examples:
        ``` python
        >>> visualizer = {
        ...     "visulzie_u_v": ppsci.visualize.VisualizerVtu(
        ...         vis_points,
        ...         {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
        ...         num_timestamps=ntime_all,
        ...         prefix="result_u_v",
        ...     )
        ... }
        ```
    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int): Number of timestamps. Defaults to 1.
        prefix (str): Prefix for output file. Defaults to "plot2d".
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        prefix: str = "plot2d",
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)


class Visualizer2DPlot(Visualizer2D):
    """Visualizer for 2D data use matplotlib.

    Examples:
        ``` python
        >>> visulzie_states = ppsci.visualize.Visualizer2DPlot(
        ...     vis_datas,
        ...     {
        ...         "target_ux": lambda d: d["states"][:, :, 0],
        ...         "pred_ux": lambda d: output_transform(d)[:, :, 0],
        ...         "target_uy": lambda d: d["states"][:, :, 1],
        ...         "pred_uy": lambda d: output_transform(d)[:, :, 1],
        ...         "target_p": lambda d: d["states"][:, :, 2],
        ...         "preds_p": lambda d: output_transform(d)[:, :, 2],
        ...     },
        ...     batch_size=1,
        ...     num_timestamps=10,
        ...     stride=20,
        ...     xticks=np.linspace(-2, 14, 9),
        ...     yticks=np.linspace(-4, 4, 5),
        ...     prefix="result_states",
        ... )
        ```
    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int): Number of timestamps.
        stride (int, optional): The time stride of visualization. Defaults to 1.
        xticks (Optional[Tuple[float,...]]): The list of xtick locations. Defaults to None.
        yticks (Optional[Tuple[float,...]]): The list of ytick locations. Defaults to None.
        prefix (str): Prefix for output file. Defaults to "plot2d".
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        stride: int = 1,
        xticks: Optional[Tuple[float, ...]] = None,
        yticks: Optional[Tuple[float, ...]] = None,
        prefix: str = "plot2d",
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)
        self.stride = stride
        self.xticks = xticks
        self.yticks = yticks

    def save(self, filename, data_dict):
        data_dict = {
            key: value for key, value in data_dict.items() if key in self.output_keys
        }
        value = data_dict[self.output_keys[0]]
        dim = len(value.shape)
        if dim == 4:
            # value.shape=(B, T, H, W)
            for i in range(value.shape[0]):
                cur_data_dict = {key: value[i] for key, value in data_dict.items()}
                plot.save_plot_from_2d_dict(
                    filename + str(i),
                    cur_data_dict,
                    self.output_keys,
                    self.num_timestamps,
                    self.stride,
                    self.xticks,
                    self.yticks,
                )
        else:
            # value.shape=(T, H, W)
            plot.save_plot_from_2d_dict(
                filename,
                data_dict,
                self.output_keys,
                self.num_timestamps,
                self.stride,
                self.xticks,
                self.yticks,
            )


class Visualizer3D(base.Visualizer):
    """Visualizer for 3D plot data.

    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int): Number of timestamps
        prefix (str): Prefix for output file.
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        prefix: str = "plot3d",
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)

    def save(self, filename, data_dict):
        vtu.save_vtu_from_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )
