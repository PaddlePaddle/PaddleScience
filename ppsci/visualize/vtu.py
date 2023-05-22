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

from typing import Dict
from typing import Tuple

import meshio
import numpy as np
import paddle
from pyevtk import hl

from ppsci.utils import logger


def _save_vtu_from_array(filename, coord, value, value_keys, num_timestamps=1):
    """Save data to '*.vtu' file(s).

    Args:
        filename (str): Output filename.
        coord (np.ndarray): Coordinate points with shape of [N, 2] or [N, 3].
        value (np.ndarray): Value of each coord points with shape of [N, M].
        value_keys (Tuple[str, ...]): Names of each dimension of value, such as ("u", "v").
        num_timestamps (int, optional): Number of timestamp over coord and value.
            Defaults to 1.
    """
    if not isinstance(coord, np.ndarray):
        raise ValueError(f"type of coord({type(coord)}) should be ndarray.")
    if value is not None and not isinstance(value, np.ndarray):
        raise ValueError(f"type of value({type(value)}) should be ndarray.")
    if value is not None and len(coord) != len(value):
        raise ValueError(
            f"coord length({len(coord)}) should be equal to value length({len(value)})"
        )
    if len(coord) % num_timestamps != 0:
        raise ValueError(
            f"coord length({len(coord)}) should be an integer multiple of "
            f"num_timestamps({num_timestamps})"
        )
    if coord.shape[1] not in [2, 3]:
        raise ValueError(f"ndim of coord({coord.shape[1]}) should be 2 or 3.")

    # discard extension name
    if filename.endswith(".vtu"):
        filename = filename[:-4]
    npoint = len(coord)
    coord_ndim = coord.shape[1]

    if value is None:
        value = np.ones([npoint, 1], dtype=coord.dtype)
        value_keys = ["dummy_key"]

    data_ndim = value.shape[1]
    nx = npoint // num_timestamps
    for t in range(num_timestamps):
        # NOTE: each array in data_vtu should be 1-dim, i.e. [N, 1] will occur error.
        if coord_ndim == 2:
            axis_x = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 0])
            axis_y = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 1])
            axis_z = np.zeros([nx], dtype=paddle.get_default_dtype())
        elif coord_ndim == 3:
            axis_x = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 0])
            axis_y = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 1])
            axis_z = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 2])

        data_vtu = {}
        for j in range(data_ndim):
            data_vtu[value_keys[j]] = np.ascontiguousarray(
                value[t * nx : (t + 1) * nx, j]
            )

        if num_timestamps > 1:
            hl.pointsToVTK(f"{filename}_t-{t}", axis_x, axis_y, axis_z, data=data_vtu)
        else:
            hl.pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtu)

    if num_timestamps > 1:
        logger.info(
            f"Visualization results are saved to {filename}_t-0.vtu ~ {filename}_t-{num_timestamps - 1}.vtu"
        )
    else:
        logger.info(f"Visualization result is saved to {filename}.vtu")


def save_vtu_from_dict(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
    num_timestamps: int = 1,
):
    """Save dict data to '*.vtu' file.

    Args:
        filename (str): Output filename.
        data_dict (Dict[str, np.ndarray]): Data in dict.
        coord_keys (Tuple[str, ...]): Tuple of coord key. such as ("x", "y").
        value_keys (Tuple[str, ...]): Tuple of value key. such as ("u", "v").
        num_timestamps (int, optional): Number of timestamp in data_dict. Defaults to 1.
    """
    if len(coord_keys) not in [2, 3, 4]:
        raise ValueError(f"ndim of coord ({len(coord_keys)}) should be 2, 3 or 4")

    coord = [data_dict[k] for k in coord_keys if k != "t"]
    value = [data_dict[k] for k in value_keys] if value_keys else None

    coord = np.concatenate(coord, axis=1)

    if value is not None:
        value = np.concatenate(value, axis=1)

    _save_vtu_from_array(filename, coord, value, value_keys, num_timestamps)


def save_vtu_to_mesh(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
):
    """Save data into .vtu format by meshio.

    Args:
        filename (str): File name.
        data_dict (Dict[str, np.ndarray]): Data in dict.
        coord_keys (Tuple[str, ...]): Tuple of coord key. such as ("x", "y").
        value_keys (Tuple[str, ...]): Tuple of value key. such as ("u", "v").
    """
    npoint = len(next(iter(data_dict.values())))
    coord_ndim = len(coord_keys)

    # get the list variable transposed
    points = np.stack((data_dict[key] for key in coord_keys)).reshape(
        coord_ndim, npoint
    )
    mesh = meshio.Mesh(
        points=points.T, cells=[("vertex", np.arange(npoint).reshape(npoint, 1))]
    )
    mesh.point_data = {key: data_dict[key] for key in value_keys}
    mesh.write(filename)
