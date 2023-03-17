"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import paddle
from pyevtk import hl

from ppsci.utils import logger


def save_vtu_from_array(filename, coord, value, value_keys, num_timestamp=1):
    """Save data to '*.vtu' file.

    Args:
        filename (str): Output filename.
        coord (np.ndarray): Coordinate points with shape of [N, 2] or [N, 3].
        value (np.ndarray): Value of each coord points with shape of [N, M].
        value_keys (List[str]): Names of each dimension of value, such as ["u", "v"].
        num_timestamp (int, optional): Number of timestamp over coord and value.
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
    if len(coord) % num_timestamp != 0:
        raise ValueError(
            f"coord length({len(coord)}) should be an integer multiple of "
            f"num_timestamp({num_timestamp})"
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
    _n = npoint // num_timestamp
    for t in range(num_timestamp):
        # NOTE: each array in data_vtu should be 1-dim, i.e. [N, 1] will occur error.
        if coord_ndim == 2:
            axis_x = np.ascontiguousarray(coord[t * _n : (t + 1) * _n, 0])
            axis_y = np.ascontiguousarray(coord[t * _n : (t + 1) * _n, 1])
            axis_z = np.zeros([_n], dtype="float32")
        elif coord_ndim == 3:
            axis_x = np.ascontiguousarray(coord[t * _n : (t + 1) * _n, 0])
            axis_y = np.ascontiguousarray(coord[t * _n : (t + 1) * _n, 1])
            axis_z = np.ascontiguousarray(coord[t * _n : (t + 1) * _n, 2])

        data_vtu = {}
        for j in range(data_ndim):
            data_vtu[value_keys[j]] = np.ascontiguousarray(
                value[t * _n : (t + 1) * _n, j]
            )

        filename_t = f"{filename}_t-{t}" if num_timestamp > 1 else filename
        hl.pointsToVTK(filename_t, axis_x, axis_y, axis_z, data=data_vtu)
    logger.info(f"Vtu results has been saved to {filename}")


def save_vtu_from_dict(filename, data_dict, coord_keys, value_keys, num_timestamp=1):
    """Save dict data to '*.vtu' file.

    Args:
        filename (str): Output filename.
        data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): Data in dict.
        coord_keys (List[str]): List of coord key. such as ["x", "y"].
        value_keys (List[str]): List of value key. such as ["u", "v"].
        ndim (int): Number of coord dimension in data_dict.
        num_timestamp (int, optional): Number of timestamp in data_dict. Defaults to 1.
    """
    if len(coord_keys) not in [2, 3, 4]:
        raise ValueError(f"ndim of coord ({len(coord_keys)}) should be 2, 3 or 4")

    coord = [data_dict[k] for k in coord_keys if k != "t"]
    value = [data_dict[k] for k in value_keys] if value_keys else None

    if isinstance(coord[0], paddle.Tensor):
        coord = [x.numpy() for x in coord]
    else:
        coord = [x for x in coord]
    coord = np.concatenate(coord, axis=1)

    if value is not None:
        if isinstance(value[0], paddle.Tensor):
            value = [x.numpy() for x in value]
        else:
            value = [x for x in value]
        value = np.concatenate(value, axis=1)

    save_vtu_from_array(filename, coord, value, value_keys, num_timestamp)
