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

import numpy as np
import paddle
from matplotlib import pyplot as plt

from ppsci.utils import logger

cnames = [
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
]


def _save_plot_from_1d_array(filename, coord, value, value_keys, num_timestamp=1):
    """Save plot from given 1D data.

    Args:
        filename (str): Filename.
        coord (np.ndarray): Coordinate array.
        value (Dict[str, np.ndarray]): Dict of value array.
        value_keys (List[str]): Value keys.
        num_timestamp (int, optional): Number of timestamps coord/value contains. Defaults to 1.
    """
    fig, a = plt.subplots(len(value_keys), num_timestamp, squeeze=False)

    len_ts = len(coord) // num_timestamp
    for t in range(num_timestamp):
        st = t * len_ts
        ed = (t + 1) * len_ts
        coord_t = coord[st:ed]

        for i, key in enumerate(value_keys):
            _value_t: np.ndarray = value[st:ed, i]
            a[i][t].scatter(
                coord_t,
                _value_t,
                color=cnames[i],
                label=key,
            )
            a[i][t].set_title(f"{key}(t={t})")
            a[i][t].grid()
            a[i][t].legend()

        if num_timestamp == 1:
            fig.savefig(filename, dpi=300)
        else:
            fig.savefig(f"{filename}_{t}", dpi=300)

    if num_timestamp == 1:
        logger.info(f"1D result is saved to {filename}.png")
    else:
        logger.info(
            f"1D result is saved to {filename}_0.png"
            f" ~ {filename}_{num_timestamp - 1}.png"
        )


def save_plot_from_1d_dict(
    filename, data_dict, coord_keys, value_keys, num_timestamp=1
):
    """Plot dict data as file.

    Args:
        filename (str): Output filename.
        data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): Data in dict.
        coord_keys (List[str]): List of coord key. such as ["x", "y"].
        value_keys (List[str]): List of value key. such as ["u", "v"].
        num_timestamp (int, optional): Number of timestamp in data_dict. Defaults to 1.
    """
    space_ndim = len(coord_keys) - int("t" in coord_keys)
    if space_ndim not in [1, 2, 3]:
        raise ValueError(f"ndim of space coord ({space_ndim}) should be 1, 2 or 3")

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

    _save_plot_from_1d_array(filename, coord, value, value_keys, num_timestamp)
