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

from __future__ import annotations

import csv
import os
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle

from ppsci.utils import logger

__all__ = [
    "save_csv_file",
]


def save_csv_file(
    filename: str,
    data_dict: Dict[str, Union[np.ndarray, "paddle.Tensor"]],
    keys: Tuple[str, ...],
    alias_dict: Optional[Dict[str, str]] = None,
    use_header: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8",
):
    """Write numpy data to csv file.

    Args:
        filename (str): Dump file path.
        data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): Numpy data in dict.
        keys (Tuple[str, ...]): Keys for data_dict to be fetched.
        alias_dict (Optional[Dict[str, str]], optional): Alias dict for keys,
            i.e. {dump_key: dict_key}. Defaults to None.
        use_header (bool, optional): Whether save csv with header. Defaults to True.
        delimiter (str, optional): Delemiter for splitting different data field. Defaults to ",".
        encoding (str, optional): Encoding. Defaults to "utf-8".

    Examples:
        >>> import numpy as np
        >>> from ppsci.utils import save_csv_file
        >>> data_dict = {
        ...     "a": np.array([[1], [2], [3]]).astype("int64"), # [3, 1]
        ...     "b": np.array([[4.12], [5.25], [6.3370]]).astype("float32"), # [3, 1]
        ... }
        >>> save_csv_file(
        ...     "test.csv",
        ...     data_dict,
        ...     ("A", "B"),
        ...     alias_dict={"A": "a", "B": "b"},
        ...     use_header=True,
        ...     delimiter=",",
        ...     encoding="utf-8",
        ... )

        >>> # == test.csv ==
        >>> # A,B
        >>> # 1,4.12
        >>> # 2,5.25
        >>> # 3,6.337
    """

    if alias_dict is None:
        alias_dict = {}

    # convert to numpy array
    data_fields = []
    header = []
    for key in keys:
        fetch_key = alias_dict[key] if key in alias_dict else key
        data = data_dict[fetch_key]
        if isinstance(data, paddle.Tensor):
            data = data.numpy()  # [num_of_samples, ]

        data = data.flatten()
        data_fields.append(data)

        header.append(key)

    assert len(header) == len(data_fields)

    data_fields = zip(*data_fields)  # transpose col data to row data
    with open(filename, "w", newline="", encoding=encoding) as file:
        writer = csv.writer(file, delimiter=delimiter)

        if use_header:
            writer.writerow(header)

        writer.writerows(data_fields)

    logger.message(f"csv file has been dumped to {filename}")


def save_tecplot_file(
    filename: str,
    data_dict: Dict[str, Union[np.ndarray, "paddle.Tensor"]],
    keys: Tuple[str, ...],
    alias_dict: Optional[Dict[str, str]] = None,
    delimiter: str = " ",
    encoding: str = "utf-8",
    num_timestamps: int = 1,
):
    """Write numpy data to tecplot file.

    Args:
        filename (str): Tecplot file path.
        data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): Numpy or Tensor data in dict.
        keys (Tuple[str, ...]): Target keys to be dumped.
        alias_dict (Optional[Dict[str, str]], optional): Alias dict for keys,
            i.e. {dump_key: dict_key}. Defaults to None.
        delimiter (str, optional): Delemiter for splitting different data field. Defaults to ",".
        encoding (str, optional): Encoding. Defaults to "utf-8".
        num_timestamps (int, optional): Number of timestamp over coord and value. Defaults to 1.

    Examples:
        >>> import numpy as np
        >>> from ppsci.utils import save_tecplot_file
        >>> data_dict = {
        ...     "x": np.array([[1.0], [2.0], [10.0], [20.0], [100.0], [200.0]]), # [6, 1]
        ...     "y": np.array([[-1.0], [-2.0], [-10.0], [-20.0], [-100.0], [-200.0]]) # [6, 1]
        ... }
        >>> save_tecplot_file(
        ...    "./test.tec",
        ...    data_dict,
        ...    ("X", "Y"),
        ...    alias_dict={"X": "x", "Y": "y"},
        ...    num_timestamps=3,
        ... )

        >>> # == test_t-0.tec ==
        >>> # title="./test_t-0.tec"
        >>> # variables = "X", "Y"
        >>> # Zone I = 2, J = 1, F = POINT
        >>> # -1.0 1.0
        >>> # -2.0 2.0

        >>> # == test_t-1.tec ==
        >>> # title="./test_t-1.tec"
        >>> # variables = "X", "Y"
        >>> # Zone I = 2, J = 1, F = POINT
        >>> # -10.0 10.0
        >>> # -20.0 20.0

        >>> # == test_t-2.tec ==
        >>> # title="./test_t-2.tec"
        >>> # variables = "X", "Y"
        >>> # Zone I = 2, J = 1, F = POINT
        >>> # -100.0 100.0
        >>> # -200.0 200.0
    """
    ntxy = len(next(iter(data_dict.values())))
    if ntxy % num_timestamps != 0:
        raise ValueError(
            f"num_points({ntxy}) must be a multiple of "
            f"num_timestamps({num_timestamps})."
        )
    nxy = ntxy // num_timestamps

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if filename.endswith(".tec"):
        filename = filename[:-4]

    for t in range(num_timestamps):
        # write 1 tecplot file for each timestep
        if num_timestamps > 1:
            dump_filename = f"{filename}_t-{t}.tec"
        else:
            dump_filename = f"{filename}.tec"

        with open(dump_filename, "w", encoding=encoding) as f:
            # write meta information of tec
            f.write(f'title="{dump_filename}"\n')
            fetch_keys = {(alias_dict[key] if alias_dict else key) for key in keys}
            header = ", ".join([f'"{key}"' for key in keys])
            f.write(f"variables = {header}\n")
            f.write(f"Zone I = {nxy}, J = 1, F = POINT\n")

            # write points data into file
            data_cur_time_step = [
                data_dict[key][t * nxy : (t + 1) * nxy] for key in fetch_keys
            ]

            for items in zip(*data_cur_time_step):
                f.write(delimiter.join([str(float(x)) for x in items]) + "\n")

        logger.message(f"csv file has been dumped to {dump_filename}")
