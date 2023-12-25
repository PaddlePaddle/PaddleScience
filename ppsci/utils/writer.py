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

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle

__all__ = [
    "save_csv_file",
]


def save_csv_file(
    file_path: str,
    data_dict: Dict[str, Union[np.ndarray, paddle.Tensor]],
    keys: Tuple[str, ...],
    alias_dict: Optional[Dict[str, str]] = None,
    use_header: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> Dict[str, np.ndarray]:
    """Load *.csv file and fetch data as given keys.

    Args:
        file_path (str): CSV file path.
        keys (Tuple[str, ...]): Required fetching keys.
        alias_dict (Optional[Dict[str, str]]): Alias for keys,
            i.e. {inner_key: outer_key}. Defaults to None.
        encoding (str, optional): Encoding code when open file. Defaults to "utf-8".

    Returns:
        Dict[str, np.ndarray]: Loaded data in dict.
    """
    if alias_dict is None:
        alias_dict = {}

    # convert to numpy array
    data_array = []
    header = []
    for key in keys:
        if key not in data_dict:
            raise KeyError(f"key({key}) do not exist in raw_data.")

        dump_key = alias_dict[key] if key in alias_dict else key

        data = data_dict[dump_key]
        if isinstance(data, paddle.Tensor):
            data = data.numpy()
        if data.ndim != 2:
            data = data.reshape([-1, 1])  # [num_of_samples, 1]
        data_array.append(data)
        header.append(dump_key)

    data_array = np.concatenate(data_array, axis=1)  # [num_of_samples, num_of_fields]
    header = ",".join(header)

    np.savetxt(
        file_path,
        data_array,
        delimiter=delimiter,
        header=header if use_header else "",
        encoding=encoding,
        comments="",
    )
