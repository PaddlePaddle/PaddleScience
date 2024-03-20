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
import pytest

from ppsci.utils import reader
from ppsci.utils import writer


def test_save_csv_file():
    keys = ["x1", "y1", "z1"]
    alias_dict = (
        {
            "x": "x1",
            "y": "y1",
            "z": "z1",
        },
    )
    data_dict = {
        keys[0]: np.random.randint(0, 255, (10, 1)),
        keys[1]: np.random.rand(10, 1),
        keys[2]: np.random.rand(10, 1),
    }
    file_path = "test_writer.csv"
    writer.save_csv_file(
        file_path,
        data_dict,
        keys,
        alias_dict=alias_dict,
        use_header=True,
    )

    reload_data_dict = reader.load_csv_file(
        file_path,
        keys,
        alias_dict,
    )

    assert data_dict.keys() == reload_data_dict.keys()
    for k in reload_data_dict:
        assert reload_data_dict[k].shape == data_dict[k].shape
        assert np.allclose(reload_data_dict[k], data_dict[k])


if __name__ == "__main__":
    pytest.main()
