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

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


class FunctionalBatchTransform:
    """Functional data transform class, which allows to use custom data transform function from given transform_func for special cases.

    Args:
        transform_func (Callable): Function of batch data transform.

    Examples:
        >>> import ppsci
        >>> from typing import Tuple, Dict, Optional
        >>> def batch_transform_func(
        ...     data_list: List[
        ...         Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]
        ...     ],
        ... ) -> List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]]:
        ...     input_dicts, label_dicts, weight_dicts = zip(*data_list)
        ...
        ...     for input_dict in input_dicts:
        ...         for key in input_dict:
        ...             input_dict[key] = input_dict[key] * 2
        ...
        ...     for label_dict in label_dicts:
        ...         for key in label_dict:
        ...             label_dict[key] = label_dict[key] + 1.0
        ...
        ...     return list(zip(input_dicts, label_dicts, weight_dicts))
        ...
        >>> # Create a FunctionalBatchTransform object with the batch_transform_func function
        >>> transform = ppsci.data.batch_transform.FunctionalBatchTransform(batch_transform_func)
        >>> # Define some sample data, labels, and weights
        >>> data = [({'x': 1}, {'y': 2}, None), ({'x': 11}, {'y': 22}, None)]
        >>> transformed_data = transform(data)
        >>> for tuple in transformed_data:
        ...     print(tuple)
        ({'x': 2}, {'y': 3.0}, None)
        ({'x': 22}, {'y': 23.0}, None)
    """

    def __init__(
        self,
        transform_func: Callable[[List[Any]], List[Any]],
    ):
        self.transform_func = transform_func

    def __call__(
        self,
        data_list: List[Tuple[Optional[Dict[str, np.ndarray]], ...]],
    ) -> List[Tuple[Optional[Dict[str, np.ndarray]], ...]]:
        return self.transform_func(data_list)
