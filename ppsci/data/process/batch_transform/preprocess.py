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

import numpy as np


class FunctionalBatchTransform:
    """Functional data transform class, which allows to use custom data transform function from given transform_func for special cases.

    Args:
        transform_func (Callable): Function of data transform.

    Examples:
        >>> # This is the transform_func function. It takes three dictionaries as input: data_dict, label_dict, and weight_dict.
        >>> # The function will perform some transformations on the data in data_dict, convert all labels in label_dict to uppercase,
        >>> # and modify the weights in weight_dict by dividing each weight by 10.
        >>> # Finally, it returns the transformed data, labels, and weights as a tuple.
        >>> import ppsci
        >>> def transform_func(data_dict, label_dict, weight_dict):
        ...     for key in data_dict:
        ...         data_dict[key] = data_dict[key] * 2
        ...     for key in label_dict:
        ...         label_dict[key] = label_dict[key] + 1.0
        ...     for key in weight_dict:
        ...         weight_dict[key] = weight_dict[key] / 10
        ...     return data_dict, label_dict, weight_dict
        >>> transform = ppsci.data.transform.FunctionalTransform(transform_func)
        >>> # Define some sample data, labels, and weights
        >>> data = {'feature1': np.array([1, 2, 3]), 'feature2': np.array([4, 5, 6])}
        >>> label = {'class': 0.0, 'instance': 0.1}
        >>> weight = {'weight1': 0.5, 'weight2': 0.5}
        >>> # Apply the transform function to the data, labels, and weights using the FunctionalTransform instance
        >>> transformed_data = transform(data, label, weight)
        >>> print(transformed_data)
        ({'feature1': array([2, 4, 6]), 'feature2': array([ 8, 10, 12])}, {'class': 1.0, 'instance': 1.1}, {'weight1': 0.05, 'weight2': 0.05})
    """

    def __init__(
        self,
        transform_func: Callable[[List[Any]], List[Any]],
    ):
        self.transform_func = transform_func

    def __call__(
        self,
        list_data: List[List[Dict[str, np.ndarray]]],
        # [{'u': arr, 'y': arr}, {'u': arr, 'y': arr}, {'u': arr, 'y': arr}], [{'s': arr}, {'s': arr}, {'s': arr}], [{}, {}, {}]
    ) -> List[Dict[str, np.ndarray]]:
        return self.transform_func(list_data)
