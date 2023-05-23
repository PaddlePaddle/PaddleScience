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
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle

from ppsci.metric import base


class LatitudeWeightedACC(base.Metric):
    r"""Latitude weighted anomaly correlation coefficient.

    $$
    metric =
        \dfrac{\sum\limits_{m,n}{L_mX_{mn}Y_{mn}}}{\sqrt{\sum\limits_{m,n}{L_mX_{mn}^{2}}\sum\limits_{m,n}{L_mY_{mn}^{2}}}}
    $$

    $$
    L_m = N_{lat}\dfrac{cos(lat_m)}{\sum\limits_{j=1}^{N_{lat}}cos(lat_j)}
    $$

    $lat_m$ is the latitude at m.
    $N_{lat}$ is the number of latitude set by `num_lat`.

    Args:
        num_lat (int): Number of latitude.
        mean (Optional[Union[np.array, Tuple[float, ...]]]): Mean of training data. Defaults to None.
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.
        variable_dict (Optional[Dict[str, int]]): Variable dictionary, the key is the name of a variable and
            the value is its index. Defaults to None.
        unlog (bool, optional): whether calculate expm1 for all elements in the array. Defaults to False.
        scale (float, optional): The scale value used after expm1. Defaults to 1e-5.

    Examples:
        >>> import numpy as np
        >>> import ppsci
        >>> mean = np.random.randn(20, 720, 1440)
        >>> metric = ppsci.metric.LatitudeWeightedACC(720, mean=mean)
    """

    def __init__(
        self,
        num_lat: int,
        mean: Optional[Union[np.array, Tuple[float, ...]]],
        keep_batch: bool = False,
        variable_dict: Optional[Dict[str, int]] = None,
        unlog: bool = False,
        scale: float = 1e-5,
    ):
        super().__init__(keep_batch)
        self.num_lat = num_lat
        self.mean = (
            None if mean is None else paddle.to_tensor(mean, paddle.get_default_dtype())
        )
        self.variable_dict = variable_dict
        self.unlog = unlog
        self.scale = scale

        self.weight = self.get_latitude_weight(num_lat)

    def get_latitude_weight(self, num_lat: int = 720):
        lat_t = paddle.linspace(start=0, stop=1, num=num_lat)
        lat_t = paddle.cos(3.1416 * (0.5 - lat_t))
        weight = num_lat * lat_t / paddle.sum(lat_t)
        weight = weight.reshape((1, 1, -1, 1))
        return weight

    def scale_expm1(self, x: paddle.Tensor):
        return self.scale * paddle.expm1(x)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            output = (
                self.scale_expm1(output_dict[key]) if self.unlog else output_dict[key]
            )
            label = self.scale_expm1(label_dict[key]) if self.unlog else label_dict[key]

            if self.mean is not None:
                output = output - self.mean
                label = label - self.mean

            rmse = paddle.sum(
                self.weight * output * label, axis=(-1, -2)
            ) / paddle.sqrt(
                paddle.sum(self.weight * output**2, axis=(-1, -2))
                * paddle.sum(self.weight * label**2, axis=(-1, -2))
            )

            if self.variable_dict is not None:
                for variable_name, idx in self.variable_dict.items():
                    if self.keep_batch:
                        metric_dict[f"{key}.{variable_name}"] = rmse[:, idx]
                    else:
                        metric_dict[f"{key}.{variable_name}"] = rmse[:, idx].mean()
            else:
                if self.keep_batch:
                    metric_dict[key] = rmse.mean(axis=1)
                else:
                    metric_dict[key] = rmse.mean()
        return metric_dict
