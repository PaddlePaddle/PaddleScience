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
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from typing_extensions import Literal

from ppsci.metric import base


class RMSE(base.Metric):
    r"""Root mean square error

    $$
    metric = \sqrt{\dfrac{1}{N}\sum\limits_{i=1}^{N}{(x_i-y_i)^2}}
    $$

    Examples:
        >>> import ppsci
        >>> metric = ppsci.metric.RMSE()
    """

    def __init__(self):
        super().__init__()

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            rmse = F.mse_loss(output_dict[key], label_dict[key], "mean") ** 0.5
            metric_dict[key] = float(rmse)

        return metric_dict


class LatitudeWeightedRMSE(base.Metric):
    r"""Latitude weighted root mean square error.

    $$
    metric =\sqrt{\dfrac{1}{MN}\sum\limits_{m=1}^{M}\sum\limits_{n=1}^{N}L(m)(X[m,n]-Y[m,n])^{2}}
    $$

    $$
    L(m) = N_{lat}\dfrac{cos(lat(m))}{\sum\limits_{j=1}^{N_{lat}}cos(lat(j))}
    $$

    $lat(m)$ is the latitude at m.
    $N_{lat}$ is the number of latitude set by `num_lat`.

    Args:
        num_lat (int): Number of latitude.
        std (Union[np.array, Tuple[float, ...]]): Standard Deviation of training dataset.
        reduction (Literal[mean, none], optional): Reduction method. Defaults to "mean".
        variable_dict (Dict[str, int], optional): Variable dictionary. Defaults to None.
        unlog (bool, optional): whether calculate expm1 for all elements in the array. Defaults to False.
        scale (float, optional): The scale value used after expm1. Defaults to 1e-5.

    Examples:
        >>> import numpy as np
        >>> import ppsci
        >>> std = np.random.randn(20, 1, 1)
        >>> metric = ppsci.metric.LatitudeWeightedRMSE(720, std=std)
    """

    def __init__(
        self,
        num_lat: int,
        std: Union[np.array, Tuple[float, ...]],
        reduction: Literal["mean", "none"] = "mean",
        variable_dict: Dict[str, int] = None,
        unlog: bool = False,
        scale: float = 1e-5,
    ):
        super().__init__()
        self.num_lat = num_lat
        self.std = paddle.to_tensor(std).reshape((1, -1))
        self.reduction = reduction
        self.variable_dict = variable_dict
        self.unlog = unlog
        self.scale = scale

        self.weight = self.get_latitude_weight(num_lat)

    def expm1_data(self, x: paddle.Tensor):
        return self.scale * paddle.expm1(x)

    def get_latitude_weight(self, num_lat: int = 720):
        lat_t = paddle.linspace(start=0, stop=1, num=num_lat)
        lat_t = paddle.cos(3.1416 * (0.5 - lat_t))
        weight = num_lat * lat_t / paddle.sum(lat_t)
        weight = weight.reshape((1, 1, -1, 1))
        return weight

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            output = (
                self.expm1_data(output_dict[key])
                if self.unlog is True
                else output_dict[key]
            )
            label = (
                self.expm1_data(label_dict[key])
                if self.unlog is True
                else label_dict[key]
            )

            mse = F.mse_loss(output, label, "none")
            rmse = (mse * self.weight).mean(axis=(-1, -2)) ** 0.5
            rmse = rmse * self.std
            if self.variable_dict is not None:
                for variable_name, idx in self.variable_dict.items():
                    if self.reduction == "mean":
                        metric_dict[f"{key}.{variable_name}"] = float(
                            rmse[:, idx].mean()
                        )
                    elif self.reduction == "none":
                        metric_dict[f"{key}.{variable_name}"] = rmse[:, idx]
            else:
                if self.reduction == "mean":
                    rmse = rmse.mean()
                    metric_dict[key] = float(rmse)
                elif self.reduction == "none":
                    rmse = rmse.mean(axis=1)
                    metric_dict[key] = rmse

        return metric_dict
