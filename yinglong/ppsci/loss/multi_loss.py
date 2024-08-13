from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from typing_extensions import Literal

from ppsci.loss import base


class MultiLoss(nn.Layer):
    r"""Latitude weighted anomaly correlation coefficient.

    $$
    metric =
        \dfrac{\sum\limits_{m,n}{L_mX_{mn}Y_{mn}}}{\sqrt{\sum\limits_{m,n}{L_mX_{mn}^{2}}\sum\limits_{m,n}{L_mY_{mn}^{2}}}}
    $$

    $$
    L_m = N_{lat}\dfrac{\cos(lat_m)}{\sum\limits_{j=1}^{N_{lat}}\cos(lat_j)}
    $$

    $lat_m$ is the latitude at m.
    $N_{lat}$ is the number of latitude set by `num_lat`.

    Args:
        num_lat (Optional[int]): Number of latitude for compute weight, if is None, no weight applied.
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

    def __init__(self, loss_list):
        # weight = self.get_latitude_weight(num_lat) if num_lat is not None else None
        super().__init__()
        self.loss_list = loss_list

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0
        for loss_fun in self.loss_list:
            losses += loss_fun(output_dict, label_dict, weight_dict)
        return losses
