from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from typing_extensions import Literal

from ppsci.loss import base


class ACCLoss(base.Loss):
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

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
        mean: Optional[Union[np.array, Tuple[float, ...]]] = None,
        thresh=0.5,
    ):
        # weight = self.get_latitude_weight(num_lat) if num_lat is not None else None
        super().__init__(reduction, weight)

        self.mean = (
            None if mean is None else paddle.to_tensor(mean, paddle.get_default_dtype())
        )
        self.thresh = thresh

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0
        for key in label_dict:
            output = output_dict[key]
            label = label_dict[key]

            if self.mean is not None:
                output = output - self.mean
                label = label - self.mean

            if weight_dict is not None:
                weight = weight_dict[key]
                loss = paddle.sum(weight * output * label, axis=(-1, -2)) / paddle.sqrt(
                    paddle.sum(weight * output**2, axis=(-1, -2))
                    * paddle.sum(weight * label**2, axis=(-1, -2))
                )
            else:
                loss = paddle.sum(output * label, axis=(-1, -2)) / paddle.sqrt(
                    paddle.sum(output**2, axis=(-1, -2))
                    * paddle.sum(label**2, axis=(-1, -2))
                )
            loss = paddle.clip(loss, self.thresh)
            loss = -paddle.log(loss)
            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
            if isinstance(self.weight, float):
                loss *= self.weight
            elif isinstance(self.weight, dict) and key in self.weight:
                loss *= self.weight[key]
            losses += loss
        print("acc loss", losses)
        return losses
