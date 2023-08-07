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

from ppsci.metric import base


class L2Rel(base.Metric):
    r"""Class for l2 relative error.

    NOTE: This metric API is different from `MeanL2Rel`, difference is as below:
    - `L2Rel` regards the input sample as a whole and calculates the L2Rel of the whole;
    - `MeanL2Rel` will calculate L2Rel separately for each input sample and return the average of L2Rel for all samples.

    $$
    metric = \dfrac{\Vert \mathbf{x} - \mathbf{y} \Vert_2}{\Vert \max(\mathbf{y}, \epsilon) \Vert_2}
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import ppsci
        >>> metric = ppsci.metric.L2Rel()
    """

    # NOTE: Avoid divide by zero in result
    # see https://github.com/scikit-learn/scikit-learn/pull/15007
    EPS: float = np.finfo(np.float32).eps

    def __init__(self, keep_batch: bool = False):
        if keep_batch:
            raise ValueError(f"keep_batch should be False, but got {keep_batch}.")
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            rel_l2 = paddle.norm(label_dict[key] - output_dict[key], p=2) / paddle.norm(
                label_dict[key], p=2
            ).clip(min=self.EPS)
            metric_dict[key] = rel_l2

        return metric_dict


class MeanL2Rel(base.Metric):
    r"""Class for mean l2 relative error.

    NOTE: This metric API is different from `L2Rel`, difference is as below:
    - `MeanL2Rel` will calculate L2Rel separately for each input sample and return the average of L2Rel for all samples.
    - `L2Rel` regards the input sample as a whole and calculates the L2Rel of the whole;

    $$
    metric = \dfrac{\Vert \mathbf{x} - \mathbf{y} \Vert_2}{\Vert \max(\mathbf{y}, \epsilon) \Vert_2}
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import ppsci
        >>> metric = ppsci.metric.MeanL2Rel()
    """

    # NOTE: Avoid divide by zero in result
    # see https://github.com/scikit-learn/scikit-learn/pull/15007
    EPS: float = np.finfo(np.float32).eps

    def __init__(self, keep_batch: bool = False):
        if keep_batch:
            raise ValueError(f"keep_batch should be False, but got {keep_batch}.")
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            mean_rel_l2 = (
                paddle.norm(label_dict[key] - output_dict[key], p=2)
                / paddle.norm(label_dict[key], p=2).clip(min=self.EPS)
            ).mean()
            metric_dict[key] = mean_rel_l2

        return metric_dict
