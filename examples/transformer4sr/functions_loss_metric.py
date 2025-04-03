# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://github.com/omron-sinicx/transformer4sr
"""

from typing import Dict

import numpy as np
import paddle
from utils import compute_norm_zss_dist


def cross_entropy_loss_func(output_dict, label_dict, *args):
    custom_loss = paddle.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    loss = custom_loss(output_dict["output"], label_dict["output"])
    return {"ce_loss": loss}


def compute_inaccuracy(
    output_dict: Dict[str, paddle.Tensor],
    label_dict: Dict[str, paddle.Tensor],
    *args,
) -> Dict[str, paddle.Tensor]:
    """Calculate the ratio of incorrectly matched tokens to the total number."""
    preds = output_dict["output"]
    labels = label_dict["output"]
    padding_not_mask = labels != 0
    correct_bool = paddle.equal(paddle.argmax(preds, axis=-1), labels)
    correct_bool = paddle.logical_and(
        correct_bool,
        padding_not_mask,
    )
    inacc = 1 - paddle.sum(correct_bool) / paddle.sum(padding_not_mask)
    return {"inaccuracy_mean": inacc}


def compute_zss(
    output_dict: Dict[str, paddle.Tensor],
    label_dict: Dict[str, paddle.Tensor],
    *args,
) -> Dict[str, paddle.Tensor]:
    """Calculate zss distance, which is a kind of normalized tree-based edit distance. Refer to https://arxiv.org/abs/2206.10540."""
    num_samples = output_dict["output"].shape[-1]
    preds = output_dict["output"].reshape([-1, num_samples])
    labels = label_dict["output"].reshape([-1, num_samples])
    zss_dist = []
    for i in range(labels.shape[0]):
        zss_dist.append(compute_norm_zss_dist(preds[i][0], labels[i]))
    zss_dist_mean = np.nanmean(zss_dist)
    return {
        "zss_distance": paddle.to_tensor(
            zss_dist_mean, dtype=paddle.get_default_dtype()
        )
    }
