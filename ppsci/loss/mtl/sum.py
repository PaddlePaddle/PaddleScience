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

from typing import TYPE_CHECKING
from typing import Sequence

if TYPE_CHECKING:
    import paddle

from ppsci.loss.mtl.base import LossAggregator


class Sum(LossAggregator):
    r"""
    **Default loss aggregator** which do simple summation for given losses as below.

    $$
    loss = \sum_i^N losses_i
    $$
    """

    def __init__(self) -> None:
        self.step = 0

    def __call__(
        self, losses: Sequence["paddle.Tensor"], step: int = 0
    ) -> paddle.Tensor:
        assert (
            len(losses) > 0
        ), f"Number of given losses({len(losses)}) can not be empty."
        self.step = step

        loss = losses[0]
        for i in range(1, len(losses)):
            loss += losses[i]

        return loss
