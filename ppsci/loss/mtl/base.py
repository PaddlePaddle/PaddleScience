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
from typing import ClassVar
from typing import Dict
from typing import Union

from paddle import nn

if TYPE_CHECKING:
    import paddle


class LossAggregator(nn.Layer):
    """Base class of loss aggregator mainly for multitask learning.

    Attributes:
        should_persist(bool): Whether to persist the loss aggregator when saving.
            Those loss aggregators with parameters and/or buffers should be persisted.

    Args:
        model (nn.Layer): Training model.
    """

    should_persist: ClassVar[bool] = False

    def __init__(self, model: nn.Layer) -> None:
        super().__init__()
        self.model = model
        self.step = 0
        self.param_num = 0
        for param in self.model.parameters():
            if not param.stop_gradient:
                self.param_num += 1

    def forward(
        self, losses: Dict[str, "paddle.Tensor"], step: int = 0
    ) -> Union["paddle.Tensor", "LossAggregator"]:
        self.losses = losses
        self.loss_num = len(losses)
        self.step = step
        return self

    def backward(self) -> None:
        raise NotImplementedError(
            f"'backward' should be implemented in subclass {self.__class__.__name__}"
        )

    def state_dict(self):
        agg_state = super().state_dict()
        model_state = self.model.state_dict()
        # remove model parameters from state dict for already in pdparams
        agg_state = {k: v for k, v in agg_state.items() if k not in model_state}
        return agg_state
