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

import copy
from typing import Callable
from typing import Optional

import paddle
from paddle import nn

__all__ = [
    "EMA",
]


class EMA(nn.Layer):
    """Class for exponential moving average.

    Args:
        model (nn.Layer): Model used in moving average.
        momentum (float, optional): Momentum of moving average. Defaults to 0.9.
        update_lambda (Optional[Callable[[paddle.Tensor, paddle.Tensor, int], paddle.Tensor]], optional):
            Custom update function, the three args corresponds to: (ema_param, model_param, ema_step). Defaults to None.
            Other moving average algorithm can be implemented by update_lambda, e.g. SWA.
    """

    def __init__(
        self,
        model: nn.Layer,
        momentum: float = 0.9,
        update_lambda: Optional[
            Callable[["paddle.Tensor", "paddle.Tensor", int], "paddle.Tensor"]
        ] = None,
    ):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        # freeze parameters for ema model
        for param in self.ema_model.parameters():
            param.stop_gradient = True

        self.step = 0
        self.momentum = momentum
        self.update_lambda = update_lambda

    def update(self, model: nn.Layer):
        """Update ema model using given model

        Args:
            model (nn.Layer): Online model.
        """
        m_ = self.momentum
        with paddle.no_grad():
            if isinstance(model, (paddle.DataParallel,)):
                msd, esd = model.module.state_dict(), self.ema_model.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema_model.state_dict()

            for k, v in esd.items():
                if not paddle.is_floating_point(v):
                    if self.update_lambda:
                        esd[k] = self.update_lambda(v, msd[k], self.step)
                    else:
                        esd[k] = v * m_ + (1.0 - m_) * msd[k].detach()

        self.step += 1
