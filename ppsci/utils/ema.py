# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

import itertools
from typing import Dict
from typing import Optional

import paddle
from paddle import nn

__all__ = [
    "AveragedModel",
    "ExponentialMovingAverage",
    "StochasticWeightAverage",
]


class AveragedModel(nn.Layer):
    """Base class for Averaged Model.

    Args:
        model (nn.Layer): The model to be averaged.
        decay (float): The decay rate for averaging.
    """

    def __init__(self, model: nn.Layer, decay: Optional[float] = None):
        super().__init__()
        self.model = model  # As a quick reference to online model
        self.decay = decay

        self.params_shadow: Dict[str, paddle.Tensor] = {}  # ema param or buffer
        self.params_backup: Dict[str, paddle.Tensor] = {}  # used for apply and restore
        for name, param_or_buffer in itertools.chain(
            self.model.named_parameters(), self.model.named_buffers()
        ):
            self.params_shadow[name] = param_or_buffer.clone().detach()

        self.register_buffer("n_avg", paddle.to_tensor(0, "int64"), True)

    def _update_fn_(
        self,
        shadow_param: paddle.Tensor,
        model_param: paddle.Tensor,
        step: paddle.Tensor,
    ):
        raise NotImplementedError("AveragedModel._update_fn_ should be implemented.")

    def update(self):
        for name, param_or_buffer in itertools.chain(
            self.model.named_parameters(), self.model.named_buffers()
        ):
            if not param_or_buffer.stop_gradient:
                assert (
                    name in self.params_shadow
                ), f"Parameter: {name} should be in params_shadow dict, but not found."

                # only update floating and complex data
                if paddle.is_floating_point(param_or_buffer) or paddle.is_complex(
                    param_or_buffer
                ):
                    with paddle.no_grad():
                        self._update_fn_(
                            self.params_shadow[name],
                            param_or_buffer,
                            self.n_avg,
                        )
        self.n_avg += 1

    def apply_shadow(self):
        """Set averaged model parameters to online model."""
        for name, param_or_buffer in itertools.chain(
            self.model.named_parameters(), self.model.named_buffers()
        ):
            if name in self.params_shadow:
                stop_gradient = param_or_buffer.stop_gradient
                with paddle.no_grad():
                    self.params_backup[name] = paddle.assign(param_or_buffer)
                    paddle.assign(self.params_shadow[name], param_or_buffer)
                param_or_buffer.stop_gradient = stop_gradient

    def restore(self):
        """Restore online model parameters from backup parameter dict."""
        assert self.params_backup, (
            "params_backup should not be empty, may be caused by calling 'restore' "
            "before 'apply_shadow'."
        )
        for name, param_or_buffer in itertools.chain(
            self.model.named_parameters(), self.model.named_buffers()
        ):
            if name in self.params_backup:
                assert name in self.params_shadow
                stop_gradient = param_or_buffer.stop_gradient
                with paddle.no_grad():
                    paddle.assign(self.params_backup[name], param_or_buffer)
                param_or_buffer.stop_gradient = stop_gradient

        self.params_backup = {}

    def set_state_dict(self, state_dict: Dict[str, paddle.Tensor]):
        assert (
            "n_avg" in state_dict
        ), "state_dict should contain 'n_avg' key, but not found."
        self.n_avg.set_value(state_dict.pop("n_avg"))
        self.params_shadow.update(state_dict)

    def state_dict(self) -> Dict[str, paddle.Tensor]:
        return {
            **self.params_shadow,
            "n_avg": self.n_avg,
        }


class ExponentialMovingAverage(AveragedModel):
    r"""Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

    $$
    \mathbf{\theta}_{EMA}^{t+1} = \alpha \mathbf{\theta}_{EMA}^{t} + (1 - \alpha) \mathbf{\theta}^{t}
    $$

    Where $\alpha$ is the decay rate, $\theta_{EMA}^{t}$ is the moving average parameters and $\theta^{t}$ is the online parameters at step $t$.

    Args:
        model (nn.Layer): The model to be averaged.
        decay (float): The decay rate for averaging.
    """

    def __init__(self, model: nn.Layer, decay: float = 0.9):
        super().__init__(model, decay)

    def _update_fn_(self, shadow_param, model_param, step):
        shadow_param.lerp_(model_param, 1.0 - self.decay)


class StochasticWeightAverage(AveragedModel):
    r"""Implements the stochastic weight averaging (SWA) of the model.

    Stochastic Weight Averaging was proposed in [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407),

    All parameters are updated by the formula as below:

    $$
    \mathbf{\theta}_{SWA}^{t} = \frac{1}{t-t_0+1}\sum_{i=t_0}^t{\mathbf{\theta}^{i}}
    $$

    Where $\theta_{SWA}^{t}$ is the average parameters between step $t_0$ and $t$, $\theta^{i}$ is the online parameters at step $i$.

    Args:
        model (nn.Layer): The model to be averaged.
    """

    def __init__(self, model: nn.Layer):
        super().__init__(model, None)
        self.n_avg += 1  # Set to 1 for model already initialized

    def _update_fn_(self, shadow_param, model_param, step):
        dynamic_decay = step / (step + 1)
        shadow_param.lerp_(model_param, 1.0 - dynamic_decay)
