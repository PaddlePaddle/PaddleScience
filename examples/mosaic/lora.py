# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Minimal Paddle Linear LoRA for Mosaic, with an explicit adapter manifest."""

import math

import paddle
from paddle import nn


class LoRALinear(nn.Layer):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = self.create_parameter(
            [base.weight.shape[0], rank],
            default_initializer=nn.initializer.Uniform(
                -1 / math.sqrt(base.weight.shape[0]),
                1 / math.sqrt(base.weight.shape[0]),
            ),
        )
        self.lora_b = self.create_parameter(
            [rank, base.weight.shape[1]],
            default_initializer=nn.initializer.Constant(0),
        )

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a @ self.lora_b) * self.scale


def inject_lora(model, rank=8, alpha=16):
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    for param in model.parameters():
        param.clear_gradient(set_to_zero=False)
        param.stop_gradient = True
    targets = []
    for name, layer in list(model.named_sublayers()):
        # Adapt attention Q/V and FFN; keep geometry, interpolation, noise,
        # norms and the official pretrained checkpoint frozen.
        if (
            isinstance(layer, nn.Linear)
            and name.rsplit(".", 1)[-1] in ("to_q", "to_v", "w13", "w2")
            and (".attention." in name or ".ffn." in name)
        ):
            parent_name, leaf = name.rsplit(".", 1)
            parent = model.get_sublayer(parent_name)
            setattr(parent, leaf, LoRALinear(layer, rank, alpha))
            targets.append(name)
    if not targets:
        raise ValueError("No Mosaic LoRA targets found")
    return dict(rank=rank, alpha=alpha, targets=targets)


def adapter_state(model):
    return {
        name: param
        for name, param in model.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }


def load_adapter(model, path):
    state = paddle.load(str(path))
    expected = adapter_state(model)
    if set(state) != set(expected):
        raise ValueError("Adapter keys do not match injected model")
    for name, param in expected.items():
        if state[name].shape != param.shape or not bool(
            paddle.isfinite(state[name]).all()
        ):
            raise ValueError(f"Invalid adapter tensor: {name}")
        param.set_value(state[name])
