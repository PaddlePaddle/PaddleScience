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

import paddle

from ppsci.arch import UCast


def test_ucast_forward_and_backward():
    model = UCast(
        input_keys=("input", "dynamical_condition", "static_condition"),
        output_keys=("output",),
        in_channels=7,
        out_channels=3,
        model_channels=8,
        channel_mult=(1, 2),
        num_blocks=1,
        attn_levels=(1,),
        channels_per_head=4,
        dropout=0.1,
        image_resolution=8,
    )
    inputs = {
        "input": paddle.randn([2, 3, 8, 5]),
        "dynamical_condition": paddle.randn([2, 2, 8, 5]),
        "static_condition": paddle.randn([2, 2, 8, 5]),
    }
    output = model(inputs)["output"]
    assert output.shape == [2, 3, 8, 5]
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_ucast_mc_dropout_mode():
    model = UCast(
        input_keys=("input",),
        output_keys=("output",),
        in_channels=3,
        out_channels=2,
        model_channels=8,
        channel_mult=(1,),
        num_blocks=1,
        dropout=0.1,
        image_resolution=8,
    )
    model.enable_mc_dropout()
    assert not model.training
    assert any(
        isinstance(layer, paddle.nn.Dropout) and layer.training
        for layer in model.sublayers()
    )
