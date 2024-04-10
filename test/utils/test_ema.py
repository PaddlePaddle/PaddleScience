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

import numpy as np
import paddle
import pytest

import ppsci
from ppsci.utils import ema


def test_ema_accumulation():
    model = ppsci.arch.MLP(
        ("x", "y", "z"),
        ("u", "v", "w"),
        4,
        25,
    )
    decay = 0.9
    avg_model = ema.ExponentialMovingAverage(model, decay)

    opt = ppsci.optimizer.Adam()(model)

    model_params_hist = {
        k: [
            p.clone().detach(),
        ]
        for k, p in model.named_parameters()
    }
    N = 32
    T = 5
    for i in range(T):
        input_data = {
            "x": paddle.randn([N, 1]),
            "y": paddle.randn([N, 1]),
            "z": paddle.randn([N, 1]),
        }
        label_data = {
            "u": paddle.randn([N, 1]),
            "v": paddle.randn([N, 1]),
            "w": paddle.randn([N, 1]),
        }
        output_data = model(input_data)
        loss = sum(
            [
                paddle.nn.functional.mse_loss(output, label)
                for output, label in zip(output_data.values(), label_data.values())
            ]
        )
        loss.backward()
        opt.step()
        opt.clear_grad()
        avg_model.update()

        for k, p in model.named_parameters():
            model_params_hist[k].append(p.clone().detach())

    for k, plist in model_params_hist.items():
        ema_p = model_params_hist[k][0]
        for p in plist[1:]:
            ema_p = ema_p * decay + p * (1 - decay)

        np.testing.assert_allclose(ema_p, avg_model.params_shadow[k], 1e-7, 1e-7)


def test_ema_apply_restore():
    model = ppsci.arch.MLP(
        ("x", "y", "z"),
        ("u", "v", "w"),
        4,
        25,
    )
    model.linears[-1].weight.stop_gradient = True
    model.linears[-2].bias.stop_gradient = True
    decay = 0.9
    avg_model = ema.ExponentialMovingAverage(model, decay)

    opt = ppsci.optimizer.Adam()(model)

    N = 32
    T = 5
    for i in range(T):
        input_data = {
            "x": paddle.randn([N, 1]),
            "y": paddle.randn([N, 1]),
            "z": paddle.randn([N, 1]),
        }
        label_data = {
            "u": paddle.randn([N, 1]),
            "v": paddle.randn([N, 1]),
            "w": paddle.randn([N, 1]),
        }
        output_data = model(input_data)
        loss = sum(
            [
                paddle.nn.functional.mse_loss(output, label)
                for output, label in zip(output_data.values(), label_data.values())
            ]
        )
        loss.backward()
        opt.step()
        opt.clear_grad()
        avg_model.update()

    orignal_param = {k: v.clone() for k, v in model.named_parameters()}

    # test if stop_gradient are excluded
    assert model.linears[-1].weight.name not in avg_model.params_shadow
    assert model.linears[-2].bias.name not in avg_model.params_shadow

    # test if model paramter == backup
    avg_model.apply_shadow()
    for k in orignal_param:
        if not orignal_param[k].stop_gradient:
            np.testing.assert_allclose(
                avg_model.params_backup[k], orignal_param[k], 1e-7, 1e-7
            )
        assert model.state_dict()[k].stop_gradient == orignal_param[k].stop_gradient

    # test if restored successfully
    avg_model.restore()
    for k in orignal_param:
        np.testing.assert_allclose(model.state_dict()[k], orignal_param[k], 1e-7, 1e-7)
        assert model.state_dict()[k].stop_gradient == orignal_param[k].stop_gradient
    assert len(avg_model.params_backup) == 0


if __name__ == "__main__":
    pytest.main()