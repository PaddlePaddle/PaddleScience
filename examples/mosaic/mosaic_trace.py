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

"""Bounded-size layer/route traces for cross-framework Mosaic diagnostics."""

import json

import numpy as np


class Trace:
    def __init__(self, directory):
        self.directory = directory
        self.values = {}
        self.shapes = {}

    def record(self, name, tensor, full=False):
        array = tensor.detach().cpu().numpy()
        self.shapes[name] = list(array.shape)
        if not full:
            flat = array.reshape(-1)
            array = flat[
                np.linspace(0, len(flat) - 1, min(len(flat), 16384), dtype="int64")
            ]
        self.values[name] = array

    def install_torch(self, model):
        def hook_for(name):
            def hook(layer, inputs, output):
                if name == "preprocess.0":
                    self.record("preprocess_input", inputs[0], full=True)
                self.record(name, output)

            return hook

        for name, layer in model.named_modules():
            if name and (
                name.startswith(
                    (
                        "preprocess",
                        "interp_to_",
                        "downsample_layers",
                        "upsample_layers",
                        "postprocess",
                    )
                )
                or name.endswith(("norm1", "attention", "to_q", "to_k", "to_v"))
            ):
                layer.register_forward_hook(hook_for(name))
        for name, value in model.named_buffers():
            self.record("buffer." + name, value, full=True)

    def install_paddle(self, model):
        def hook_for(name):
            def hook(layer, inputs, output):
                if name == "preprocess.0":
                    self.record("preprocess_input", inputs[0], full=True)
                self.record(name, output)

            return hook

        for name, layer in model.named_sublayers():
            if name.startswith(
                (
                    "preprocess",
                    "interp_to_",
                    "downsample_layers",
                    "upsample_layers",
                    "postprocess",
                )
            ) or name.endswith(("norm1", "attention", "to_q", "to_k", "to_v")):
                layer.register_forward_post_hook(hook_for(name))
        for name, value in model.named_buffers():
            self.record("buffer." + name, value, full=True)

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez(self.directory / "trace.npz", **self.values)
        (self.directory / "shapes.json").write_text(json.dumps(self.shapes, indent=2))
