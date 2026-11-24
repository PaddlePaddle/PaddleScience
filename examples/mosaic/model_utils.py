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

"""Initialize the Paddle model and its static geometry."""

import numpy as np
import paddle

from ppsci.arch.mosaic import Mosaic


def initialize_model(config, static_path):
    model = Mosaic(config)
    static = np.load(static_path)
    longitude = paddle.to_tensor(static["longitude"].astype("float32"))
    latitude = paddle.to_tensor(static["latitude"].astype("float32"))
    values = paddle.to_tensor(static["data"].astype("float32"))
    model.initialize_static_vars(values, longitude, latitude)
    model.initialize_interpolation(longitude, latitude)
    return model
