# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from datetime import datetime

import paddle


def gaussian_perturb(x, level=0.01):
    noise = level * paddle.randn(x.shape)
    return x + noise


def load_model(model, checkpoint_file):
    checkpoint_fname = checkpoint_file
    checkpoint = paddle.load(checkpoint_fname)
    model.set_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def downsample(x, scale=0.125):
    return paddle.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear")


def date_to_hours(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
    return hours_since_jan_01_epoch
