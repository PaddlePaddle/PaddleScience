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


# Code is heavily based on paper "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", we use paddle to reproduce the results of the paper


import paddle


class BaseModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        """Returns the device that the model is on."""
        return paddle.device.get_device()

    def data_dict_to_input(self, data_dict, **kwargs):
        """
        Convert data dictionary to appropriate input for the model.
        """
        raise NotImplementedError

    def loss_dict(self, data_dict, **kwargs):
        """
        Compute the loss dictionary for the model.
        """
        raise NotImplementedError

    @paddle.no_grad()
    def eval_dict(self, data_dict, **kwargs):
        """
        Compute the evaluation dictionary for the model.
        """
        raise NotImplementedError
