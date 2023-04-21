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

from typing import Tuple

from ppsci.arch import base


class ModelList(base.NetBase):
    """ModelList layer which wrap more than one model that shares inputs.

    Args:
        model_list (Tuple[base.NetBase, ...]): Model(s) nesteed in tuple.
    """

    def __init__(
        self,
        model_list: Tuple[base.NetBase, ...],
    ):
        super().__init__()
        output_keys_set = set()
        for model in model_list:
            if len(output_keys_set & set(model.output_keys)):
                raise ValueError(
                    "output_keys of model from model_list should be unique,"
                    f"but got duplicate keys: {output_keys_set & set(model.output_keys)}"
                )
            output_keys_set = output_keys_set | set(model.output_keys)

        self.model_list = model_list

    def forward(self, x):
        y_all = {}
        for model in self.model_list:
            if model._input_transform is not None:
                x = model._input_transform(x)

            y = model.concat_to_tensor(x, model.input_keys, axis=-1)
            y = model.forward_tensor(y)
            y = model.split_to_dict(y, model.output_keys, axis=-1)

            if model._output_transform is not None:
                y = model._output_transform(y)
            y_all.update(y)

        return y_all
