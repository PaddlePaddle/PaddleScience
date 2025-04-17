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


class BaseDataModule:
    @property
    def train_dataset(self) -> paddle.io.Dataset:
        raise NotImplementedError

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> paddle.io.Dataset:
        raise NotImplementedError

    def train_dataloader(self, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return paddle.io.DataLoader(self.train_data, collate_fn=collate_fn, **kwargs)

    def val_dataloader(self, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return paddle.io.DataLoader(self.val_data, collate_fn=collate_fn, **kwargs)

    def test_dataloader(self, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return paddle.io.DataLoader(self.test_data, collate_fn=collate_fn, **kwargs)
