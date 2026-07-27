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

from typing import Union

from paddle import io


class InfiniteDataLoader:
    """A wrapper for infinite dataloader.

    Args:
        dataloader (Union[io.DataLoader, io.IterableDataset]): A finite and iterable loader or iterable dataset to be wrapped.
    """

    def __init__(self, dataloader: Union[io.DataLoader, io.IterableDataset]):
        self.dataloader = dataloader
        if isinstance(dataloader, io.DataLoader):
            self.dataset = dataloader.dataset
        elif isinstance(dataloader, io.IterableDataset):
            self.dataset = dataloader
        else:
            raise TypeError(
                f"dataloader should be io.DataLoader or io.IterableDataset, but got {type(dataloader)}"
            )

    def __iter__(self):
        while True:
            dataloader_iter = iter(self.dataloader)
            for batch in dataloader_iter:
                yield batch

    def __len__(self):
        return len(self.dataloader)
