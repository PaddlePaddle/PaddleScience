# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""This file provide the dataloader iterator"""
import paddle


def load_data(inputs_labels_iter, var_type):
    """for each sort of input, specify the corresponding iterator and fetch data

    Args:
        inputs_labels_iter (_type_): _description_
        var_type (_type_): _description_

    Returns:
        _type_: _description_
    """
    inputs_labels = [None for _ in range(len(inputs_labels_iter))]
    if var_type == "dict":
        for i, input_iter in enumerate(inputs_labels_iter):
            inputs_labels[i] = next(input_iter)
    return inputs_labels


def get_batch_iterator(bsize, num_samples, input_list):
    """for each sort of input, create the corresponding iterator

    Args:
        bsize (int): _description_
        n (int): _description_
        input (list): _description_

    Returns:
        _type_: _description_
    """
    dataset = EqDataSet(num_samples=num_samples, input_list=input_list)
    batch_sampler = paddle.io.BatchSampler(
        dataset=dataset, shuffle=False, batch_size=bsize, drop_last=False
    )
    loader = paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, num_workers=2
    )
    loader = InfiniteDataLoader(loader)
    return iter(loader)


class InfiniteDataLoader:
    """_summary_"""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            dataloader = iter(self.dataloader)
            for batch in dataloader:
                yield batch

    def method_one(self):
        """_summary_"""


class EqDataSet(paddle.io.Dataset):
    """_summary_

    Args:
        paddle (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, num_samples, input_list):
        self.input_list = input_list
        self.num_samples = num_samples
        super().__init__()

    def __getitem__(self, index):
        output = self.input_list[index]
        return output

    def __len__(self):
        return self.num_samples
