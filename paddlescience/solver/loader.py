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
import paddle


def load_data(inputs_labels_iter, var_type):
    """_summary_

    Args:
        inputs_labels (_type_): _description_
        inputs_labels_iter (_type_): _description_
        var_type (_type_): _description_
    """
    inputs_labels = [None for _ in range(len(inputs_labels_iter))]
    if var_type == 'dict':
        for i, iter in enumerate(inputs_labels_iter):
            inputs_labels[i] = next(iter)
    return inputs_labels


def get_batch_iterator(bsize, n, input):
    """_summary_

    Args:
        bsize (_type_): _description_
        n (_type_): _description_
        input (_type_): _description_

    Returns:
        _type_: _description_
    """
    ds = EqDataSet(num_samples = n, input = input)
    bs = paddle.io.BatchSampler(dataset = ds, shuffle = False, batch_size = bsize, drop_last = False)
    loader = paddle.io.DataLoader(dataset = ds, batch_sampler = bs, num_workers=0)
    loader = InfiniteDataLoader(loader)
    return iter(loader)


class InfiniteDataLoader(object):
    """_summary_
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            dataloader = iter(self.dataloader)
            for batch in dataloader:
                yield batch


class EqDataSet(paddle.io.Dataset):
    """_summary_

    Args:
        paddle (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, num_samples, input):
        self.input = input
        self.num_samples = num_samples

    def __getitem__(self, index):
        output = self.input[index]
        return output

    def __len__(self):
        return self.num_samples  
 