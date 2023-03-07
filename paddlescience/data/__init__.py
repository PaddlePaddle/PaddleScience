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
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist

from .data_process import save_data, load_data
from .trphysx_dataset import LorenzDataset, CylinderDataset, RosslerDataset


def build_dataloader(dataset_name,
                     batch_size,
                     shuffle,
                     drop_last,
                     num_workers=8,
                     dataset_args=dict()):
    """
    Build the dataloader according to arguments 
    dataset_name - name of the dataset 
    batch_size - batch size, 
    shuffle - is shuffle data
    drop_last -  """

    assert dataset_name in [
        'LorenzDataset', 'CylinderDataset', 'RosslerDataset'
    ]
    dataset = eval(dataset_name)(**dataset_args)

    if dist.get_world_size() > 1:
        # Distribute data to multiple cards
        batch_sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    else:
        # Distribute data to single card
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    data_loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)

    return data_loader
