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

import copy
import numbers
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import List

import numpy as np
import paddle

try:
    import pgl
except ModuleNotFoundError:
    pass

from ppsci.data.process import transform

__all__ = ["build_batch_transforms", "default_collate_fn"]


def default_collate_fn(batch: List[Any]) -> Any:
    """Default_collate_fn for paddle dataloader.

    ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/io/dataloader/collate.py#L25

    Args:
        batch (List[Any]): Batch of samples to be collated.

    Returns:
        Any: Collated batch data.
    """
    sample = batch[0]
    if sample is None:
        return None
    elif isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, (paddle.Tensor, paddle.framework.core.eager.Tensor)):
        return paddle.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {key: default_collate_fn([d[key] for d in batch]) for key in sample}
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError("fileds number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]
    elif str(type(sample)) == "<class 'pgl.graph.Graph'>":
        # use str(type()) instead of isinstance() in case of pgl is not installed.
        graph = pgl.Graph(num_nodes=sample.num_nodes, edges=sample.edges)
        graph.x = paddle.concat([g.x for g in batch])
        graph.y = paddle.concat([g.y for g in batch])
        graph.edge_index = paddle.concat([g.edge_index for g in batch], axis=1)
        graph.edge_attr = paddle.concat([g.edge_attr for g in batch])
        graph.pos = paddle.concat([g.pos for g in batch])
        graph.shape = [len(batch)]
        return graph

    raise TypeError(
        "batch data can only contains: Tensor, numpy.ndarray, "
        f"dict, list, number, None, pgl.Graph, but got {type(sample)}"
    )


def build_batch_transforms(cfg):
    cfg = copy.deepcopy(cfg)
    batch_transforms = transform.build_transforms(cfg)

    def collate_fn_batch_transforms(batch: List[Any]):
        # apply batch transform on uncollated data
        batch = batch_transforms(batch)
        # then do collate
        return default_collate_fn(batch)

    return collate_fn_batch_transforms
