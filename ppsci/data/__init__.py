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
import random
from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial
from typing import Any
from typing import List

import numpy as np
import paddle
import paddle.device as device
import paddle.distributed as dist
import paddle.io as io
from paddle.fluid import core

from ppsci.data import dataloader
from ppsci.data import dataset
from ppsci.data import process
from ppsci.data.process import batch_transform
from ppsci.utils import logger

__all__ = ["dataset", "process", "dataloader", "build_dataloader"]


def _default_collate_fn_allow_none(batch: List[Any]) -> Any:
    """Modified collate function to allow some fields to be None, such as weight field.

    ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dataloader/collate.py#L24

    Args:
        batch (List[Any]): Batch of samples to be collated.

    Returns:
        Any: Collated batch data.
    """
    sample = batch[0]

    # allow field to be None
    if sample is None:
        return None

    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, (paddle.Tensor, core.eager.Tensor)):
        return paddle.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: default_collate_fn_allow_none([d[key] for d in batch])
            for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError("fileds number not same among samples in a batch")
        return [default_collate_fn_allow_none(fields) for fields in zip(*batch)]

    raise TypeError(
        "batch data can only contains: tensor, numpy.ndarray, "
        f"dict, list, number, None, but got {type(sample)}"
    )


def worker_init_fn(worker_id, num_workers, rank, base_seed):
    """Callback function on each worker subprocess after seeding and before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1]
        num_workers (int): Number of subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in non-distributed environment, it is a constant number `0`.
        seed (int): Random seed
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + base_seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(_dataset, cfg):
    world_size = dist.get_world_size()
    # just return IterableDataset as datalaoder
    if isinstance(_dataset, io.IterableDataset):
        if world_size > 1:
            raise ValueError(
                f"world_size({world_size}) should be 1 when using IterableDataset"
            )
        return _dataset

    cfg = copy.deepcopy(cfg)

    # build sampler
    sampler_cfg = cfg.pop("sampler")
    sampler_cls = sampler_cfg.pop("name")
    if sampler_cls == "BatchSampler":
        if world_size > 1:
            sampler_cls = "DistributedBatchSampler"
            logger.warning(
                f"Automatically use 'DistributedBatchSampler' instead of "
                f"'BatchSampler' when world_size({world_size}) > 1"
            )

    sampler_cfg["batch_size"] = cfg["batch_size"]
    sampler = getattr(io, sampler_cls)(_dataset, **sampler_cfg)

    # build collate_fn if specified
    batch_transforms_cfg = cfg.pop("batch_transforms", None)

    if isinstance(batch_transforms_cfg, dict) and batch_transforms_cfg:
        collate_fn = batch_transform.build_batch_transforms(batch_transforms_cfg)
    else:
        collate_fn = _default_collate_fn_allow_none

    # build init function
    init_fn = partial(
        worker_init_fn,
        num_workers=cfg.get("num_workers", 0),
        rank=dist.get_rank(),
        base_seed=cfg.get("seed", 42),
    )

    # build dataloader
    dataloader = io.DataLoader(
        dataset=_dataset,
        places=device.get_device(),
        return_list=True,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 0),
        use_shared_memory=cfg.get("use_shared_memory", False),
        worker_init_fn=init_fn,
    )

    return dataloader
