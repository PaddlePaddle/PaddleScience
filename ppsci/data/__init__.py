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
import random
from functools import partial

import numpy as np
import paddle.distributed as dist
from paddle import device
from paddle import io

from ppsci.data import dataloader
from ppsci.data import dataset
from ppsci.data import process
from ppsci.data.process import batch_transform
from ppsci.data.process import transform
from ppsci.utils import logger

__all__ = [
    "dataset",
    "process",
    "dataloader",
    "build_dataloader",
    "transform",
    "batch_transform",
]


def worker_init_fn(worker_id: int, num_workers: int, rank: int, base_seed: int) -> None:
    """Callback function on each worker subprocess after seeding and before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1].
        num_workers (int): Number of subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in non-distributed
            environment, it is a constant number `0`.
        base_seed (int): Base random seed.
    """
    # The seed of each worker equals to: user_seed + num_worker * rank + worker_id
    worker_seed = base_seed + num_workers * rank + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(_dataset, cfg):
    world_size = dist.get_world_size()
    # just return IterableDataset as dataloader
    if isinstance(_dataset, io.IterableDataset):
        if world_size > 1:
            raise ValueError(
                f"world_size({world_size}) should be 1 when using IterableDataset."
            )
        return _dataset

    cfg = copy.deepcopy(cfg)

    # build sampler
    sampler_cfg = cfg.pop("sampler", None)
    if sampler_cfg is not None:
        sampler_cls = sampler_cfg.pop("name")

        if sampler_cls == "BatchSampler":
            if world_size > 1:
                sampler_cls = "DistributedBatchSampler"
                logger.warning(
                    f"Automatically use 'DistributedBatchSampler' instead of "
                    f"'BatchSampler' when world_size({world_size}) > 1."
                )

        sampler_cfg["batch_size"] = cfg["batch_size"]
        sampler = getattr(io, sampler_cls)(_dataset, **sampler_cfg)
    else:
        if cfg["batch_size"] != 1:
            raise ValueError(
                f"`batch_size` should be 1 when sampler config is None, but got {cfg['batch_size']}."
            )
        logger.warning(
            "`batch_size` is set to 1 as neither sampler config nor batch_size is set."
        )
        sampler = None

    # build collate_fn if specified
    batch_transforms_cfg = cfg.pop("batch_transforms", None)

    collate_fn = None
    if isinstance(batch_transforms_cfg, (list, tuple)):
        collate_fn = batch_transform.build_batch_transforms(batch_transforms_cfg)

    # build init function
    init_fn = partial(
        worker_init_fn,
        num_workers=cfg.get("num_workers", 0),
        rank=dist.get_rank(),
        base_seed=cfg.get("seed", 42),
    )

    # build dataloader
    if getattr(_dataset, "use_pgl", False):
        # Use special dataloader from "Paddle Graph Learning" toolkit.
        try:
            from pgl.utils import data as pgl_data
        except ModuleNotFoundError as e:
            logger.error("Please install pgl with `pip install pgl`.")
            raise ModuleNotFoundError(str(e))

        if collate_fn is None:
            collate_fn = batch_transform.default_collate_fn
        dataloader_ = pgl_data.Dataloader(
            dataset=_dataset,
            batch_size=cfg["batch_size"],
            drop_last=sampler_cfg.get("drop_last", False),
            shuffle=sampler_cfg.get("shuffle", False),
            num_workers=cfg.get("num_workers", 1),
            collate_fn=collate_fn,
        )
    else:
        dataloader_ = io.DataLoader(
            dataset=_dataset,
            places=device.get_device(),
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 1),
            use_shared_memory=cfg.get("use_shared_memory", False),
            worker_init_fn=init_fn,
        )

    if len(dataloader_) == 0:
        raise ValueError(
            f"batch_size({sampler_cfg['batch_size']}) should not bigger than number of "
            f"samples({len(_dataset)}) when drop_last is {sampler_cfg.get('drop_last', False)}."
        )

    return dataloader_
