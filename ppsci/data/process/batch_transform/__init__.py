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
from typing import Callable
from typing import List

import numpy as np
import paddle

from ppsci.data.process import transform
from ppsci.data.process.batch_transform.preprocess import FunctionalBatchTransform

try:
    import pgl
except ModuleNotFoundError:
    pass


__all__ = [
    "build_batch_transforms",
    "default_collate_fn",
    "FunctionalBatchTransform",
    "collate_pool",
]


def default_collate_fn(batch: List[Any]) -> Any:
    """Default_collate_fn for paddle dataloader.

    NOTE: This `default_collate_fn` is different from official `default_collate_fn`
    which specially adapt case where sample is `None` and `pgl.Graph`.

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
            raise RuntimeError("Fields number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]
    elif str(type(sample)) == "<class 'pgl.graph.Graph'>":
        # use str(type()) instead of isinstance() in case of pgl is not installed.
        graph = pgl.Graph(num_nodes=sample.num_nodes, edges=sample.edges)
        graph.x = np.concatenate([g.x for g in batch])
        graph.y = np.concatenate([g.y for g in batch])
        graph.edge_index = np.concatenate([g.edge_index for g in batch], axis=1)

        graph.edge_attr = np.concatenate([g.edge_attr for g in batch])
        graph.pos = np.concatenate([g.pos for g in batch])
        if hasattr(sample, "aoa"):
            graph.aoa = np.concatenate([g.aoa for g in batch])
        if hasattr(sample, "mach_or_reynolds"):
            graph.mach_or_reynolds = np.concatenate([g.mach_or_reynolds for g in batch])
        graph.tensor()
        graph.shape = [len(batch)]
        return graph
    elif (
        str(type(sample))
        == "<class 'ppsci.data.dataset.atmospheric_dataset.GraphGridMesh'>"
    ):
        graph = sample
        graph.tensor()
        graph.shape = [1]
        return graph
    raise TypeError(
        "batch data can only contains: paddle.Tensor, numpy.ndarray, "
        f"dict, list, number, None, pgl.Graph, GraphGridMesh, but got {type(sample)}"
    )


def collate_pool(batch: List[Any]) -> Any:

    """
    Collate a list of data and return a batch for predicting crystal properties.

    Args:
        dataset_list (list): A list of tuples for each data point containing:
            - atom_fea (paddle.Tensor): Shape (n_i, atom_fea_len).
            - nbr_fea (paddle.Tensor): Shape (n_i, M, nbr_fea_len).
            - nbr_fea_idx (paddle.Tensor): Shape (n_i, M).
            - target (paddle.Tensor): Shape (1,).
            - cif_id (str or int).

    Returns:
        tuple: Contains the following:
            - batch_atom_fea (paddle.Tensor): Shape (N, orig_atom_fea_len). Atom features from atom type.
            - batch_nbr_fea (paddle.Tensor): Shape (N, M, nbr_fea_len). Bond features of each atom's M neighbors.
            - batch_nbr_fea_idx (paddle.Tensor): Shape (N, M). Indices of M neighbors of each atom.
            - crystal_atom_idx (list): List of paddle.Tensor of length N0. Mapping from the crystal idx to atom idx.
            - target (paddle.Tensor): Shape (N, 1). Target value for prediction.
            - batch_cif_ids (list): List of CIF IDs.

    Notes:
        - N = sum(n_i); N0 = sum(i)
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, item in enumerate(batch):
        input = item[0]["i"]
        label = item[1]["l"]
        id = item[2]["c"]
        atom_fea, nbr_fea, nbr_fea_idx = input
        target = label
        cif_id = id
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = paddle.to_tensor(np.arange(n_i) + int(base_idx), dtype="int64")
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    # Debugging: print shapes of the tensors to ensure they are consistent
    # print("Shapes of batch_atom_fea:", [x.shape for x in batch_atom_fea])
    # print("Shapes of batch_nbr_fea:", [x.shape for x in batch_nbr_fea])
    # print("Shapes of batch_nbr_fea_idx:", [x.shape for x in batch_nbr_fea_idx])

    # Ensure all tensors in the lists have consistent shapes before concatenation
    batch_atom_fea = paddle.concat(batch_atom_fea, axis=0)
    batch_nbr_fea = paddle.concat(batch_nbr_fea, axis=0)
    batch_nbr_fea_idx = paddle.concat(batch_nbr_fea_idx, axis=0)

    return (
        {
            "i": (
                paddle.to_tensor(batch_atom_fea, dtype="float32"),
                paddle.to_tensor(batch_nbr_fea, dtype="float32"),
                paddle.to_tensor(batch_nbr_fea_idx),
                [paddle.to_tensor(crys_idx) for crys_idx in crystal_atom_idx],
            )
        },
        {"l": paddle.to_tensor(paddle.stack(batch_target, axis=0))},
        {"c": batch_cif_ids},
    )


def build_transforms(cfg):
    if not cfg:
        return transform.Compose([])
    cfg = copy.deepcopy(cfg)

    transform_list = []
    for _item in cfg:
        transform_cls = next(iter(_item.keys()))
        transform_cfg = _item[transform_cls]
        transform_obj = eval(transform_cls)(**transform_cfg)
        transform_list.append(transform_obj)

    return transform.Compose(transform_list)


def build_batch_transforms(cfg):
    cfg = copy.deepcopy(cfg)
    batch_transforms: Callable[[List[Any]], List[Any]] = build_transforms(cfg)

    def collate_fn_batch_transforms(batch: List[Any]):
        # apply batch transform on separate samples
        batch = batch_transforms(batch)

        # then collate separate samples into batched data
        return default_collate_fn(batch)

    return collate_fn_batch_transforms
