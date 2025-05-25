import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple, Union

import paddle
from paddle.io import DataLoader

from paddle_geometric.data import FeatureStore, GraphStore, HeteroData
from paddle_geometric.sampler import HGTSampler
from paddle_geometric.typing import NodeType
from paddle_geometric.data import Data

class HGTLoader(DataLoader):
    r"""The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
    Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    :class:`~paddle_geometric.data.HGTLoader` tries to (1) keep a similar
    number of nodes and edges for each type and (2) keep the sampled sub-graph
    dense to minimize the information loss and reduce the sample variance.

    Methodically, :class:`~paddle_geometric.data.HGTLoader` keeps track of a
    node budget for each node type, which is then used to determine the
    sampling probability of a node.
    In particular, the probability of sampling a node is determined by the
    number of connections to already sampled nodes and their node degrees.
    With this, :class:`~paddle_geometric.data.HGTLoader` will sample a fixed
    amount of neighbors for each node type in each iteration, as given by the
    :obj:`num_samples` argument.
    """
    def __init__(
        self,
        data: Union[HeteroData, Tuple[FeatureStore, GraphStore]],
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        input_nodes: Union[NodeType, Tuple[NodeType, Optional[paddle.Tensor]]],
        is_sorted: bool = False,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        **kwargs,
    ):
        hgt_sampler = HGTSampler(
            data,
            num_samples=num_samples,
            is_sorted=is_sorted,
            share_memory=kwargs.get('num_workers', 0) > 0,
        )

        super().__init__(
            dataset=data,
            batch_sampler=hgt_sampler,
            input_nodes=input_nodes,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

    def __iter__(self):
        """Iterates over the dataset and yields batches of sampled data."""
        # Implement the logic for batch-wise iteration
        # Adapt from the sampler or data object, yielding mini-batches
        pass

    def _collate(self, data_list):
        """Handles batching of heterogeneous data objects."""
        # Similar to the PyTorch implementation but adapted for Paddle
        batch = Data()
        for key in data_list[0].keys:
            batch[key] = paddle.stack([data[key] for data in data_list])
        return batch
