from typing import Callable, Dict, List, Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle.io import DataLoader

from paddle_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from paddle_geometric.loader.link_loader import LinkLoader
from paddle_geometric.sampler import NegativeSampling, NeighborSampler
from paddle_geometric.sampler.base import SubgraphType
from paddle_geometric.typing import EdgeType, InputEdges, OptTensor


class LinkNeighborLoader(LinkLoader):
    r"""A link-based data loader derived as an extension of the node-based
    :class:`paddle_geometric.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        neg_sampling: Optional[NegativeSampling] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[NeighborSampler] = None,
        directed: bool = True,  # Deprecated.
        **kwargs,
    ):
        if (edge_label_time is not None) != (time_attr is not None):
            raise ValueError(
                f"Received conflicting 'edge_label_time' and 'time_attr' "
                f"arguments: 'edge_label_time' is "
                f"{'set' if edge_label_time is not None else 'not set'} "
                f"while 'time_attr' is "
                f"{'set' if time_attr is not None else 'not set'}. "
                f"Both arguments must be provided for temporal sampling.")

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
            )

        super().__init__(
            data=data,
            link_sampler=neighbor_sampler,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            neg_sampling=neg_sampling,
            neg_sampling_ratio=neg_sampling_ratio,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
