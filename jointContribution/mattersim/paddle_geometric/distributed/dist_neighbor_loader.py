from typing import Callable, Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.multiprocessing as mp

from paddle_geometric.distributed import (
    DistContext,
    DistLoader,
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
)
from paddle_geometric.loader import NodeLoader
from paddle_geometric.sampler.base import SubgraphType
from paddle_geometric.typing import EdgeType, InputNodes, OptTensor


class DistNeighborLoader(NodeLoader, DistLoader):
    r"""A distributed loader that performs sampling from nodes.

    Args:
        data (tuple): A (:class:`~paddle_geometric.data.FeatureStore`,
            :class:`~paddle_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
            The number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary for each
            individual edge type.
        master_addr (str): RPC address for distributed loader communication.
        master_port (Union[int, str]): Open port for RPC communication.
        current_ctx (DistContext): Distributed context information.
        concurrency (int, optional): RPC concurrency to define the maximum
            asynchronous queue size. (default: :obj:`1`)

    All other arguments follow the interface of
    :class:`paddle_geometric.loader.NeighborLoader`.
    """
    def __init__(
        self,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        master_addr: str,
        master_port: Union[int, str],
        current_ctx: DistContext,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        dist_sampler: Optional[DistNeighborSampler] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        concurrency: int = 1,
        num_rpc_threads: int = 16,
        filter_per_worker: Optional[bool] = False,
        async_sampling: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        assert isinstance(data[0], LocalFeatureStore)
        assert isinstance(data[1], LocalGraphStore)
        assert concurrency >= 1, "RPC concurrency must be greater than 1"

        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        channel = mp.Queue() if async_sampling else None

        if dist_sampler is None:
            dist_sampler = DistNeighborSampler(
                data=data,
                current_ctx=current_ctx,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                device=device,
                channel=channel,
                concurrency=concurrency,
            )

        DistLoader.__init__(
            self,
            channel=channel,
            master_addr=master_addr,
            master_port=master_port,
            current_ctx=current_ctx,
            dist_sampler=dist_sampler,
            num_rpc_threads=num_rpc_threads,
            **kwargs,
        )
        NodeLoader.__init__(
            self,
            data=data,
            node_sampler=dist_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            filter_per_worker=filter_per_worker,
            transform_sampler_output=self.channel_get if channel else None,
            worker_init_fn=self.worker_init_fn,
            **kwargs,
        )

    def __repr__(self) -> str:
        return DistLoader.__repr__(self)
