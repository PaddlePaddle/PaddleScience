from typing import Callable, Dict, List, Optional, Tuple, Union

import paddle
from paddle_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from paddle_geometric.loader.node_loader import NodeLoader
from paddle_geometric.sampler import NeighborSampler
from paddle_geometric.sampler.base import SubgraphType
from paddle_geometric.typing import EdgeType, InputNodes, OptTensor


class NeighborLoader(NodeLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~paddle_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    Args:
        data (Any): A :class:`~paddle_geometric.data.Data`,
            :class:`~paddle_geometric.data.HeteroData`, or
            (:class:`~paddle_geometric.data.FeatureStore`,
            :class:`~paddle_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[EdgeType, List[int]]): The
            number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
        input_nodes (paddle.Tensor or str or Tuple[str, paddle.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`paddle.LongTensor` or
            :obj:`paddle.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        input_time (paddle.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        subgraph_type (SubgraphType or str, optional): The type of the returned
            subgraph.
            If set to :obj:`"directional"`, the returned subgraph only holds
            the sampled (directed) edges which are necessary to compute
            representations for the sampled seed nodes.
            If set to :obj:`"bidirectional"`, sampled edges are converted to
            bidirectional edges.
            If set to :obj:`"induced"`, the returned subgraph contains the
            induced subgraph of all sampled nodes.
            (default: :obj:`"directional"`)
        disjoint (bool, optional): If set to :obj: `True`, each seed node will
            create its own disjoint subgraph.
            If set to :obj:`True`, mini-batch outputs will have a :obj:`batch`
            vector holding the mapping of nodes to their respective subgraph.
            Will get automatically set to :obj:`True` in case of temporal
            sampling. (default: :obj:`False`)
        temporal_strategy (str, optional): The sampling strategy when using
            temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            If set to :obj:`"uniform"`, will sample uniformly across neighbors
            that fulfill temporal constraints.
            If set to :obj:`"last"`, will sample the last `num_neighbors` that
            fulfill temporal constraints.
            (default: :obj:`"uniform"`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for either the nodes or edges in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier or equal timestamp than the center node.
            (default: :obj:`None`)
        weight_attr (str, optional): The name of the attribute that denotes
            edge weights in the graph.
            If set, weighted/biased sampling will be used such that neighbors
            are more likely to get sampled the higher their edge weights are.
            Edge weights do not need to sum to one, but must be non-negative,
            finite and have a non-zero sum within local neighborhoods.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`paddle_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column.
            If :obj:`time_attr` is set, additionally requires that rows are
            sorted according to time within individual neighborhoods.
            This avoids internal re-sorting of the data and can improve
            runtime and memory efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returned data in each worker's subprocess.
            If set to :obj:`False`, will filter the returned data in the main
            process.
            If set to :obj:`None`, will automatically infer the decision based
            on whether data partially lives on the GPU
            (:obj:`filter_per_worker=True`) or entirely on the CPU
            (:obj:`filter_per_worker=False`).
            There exists different trade-offs for setting this option.
            Specifically, setting this option to :obj:`True` for in-memory
            datasets will move all features to shared memory, which may result
            in too many open file handles. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`paddle.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """

    def __init__(
            self,
            data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
            num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
            input_nodes: InputNodes = None,
            input_time: OptTensor = None,
            replace: bool = False,
            subgraph_type: Union[SubgraphType, str] = 'directional',
            disjoint: bool = False,
            temporal_strategy: str = 'uniform',
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
        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

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
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
