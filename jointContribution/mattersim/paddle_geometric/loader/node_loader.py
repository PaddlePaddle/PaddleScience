from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from paddle_geometric.loader.base import DataLoaderIterator
from paddle_geometric.loader.mixin import (
    AffinityMixin,
    LogMemoryMixin,
    MultithreadingMixin,
)
from paddle_geometric.loader.utils import (
    filter_custom_hetero_store,
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    infer_filter_per_worker,
)
from paddle_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from paddle_geometric.typing import InputNodes, OptTensor


class NodeLoader(
        paddle.io.DataLoader,
        AffinityMixin,
        MultithreadingMixin,
        LogMemoryMixin,
):
    r"""A data loader that performs mini-batch sampling from node information,
    using a generic :class:`~paddle_geometric.sampler.BaseSampler`
    implementation that defines a
    :meth:`~paddle_geometric.sampler.BaseSampler.sample_from_nodes` function and
    is supported on the provided input :obj:`data` object.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        node_sampler: BaseSampler,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[HeteroData] = None,
        input_id: OptTensor = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        self.data = data
        self.node_sampler = node_sampler
        self.input_nodes = input_nodes
        self.input_time = input_time
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.input_id = input_id

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # Get node type (or `None` for homogeneous graphs):
        input_type, input_nodes, input_id = get_input_nodes(
            data, input_nodes, input_id)

        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        iterator = range(input_nodes.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        input_data: NodeSamplerInput = self.input_data[index]

        out = self.node_sampler.sample_from_nodes(input_data)

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        return out

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~paddle_geometric.data.Data` or
        :class:`~paddle_geometric.data.HeteroData` object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            if isinstance(self.data, Data):
                data = filter_data(  #
                    self.data, out.node, out.row, out.col, out.edge,
                    self.node_sampler.edge_permutation)

            else:  # Tuple[FeatureStore, GraphStore]
                # Hack to detect whether we are in a distributed setting.
                if (self.node_sampler.__class__.__name__ ==
                        'DistNeighborSampler'):
                    edge_index = paddle.stack([out.row, out.col])
                    data = Data(edge_index=edge_index)
                    # Metadata entries are populated in
                    # `DistributedNeighborSampler._collate_fn()`
                    data.x = out.metadata[-3]
                    data.y = out.metadata[-2]
                    data.edge_attr = out.metadata[-1]
                else:
                    data = filter_custom_store(  #
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls)

            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                edge = out.edge.to(paddle.int64)
                perm = self.node_sampler.edge_permutation
                data.e_id = perm[edge] if perm is not None else edge

            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            if out.orig_row is not None and out.orig_col is not None:
                data._orig_edge_index = paddle.stack([
                    out.orig_row,
                    out.orig_col,
                ], axis=0)

            data.input_id = out.metadata[0]
            data.seed_time = out.metadata[1]
            data.batch_size = out.metadata[0].size(0)

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(  #
                    self.data, out.node, out.row, out.col, out.edge,
                    self.node_sampler.edge_permutation)

            else:  # Tuple[FeatureStore, GraphStore]
                # Hack to detect whether we are in a distributed setting.
                if (self.node_sampler.__class__.__name__ ==
                        'DistNeighborSampler'):
                    import paddle_geometric.distributed as dist

                    data = dist.utils.filter_dist_store(
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls, out.metadata,
                        self.input_data.input_type)
                else:
                    data = filter_custom_hetero_store(  #
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls)

            for key, node in out.node.items():
                if 'n_id' not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if edge is not None and 'e_id' not in data[key]:
                    edge = edge.to(paddle.int64)
                    perm = self.node_sampler.edge_permutation
                    if perm is not None and perm.get(key, None) is not None:
                        edge = perm[key][edge]
                    data[key].e_id = edge

            data.set_value_dict('batch', out.batch)
            data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
            data.set_value_dict('num_sampled_edges', out.num_sampled_edges)

            if out.orig_row is not None and out.orig_col is not None:
                for key in out.orig_row.keys():
                    data[key]._orig_edge_index = paddle.stack([
                        out.orig_row[key],
                        out.orig_col[key],
                    ], axis=0)

            input_type = self.input_data.input_type
            data[input_type].input_id = out.metadata[0]
            data[input_type].seed_time = out.metadata[1]
            data[input_type].batch_size = out.metadata[0].size(0)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
