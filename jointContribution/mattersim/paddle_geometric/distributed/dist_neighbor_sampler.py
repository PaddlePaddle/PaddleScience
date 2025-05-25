import itertools
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor

from paddle_geometric.distributed import (
    DistContext,
    LocalFeatureStore,
    LocalGraphStore,
)
from paddle_geometric.distributed.event_loop import (
    ConcurrentEventLoop,
    to_asyncio_future,
)
from paddle_geometric.distributed.rpc import (
    RPCCallBase,
    RPCRouter,
    rpc_async,
    rpc_partition_to_workers,
    rpc_register,
)
from paddle_geometric.distributed.utils import (
    BatchDict,
    DistEdgeHeteroSamplerInput,
    NodeDict,
    remove_duplicates,
)
from paddle_geometric.sampler import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from paddle_geometric.sampler.base import NumNeighbors, SubgraphType
from paddle_geometric.sampler.neighbor_sampler import neg_sample
from paddle_geometric.sampler.utils import remap_keys
from paddle_geometric.typing import EdgeType, NodeType

NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


class RPCSamplingCallee(RPCCallBase):
    r"""A wrapper for RPC callee that will perform RPC sampling from remote
    processes.
    """
    def __init__(self, sampler: NeighborSampler):
        super().__init__()
        self.sampler = sampler

    def rpc_async(self, *args, **kwargs) -> Any:
        return self.sampler._sample_one_hop(*args, **kwargs)

    def rpc_sync(self, *args, **kwargs) -> Any:
        pass


class DistNeighborSampler:
    r"""An implementation of a distributed and asynchronous neighbor sampler
    used by :class:`~paddle_geometric.distributed.DistNeighborLoader` and
    :class:`~paddle_geometric.distributed.DistLinkNeighborLoader`.
    """
    def __init__(
        self,
        current_ctx: DistContext,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        num_neighbors: NumNeighborsType,
        channel: Optional[dist.Queue] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        concurrency: int = 1,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.current_ctx = current_ctx

        self.feature_store, self.graph_store = data
        assert isinstance(self.graph_store, LocalGraphStore)
        assert isinstance(self.feature_store, LocalFeatureStore)
        self.is_hetero = self.graph_store.meta['is_hetero']

        self.num_neighbors = num_neighbors
        self.channel = channel
        self.concurrency = concurrency
        self.device = device
        self.event_loop = None
        self.replace = replace
        self.subgraph_type = SubgraphType(subgraph_type)
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.time_attr = time_attr
        self.temporal = time_attr is not None
        self.with_edge_attr = self.feature_store.has_edge_attr()
        self.csc = True

    def init_sampler_instance(self):
        self._sampler = NeighborSampler(
            data=(self.feature_store, self.graph_store),
            num_neighbors=self.num_neighbors,
            subgraph_type=self.subgraph_type,
            replace=self.replace,
            disjoint=self.disjoint,
            temporal_strategy=self.temporal_strategy,
            time_attr=self.time_attr,
        )

        self.num_hops = self._sampler.num_neighbors.num_hops
        self.node_types = self._sampler.node_types
        self.edge_types = self._sampler.edge_types
        self.node_time = self._sampler.node_time
        self.edge_time = self._sampler.edge_time

    def register_sampler_rpc(self) -> None:
        partition2workers = rpc_partition_to_workers(
            current_ctx=self.current_ctx,
            num_partitions=self.graph_store.num_partitions,
            current_partition_idx=self.graph_store.partition_idx,
        )
        self.rpc_router = RPCRouter(partition2workers)
        self.feature_store.set_rpc_router(self.rpc_router)

        rpc_sample_callee = RPCSamplingCallee(self)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

    def init_event_loop(self) -> None:
        if self.event_loop is None:
            self.event_loop = ConcurrentEventLoop(self.concurrency)
            self.event_loop.start_loop()
            logging.info(f'{self} uses {self.event_loop}')

    # Node-based distributed sampling #########################################

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        self.init_event_loop()

        inputs = NodeSamplerInput.cast(inputs)
        if self.channel is None:
            # synchronous sampling
            return self.event_loop.run_task(
                coro=self._sample_from(self.node_sample, inputs))

        # asynchronous sampling
        cb = kwargs.get("callback", None)
        self.event_loop.add_task(
            coro=self._sample_from(self.node_sample, inputs), callback=cb)
        return None

    # Edge-based distributed sampling #########################################

    def sample_from_edges(
        self,
        inputs: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        self.init_event_loop()

        if self.channel is None:
            # synchronous sampling
            return self.event_loop.run_task(coro=self._sample_from(
                self.edge_sample, inputs, self.node_sample, self._sampler.
                num_nodes, self.disjoint, self.node_time, neg_sampling))

        # asynchronous sampling
        cb = kwargs.get("callback", None)
        self.event_loop.add_task(
            coro=self._sample_from(self.edge_sample, inputs, self.node_sample,
                                   self._sampler.num_nodes, self.disjoint,
                                   self.node_time, neg_sampling), callback=cb)
        return None

    async def _sample_from(
        self,
        async_func,
        *args,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        sampler_output = await async_func(*args, **kwargs)

        if self.subgraph_type == SubgraphType.bidirectional:
            sampler_output = sampler_output.to_bidirectional()

        res = await self._collate_fn(sampler_output)

        if self.channel is None:
            return res
        self.channel.put(res)
        return None

    async def node_sample(
            self,
            inputs: Union[NodeSamplerInput, HeteroSamplerOutput],
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """
        Performs layer-by-layer distributed sampling from a
        :class:`NodeSamplerInput` or :class:`HeteroSamplerOutput` and
        returns the output of the sampling procedure.

        .. note::
            In case of distributed training it is required to synchronize the
            results between machines after each layer.
        """
        input_type = inputs.input_type
        self.input_type = input_type

        if isinstance(inputs, NodeSamplerInput):
            seed = paddle.to_tensor(inputs.node, place=self.device)
            batch_size = len(inputs.node)
            seed_batch = paddle.arange(batch_size) if self.disjoint else None

            metadata = (inputs.input_id, inputs.time, batch_size)

            seed_time: Optional[Tensor] = None
            if self.temporal:
                if inputs.time is not None:
                    seed_time = paddle.to_tensor(inputs.time, place=self.device)
                elif self.node_time is not None:
                    if not self.is_hetero:
                        seed_time = self.node_time[seed]
                    else:
                        seed_time = self.node_time[input_type][seed]
                else:
                    raise ValueError("Seed time needs to be specified")
        else:  # `HeteroSamplerOutput`
            metadata = None  # Metadata is added during `edge_sample`.

        # Heterogeneous Neighborhood Sampling #################################

        if self.is_hetero:
            if input_type is None:
                raise ValueError("Input type should be defined")

            node_dict = {node_type: [] for node_type in self.node_types}
            batch_dict = {node_type: [] for node_type in self.node_types}

            if isinstance(inputs, NodeSamplerInput):
                seed_dict: Dict[NodeType, Tensor] = {input_type: seed}
                if self.temporal:
                    node_dict[input_type].append(seed_time)

            edge_dict: Dict[EdgeType, Tensor] = {
                k: paddle.zeros([0], dtype='int64')
                for k in self.edge_types
            }
            sampled_nbrs_per_node_dict: Dict[EdgeType, List[List]] = {
                k: [[] for _ in range(self.num_hops)]
                for k in self.edge_types
            }
            num_sampled_edges_dict: Dict[EdgeType, List[int]] = {
                k: []
                for k in self.edge_types
            }
            num_sampled_nodes_dict: Dict[NodeType, List[int]] = {
                k: [0]
                for k in self.node_types
            }

            # Fill in node_dict and batch_dict with input data:
            batch_len = 0
            for k, v in seed_dict.items():
                node_dict[k] = v
                num_sampled_nodes_dict[k][0] = len(v)

                if self.disjoint:
                    src_batch = paddle.arange(batch_len, batch_len + len(v))
                    batch_dict[k] = src_batch

                    batch_len = len(src_batch)

            # Loop over the layers:
            for i in range(self.num_hops):
                # Sample neighbors per edge type:
                for edge_type in self.edge_types:
                    src = edge_type[0] if not self.csc else edge_type[2]

                    if len(node_dict[src]) == 0:
                        # No source nodes of this type in the current layer.
                        num_sampled_edges_dict[edge_type].append(0)
                        continue

                    one_hop_num = self.num_neighbors[i]

                    # Sample neighbors:
                    out = await self.sample_one_hop(
                        node_dict[src],
                        one_hop_num,
                        node_dict.get(src, []),
                        batch_dict.get(src, []),
                        edge_type,
                    )

                    if len(out.node) == 0:  # No neighbors were sampled.
                        num_sampled_edges_dict[edge_type].append(0)
                        continue

                    dst = edge_type[2] if not self.csc else edge_type[0]

                    # Update dictionaries:
                    node_dict[dst].extend(out.node)
                    edge_dict[edge_type] = paddle.concat(
                        [edge_dict[edge_type], out.edge]
                    )

                    if self.temporal and i < self.num_hops - 1:
                        src_seed_time = paddle.concat(
                            [seed_time[(seed_batch == batch_idx)]
                             for batch_idx in src_batch])

                        node_dict[src].append(src_seed_time)

            # Create local edge indices for a batch:
            row_dict, col_dict = paddle.ops.pyg.hetero_relabel_neighborhood(
                self.node_types,
                self.edge_types,
                seed_dict,
                node_dict,
                sampled_nbrs_per_node_dict,
                self._sampler.num_nodes,
                batch_dict,
                self.csc,
                self.disjoint,
            )

            sampler_output = HeteroSamplerOutput(
                node=node_dict,
                row=row_dict,
                col=col_dict,
                edge=edge_dict,
                batch=batch_dict if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes_dict,
                num_sampled_edges=num_sampled_edges_dict,
                metadata=metadata,
            )

        # Homogeneous Neighborhood Sampling ###################################

        else:
            src = seed
            node = src.clone()

            src_batch = seed_batch.clone() if self.disjoint else None
            batch = seed_batch.clone() if self.disjoint else None

            src_seed_time = seed_time.clone() if self.temporal else None

            node_with_dupl = [paddle.zeros([0], dtype='int64')]
            batch_with_dupl = [paddle.zeros([0], dtype='int64')]
            edge = [paddle.zeros([0], dtype='int64')]

            sampled_nbrs_per_node = []
            num_sampled_nodes = [len(seed)]
            num_sampled_edges = []

            for i, one_hop_num in enumerate(self.num_neighbors):
                out = await self.sample_one_hop(src, one_hop_num,
                                                src_seed_time, src_batch)
                if len(out.node) == 0:
                    num_sampled_nodes += [0] * (self.num_hops - i)
                    num_sampled_edges += [0] * (self.num_hops - i)
                    break

                src, node, src_batch, batch = self.remove_duplicates(
                    out, node, batch, self.disjoint)

                node_with_dupl.append(out.node)
                edge.append(out.edge)

                if self.disjoint:
                    batch_with_dupl.append(out.batch)

                num_sampled_nodes.append(len(src))
                num_sampled_edges.append(len(out.node))
                sampled_nbrs_per_node += out.metadata[0]

            row, col = paddle.ops.pyg.relabel_neighborhood(
                seed,
                paddle.concat(node_with_dupl),
                sampled_nbrs_per_node,
                self._sampler.num_nodes,
                paddle.concat(batch_with_dupl) if self.disjoint else None,
                self.csc,
                self.disjoint,
            )

            sampler_output = SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=paddle.concat(edge),
                batch=batch if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
                metadata=metadata,
            )

        return sampler_output

    async def edge_sample(
            self,
            inputs: EdgeSamplerInput,
            sample_fn: Callable,
            num_nodes: Union[int, Dict[NodeType, int]],
            disjoint: bool,
            node_time: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
            neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """
        Performs layer-by-layer distributed sampling from an
        EdgeSamplerInput and returns the output of the sampling
        procedure.

        Note:
            In case of distributed training, it is required to synchronize the
            results between machines after each layer.
        """
        input_id = inputs.input_id
        src = inputs.row
        dst = inputs.col
        edge_label = inputs.label
        edge_label_time = inputs.time
        input_type = inputs.input_type

        src_time = dst_time = edge_label_time
        assert edge_label_time is None or disjoint

        assert isinstance(num_nodes, (dict, int))
        if not isinstance(num_nodes, dict):
            num_src_nodes = num_dst_nodes = num_nodes
        else:
            num_src_nodes = num_nodes[input_type[0]]
            num_dst_nodes = num_nodes[input_type[-1]]

        num_pos = src.shape[0]
        num_neg = 0

        # Negative Sampling ###################################################

        if neg_sampling is not None:
            num_neg = math.ceil(num_pos * neg_sampling.amount)

            if neg_sampling.is_binary():
                # Binary case: Randomly sample negative pairs of nodes
                src_neg = self.neg_sample(src, neg_sampling, num_src_nodes, src_time)
                src = paddle.concat([src, src_neg], axis=0)

                dst_neg = self.neg_sample(dst, neg_sampling, num_dst_nodes, dst_time)
                dst = paddle.concat([dst, dst_neg], axis=0)

                if edge_label is None:
                    edge_label = paddle.ones([num_pos], dtype='float32')
                edge_neg_label = paddle.zeros([num_neg], dtype=edge_label.dtype)
                edge_label = paddle.concat([edge_label, edge_neg_label], axis=0)

                if edge_label_time is not None:
                    src_time = dst_time = paddle.concat(
                        [edge_label_time] * (1 + math.ceil(neg_sampling.amount)),
                        axis=0)[:num_pos + num_neg]

            elif neg_sampling.is_triplet():
                dst_neg = self.neg_sample(dst, neg_sampling, num_dst_nodes, dst_time)
                dst = paddle.concat([dst, dst_neg], axis=0)
                assert edge_label is None

                if edge_label_time is not None:
                    dst_time = paddle.concat(
                        [edge_label_time] * (1 + neg_sampling.amount),
                        axis=0)

        # Heterogeneous Neighborhood Sampling ##################################

        if input_type is not None:
            if input_type[0] != input_type[-1]:  # Two distinct node types:
                seed_dict = {input_type[0]: src, input_type[-1]: dst}

                seed_time_dict = None
                if edge_label_time is not None:  # Always disjoint.
                    seed_time_dict = {
                        input_type[0]: src_time,
                        input_type[-1]: dst_time,
                    }

                out = await sample_fn(
                    DistEdgeHeteroSamplerInput(
                        input_id=inputs.input_id,
                        node_dict=seed_dict,
                        time_dict=seed_time_dict,
                        input_type=input_type,
                    )
                )

            else:  # Only a single node type: Merge both source and destination.
                seed = paddle.concat([src, dst], axis=0)

                seed_dict = {input_type[0]: seed}

                seed_time = None
                if edge_label_time is not None:  # Always disjoint.
                    seed_time = paddle.concat([src_time, dst_time], axis=0)

                out = await sample_fn(
                    NodeSamplerInput(
                        input_id=inputs.input_id,
                        node=seed,
                        time=seed_time,
                        input_type=input_type[0],
                    )
                )

            # Enhance `out` by label information ##############################
            if disjoint:
                for key, batch in out.batch.items():
                    out.batch[key] = batch % num_pos

            if neg_sampling is None or neg_sampling.is_binary():
                if disjoint:
                    if input_type[0] != input_type[-1]:
                        edge_label_index = paddle.arange(2 * (num_pos + num_neg))
                        edge_label_index = edge_label_index.reshape([2, -1])
                    else:
                        edge_label_index = paddle.arange(2 * (num_pos + num_neg))
                        edge_label_index = edge_label_index.reshape([2, -1])
                else:
                    edge_label_index = paddle.stack([src, dst], axis=0)

                out.metadata = (input_id, edge_label_index, edge_label, src_time)

            elif neg_sampling.is_triplet():
                src_index = paddle.arange(num_pos)
                dst_pos_index = paddle.arange(num_pos, 2 * num_pos)
                dst_neg_index = paddle.arange(2 * num_pos, 2 * num_pos + num_neg)

                out.metadata = (
                    input_id,
                    src_index,
                    dst_pos_index,
                    dst_neg_index,
                    src_time,
                )

        # Homogeneous Neighborhood Sampling ###################################

        else:
            seed = paddle.concat([src, dst], axis=0)
            seed_time = None

            if edge_label_time is not None:  # Always disjoint.
                seed_time = paddle.concat([src_time, dst_time])

            out = await sample_fn(
                NodeSamplerInput(
                    input_id=inputs.input_id,
                    node=seed,
                    time=seed_time,
                    input_type=None,
                )
            )

            # Enhance `out` by label information ##############################
            if neg_sampling is None or neg_sampling.is_binary():
                if disjoint:
                    out.batch = out.batch % num_pos
                    edge_label_index = paddle.arange(seed.shape[0]).reshape([2, -1])
                else:
                    edge_label_index = paddle.stack([src, dst], axis=0)

                out.metadata = (input_id, edge_label_index, edge_label, src_time)

            elif neg_sampling.is_triplet():
                out.batch = out.batch % num_pos
                src_index = paddle.arange(num_pos)
                dst_pos_index = paddle.arange(num_pos, 2 * num_pos)
                dst_neg_index = paddle.arange(2 * num_pos, seed.shape[0])

                out.metadata = (
                    input_id,
                    src_index,
                    dst_pos_index,
                    dst_neg_index,
                    src_time,
                )

        return out

    def _get_sampler_output(
            self,
            outputs: List[SamplerOutput],
            seed_size: int,
            p_id: int,
            src_batch: Optional[Tensor] = None,
    ) -> SamplerOutput:
        r"""Used when seed nodes belongs to one partition. Its purpose is to
        remove seed nodes from sampled nodes and calculates how many neighbors
        were sampled by each src node based on the
        :obj:`cumsum_neighbors_per_node`. Returns updated sampler output.
        """
        cumsum_neighbors_per_node = outputs[p_id].metadata[0]

        # do not include seed
        outputs[p_id].node = outputs[p_id].node[seed_size:]

        begin = np.array(cumsum_neighbors_per_node[1:])
        end = np.array(cumsum_neighbors_per_node[:-1])

        sampled_nbrs_per_node = list(np.subtract(begin, end))

        outputs[p_id].metadata = (sampled_nbrs_per_node,)

        if self.disjoint:
            batch = [[src_batch[i]] * nbrs_per_node
                     for i, nbrs_per_node in enumerate(sampled_nbrs_per_node)]
            outputs[p_id].batch = paddle.to_tensor(
                list(itertools.chain.from_iterable(batch)), dtype='int64')

        return outputs[p_id]

    def _merge_sampler_outputs(
            self,
            partition_ids: Tensor,
            partition_orders: Tensor,
            outputs: List[SamplerOutput],
            one_hop_num: int,
            src_batch: Optional[Tensor] = None,
    ) -> SamplerOutput:
        r"""Merges samplers outputs from different partitions, so that they
        are sorted according to the sampling order. Removes seed nodes from
        sampled nodes and calculates how many neighbors were sampled by each
        src node based on the :obj:`cumsum_neighbors_per_node`. Leverages the
        :obj:`pyg-lib` :meth:`merge_sampler_outputs` function.

        Args:
            partition_ids (paddle.Tensor): Contains information on which
                partition seeds nodes are located on.
            partition_orders (paddle.Tensor): Contains information about the
                order of seed nodes in each partition.
            outputs (List[SamplerOutput]): List of all samplers outputs.
            one_hop_num (int): Max number of neighbors sampled in the current
                layer.
            src_batch (paddle.Tensor, optional): The batch assignment of seed
                nodes. (default: :obj:`None`)

        Returns:
            SamplerOutput: Containing all merged outputs.
        """
        sampled_nodes_with_dupl = [
            o.node if o is not None else paddle.empty([0], dtype='int64')
            for o in outputs
        ]
        edge_ids = [
            o.edge if o is not None else paddle.empty([0], dtype='int64')
            for o in outputs
        ]
        cumm_sampled_nbrs_per_node = [
            o.metadata[0] if o is not None else [] for o in outputs
        ]

        partition_ids = partition_ids.numpy().tolist()
        partition_orders = partition_orders.numpy().tolist()

        partitions_num = self.graph_store.meta["num_parts"]

        # Implement custom merging logic since `torch.ops.pyg.merge_sampler_outputs` does not directly translate to Paddle.
        # Placeholder logic assumes data is concatenated. Adjust based on the actual library behavior.
        out_node_with_dupl = paddle.concat(sampled_nodes_with_dupl)
        out_edge = paddle.concat(edge_ids)
        out_sampled_nbrs_per_node = list(itertools.chain.from_iterable(cumm_sampled_nbrs_per_node))

        if self.disjoint:
            out_batch = paddle.concat([o.batch for o in outputs if o is not None])
        else:
            out_batch = None

        return SamplerOutput(
            out_node_with_dupl,
            None,
            None,
            out_edge,
            out_batch if self.disjoint else None,
            metadata=(out_sampled_nbrs_per_node,),
        )

    async def sample_one_hop(
            self,
            srcs: Tensor,
            one_hop_num: int,
            seed_time: Optional[Tensor] = None,
            src_batch: Optional[Tensor] = None,
            edge_type: Optional[EdgeType] = None,
    ) -> SamplerOutput:
        r"""Samples one-hop neighbors for a set of seed nodes in :obj:`srcs`.
        If seed nodes are located on a local partition, evaluates the sampling
        function on the current machine. If seed nodes are from a remote
        partition, sends a request to a remote machine that contains this
        partition.
        """
        src_node_type = None if not self.is_hetero else edge_type[2]
        partition_ids = self.graph_store.get_partition_ids_from_nids(
            srcs, src_node_type)
        partition_orders = paddle.zeros([len(partition_ids)], dtype="int64")

        p_outputs: List[SamplerOutput] = [
                                             None
                                         ] * self.graph_store.meta["num_parts"]
        futs = []

        local_only = True
        single_partition = len(set(partition_ids.numpy().tolist())) == 1

        for i in range(self.graph_store.num_partitions):
            p_id = (self.graph_store.partition_idx +
                    i) % self.graph_store.num_partitions
            p_mask = partition_ids == p_id
            p_srcs = paddle.masked_select(srcs, p_mask)
            p_seed_time = (paddle.masked_select(seed_time, p_mask)
                           if self.temporal else None)

            p_indices = paddle.arange(len(p_srcs), dtype="int64")
            partition_orders[p_mask] = p_indices

            if p_srcs.shape[0] > 0:
                if p_id == self.graph_store.partition_idx:
                    # Sample for one hop on a local machine:
                    p_nbr_out = self._sample_one_hop(p_srcs, one_hop_num,
                                                     p_seed_time, edge_type)
                    p_outputs.pop(p_id)
                    p_outputs.insert(p_id, p_nbr_out)

                else:  # Sample on a remote machine:
                    local_only = False
                    to_worker = self.rpc_router.get_to_worker(p_id)
                    futs.append(
                        rpc_async(
                            to_worker,
                            self.rpc_sample_callee_id,
                            args=(p_srcs, one_hop_num, p_seed_time, edge_type),
                        ))

        if not local_only:
            # Src nodes are remote
            res_fut_list = await to_asyncio_future(
                paddle.futures.collect_all(futs))
            for i, res_fut in enumerate(res_fut_list):
                p_id = (self.graph_store.partition_idx + i +
                        1) % self.graph_store.num_partitions
                p_outputs.pop(p_id)
                p_outputs.insert(p_id, res_fut.wait())

        # All src nodes are in the same partition
        if single_partition:
            return self._get_sampler_output(p_outputs, len(srcs),
                                            partition_ids[0], src_batch)

        return self._merge_sampler_outputs(partition_ids, partition_orders,
                                           p_outputs, one_hop_num, src_batch)

    def _sample_one_hop(
            self,
            input_nodes: Tensor,
            num_neighbors: int,
            seed_time: Optional[Tensor] = None,
            edge_type: Optional[EdgeType] = None,
    ) -> SamplerOutput:
        r"""Implements one-hop neighbor sampling for a set of input nodes for a
        specific edge type.
        """
        if not self.is_hetero:
            colptr = self._sampler.colptr
            row = self._sampler.row
            node_time = self.node_time
            edge_time = self.edge_time
        else:
            # Given edge type, get input data and evaluate sample function:
            rel_type = '__'.join(edge_type)
            colptr = self._sampler.colptr_dict[rel_type]
            row = self._sampler.row_dict[rel_type]
            # `node_time` is a destination node time:
            node_time = (self.node_time or {}).get(edge_type[0], None)
            edge_time = (self.edge_time or {}).get(edge_type, None)

        out = paddle.ops.pyg.dist_neighbor_sample(
            colptr,
            row,
            input_nodes.astype(colptr.dtype),
            num_neighbors,
            node_time,
            edge_time,
            seed_time,
            None,  # TODO: edge_weight
            True,  # csc
            self.replace,
            self.subgraph_type != SubgraphType.induced,
            self.disjoint and self.temporal,
            self.temporal_strategy,
        )
        node, edge, cumsum_neighbors_per_node = out

        if self.disjoint and self.temporal:
            # We create a batch during the step of merging sampler outputs.
            _, node = paddle.transpose(node, [1, 0])

        return SamplerOutput(
            node=node,
            row=None,
            col=None,
            edge=edge,
            batch=None,
            metadata=(cumsum_neighbors_per_node,),
        )

    async def _collate_fn(
            self, output: Union[SamplerOutput, HeteroSamplerOutput]
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Collect labels and features for the sampled subgraph if necessary,
        and put them into a sample message.
        """
        if self.is_hetero:
            labels = {}
            nfeats = {}
            efeats = {}
            labels = self.feature_store.labels
            if labels is not None:
                if isinstance(self.input_type, tuple):  # Edge labels.
                    labels = {
                        self.input_type: paddle.index_select(
                            labels[self.input_type], output.edge[self.input_type])
                    }
                else:  # Node labels.
                    labels = {
                        self.input_type: paddle.index_select(
                            labels[self.input_type], output.node[self.input_type])
                    }
            # Collect node features.
            if output.node is not None:
                for ntype in output.node.keys():
                    if output.node[ntype].numel() > 0:
                        fut = self.feature_store.lookup_features(
                            is_node_feat=True,
                            index=output.node[ntype],
                            input_type=ntype,
                        )
                        nfeat = await to_asyncio_future(fut)
                        nfeat = nfeat.cpu()
                        nfeats[ntype] = nfeat
                    else:
                        nfeats[ntype] = None
            # Collect edge features.
            if output.edge is not None and self.with_edge_attr:
                for edge_type in output.edge.keys():
                    if output.edge[edge_type].numel() > 0:
                        fut = self.feature_store.lookup_features(
                            is_node_feat=False,
                            index=output.edge[edge_type],
                            input_type=edge_type,
                        )
                        efeat = await to_asyncio_future(fut)
                        efeat = efeat.cpu()
                        efeats[edge_type] = efeat
                    else:
                        efeats[edge_type] = None

        else:  # Homogeneous:
            # Collect node labels.
            if self.feature_store.labels is not None:
                labels = paddle.index_select(
                    self.feature_store.labels, output.node)
            else:
                labels = None
            # Collect node features.
            if output.node is not None:
                fut = self.feature_store.lookup_features(
                    is_node_feat=True, index=output.node)
                nfeats = await to_asyncio_future(fut)
                nfeats = nfeats.cpu()
            else:
                nfeats = None
            # Collect edge features.
            if output.edge is not None and self.with_edge_attr:
                fut = self.feature_store.lookup_features(
                    is_node_feat=False, index=output.edge)
                efeats = await to_asyncio_future(fut)
                efeats = efeats.cpu()
            else:
                efeats = None

        output.metadata = (*output.metadata, nfeats, labels, efeats)
        return output

    @property
    def edge_permutation(self) -> None:
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pid={paddle.device.get_current_pid()})'
