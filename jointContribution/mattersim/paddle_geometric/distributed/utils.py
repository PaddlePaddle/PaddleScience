from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor

from paddle_geometric.data import HeteroData
from paddle_geometric.distributed.local_feature_store import LocalFeatureStore
from paddle_geometric.distributed.local_graph_store import LocalGraphStore
from paddle_geometric.sampler import SamplerOutput
from paddle_geometric.typing import EdgeType, NodeType


@dataclass
class DistEdgeHeteroSamplerInput:
    r"""The sampling input of
    :meth:`~paddle_geometric.distributed.DistNeighborSampler.node_sample`
    used during distributed heterogeneous link sampling when source and
    target node types of an input edge are different.
    """
    input_id: Optional[Tensor]
    node_dict: Dict[NodeType, Tensor]
    time_dict: Optional[Dict[NodeType, Tensor]] = None
    input_type: Optional[EdgeType] = None


class NodeDict:
    r"""Class used during heterogeneous sampling."""
    def __init__(self, node_types, num_hops):
        self.src: Dict[NodeType, List[Tensor]] = {
            k: (num_hops + 1) * [paddle.zeros([0], dtype='int64')]
            for k in node_types
        }
        self.with_dupl: Dict[NodeType, Tensor] = {
            k: paddle.zeros([0], dtype='int64')
            for k in node_types
        }
        self.out: Dict[NodeType, Tensor] = {
            k: paddle.zeros([0], dtype='int64')
            for k in node_types
        }
        self.seed_time: Dict[NodeType, List[Tensor]] = {
            k: num_hops * [paddle.zeros([0], dtype='int64')]
            for k in node_types
        }


class BatchDict:
    r"""Class used during disjoint heterogeneous sampling."""
    def __init__(self, node_types, num_hops):
        self.src: Dict[NodeType, List[Tensor]] = {
            k: (num_hops + 1) * [paddle.zeros([0], dtype='int64')]
            for k in node_types
        }
        self.with_dupl: Dict[NodeType, Tensor] = {
            k: paddle.zeros([0], dtype='int64')
            for k in node_types
        }
        self.out: Dict[NodeType, Tensor] = {
            k: paddle.zeros([0], dtype='int64')
            for k in node_types
        }


def remove_duplicates(
    out: SamplerOutput,
    node: Tensor,
    batch: Optional[Tensor] = None,
    disjoint: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    num_nodes = node.shape[0]
    node_combined = paddle.concat([node, out.node])

    if not disjoint:
        _, idx = np.unique(node_combined.numpy(), return_index=True)
        idx = paddle.to_tensor(idx).sort().values
        node = paddle.index_select(node_combined, index=idx)
        src = node[num_nodes:]
        return (src, node, None, None)
    else:
        batch_combined = paddle.concat([batch, out.batch])
        node_batch = paddle.stack([batch_combined, node_combined], axis=0)
        _, idx = np.unique(node_batch.numpy(), axis=1, return_index=True)
        idx = paddle.to_tensor(idx).sort().values
        batch = paddle.index_select(batch_combined, index=idx)
        node = paddle.index_select(node_combined, index=idx)
        src_batch = batch[num_nodes:]
        src = node[num_nodes:]
        return (src, node, src_batch, batch)


def filter_dist_store(
    feature_store: LocalFeatureStore,
    graph_store: LocalGraphStore,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Optional[Tensor]],
    custom_cls: Optional[HeteroData] = None,
    meta: Optional[Dict[str, Tensor]] = None,
    input_type: str = None,
) -> HeteroData:
    r"""Constructs a :class:`HeteroData` object from a feature store."""
    data = custom_cls() if custom_cls is not None else HeteroData()
    nfeats, labels, efeats = meta[-3:]

    required_edge_attrs = []
    for attr in graph_store.get_all_edge_attrs():
        key = attr.edge_type
        if key in row_dict and key in col_dict:
            required_edge_attrs.append(attr)
            edge_index = paddle.stack([row_dict[key], col_dict[key]], axis=0)
            data[attr.edge_type].edge_index = edge_index

    required_node_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        if attr.group_name in node_dict:
            attr.index = node_dict[attr.group_name]
            required_node_attrs.append(attr)
            data[attr.group_name].num_nodes = attr.index.shape[0]

    if nfeats:
        for attr in required_node_attrs:
            if nfeats[attr.group_name] is not None:
                data[attr.group_name][attr.attr_name] = nfeats[attr.group_name]

    if efeats:
        for attr in required_edge_attrs:
            if efeats[attr.edge_type] is not None:
                data[attr.edge_type].edge_attr = efeats[attr.edge_type]

    if labels:
        data[input_type].y = labels[input_type]

    return data


def as_str(inputs: Union[NodeType, EdgeType]) -> str:
    if isinstance(inputs, NodeType):
        return inputs
    elif isinstance(inputs, (list, tuple)) and len(inputs) == 3:
        return '__'.join(inputs)
    return ''


def reverse_edge_type(etype: EdgeType) -> EdgeType:
    src, rel, dst = etype
    if src != dst:
        if rel.split('_', 1)[0] == 'rev':
            rel = rel.split('_', 1)[1]
        else:
            rel = 'rev_' + rel
    return dst, rel, src
