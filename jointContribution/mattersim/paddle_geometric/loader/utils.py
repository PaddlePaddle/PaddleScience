import copy
import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.data import (
    Data,
    FeatureStore,
    GraphStore,
    HeteroData,
    TensorAttr,
    remote_backend_utils,
)
from paddle_geometric.data.storage import EdgeStorage, NodeStorage
from paddle_geometric.typing import (
    EdgeType,
    FeatureTensorType,
    InputEdges,
    InputNodes,
    NodeType,
    OptTensor,
    SparseTensor,
    TensorFrame,
)


def index_select(
    value: FeatureTensorType,
    index: Tensor,
    dim: int = 0,
) -> Tensor:
    r"""Indexes the :obj:`value` tensor along dimension :obj:`dim` using the
    entries in :obj:`index`."""

    # Paddle currently only supports indexing via `paddle.int64`:
    index = index.astype('int64')

    if isinstance(value, Tensor):
        out: Optional[Tensor] = None
        if paddle.utils.data.get_worker_info() is not None:
            # If we are in a background process, we write directly into a
            # shared memory tensor to avoid an extra copy:
            size = list(value.shape)
            size[dim] = index.shape[0]
            numel = math.prod(size)
            out = value.new_tensor([0] * numel).reshape(size)

        return paddle.index_select(value, dim, index)

    if isinstance(value, TensorFrame):
        assert dim == 0
        return value[index]

    elif isinstance(value, np.ndarray):
        return paddle.to_tensor(np.take(value, index, axis=dim))

    raise ValueError(f"Encountered invalid feature tensor type "
                     f"(got '{type(value)}')")


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index: Tensor):
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.shape[0]

        elif store.is_node_attr(key):
            if isinstance(value, (Tensor, TensorFrame)):
                index = index.astype(value.dtype)
            elif isinstance(value, np.ndarray):
                index = index.cpu()
            dim = store._parent().__cat_dim__(key, value, store)
            out_store[key] = index_select(value, index, dim=dim)


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                       col: Tensor, index: OptTensor, perm: OptTensor = None):
    # Filters an edge storage object to only hold the edges in `index`:
    for key, value in store.items():
        if key == 'edge_index':
            edge_index = paddle.concat([row, col], axis=0).to(value.device)
            out_store.edge_index = edge_index

        elif key == 'adj_t':
            row = row.astype(value.device())
            col = col.astype(value.device())
            edge_attr = value.storage.value()
            if edge_attr is not None:
                if index is not None:
                    index = index.astype(edge_attr.device)
                    edge_attr = index_select(edge_attr, index, dim=0)
                else:
                    edge_attr = None
            sparse_sizes = out_store.size()[::-1]
            out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                           sparse_sizes=sparse_sizes,
                                           is_sorted=False, trust_data=True)

        elif store.is_edge_attr(key):
            if index is None:
                out_store[key] = None
                continue

            dim = store._parent().__cat_dim__(key, value, store)
            if isinstance(value, (Tensor, TensorFrame)):
                index = index.astype(value.dtype)
            elif isinstance(value, np.ndarray):
                index = index.cpu()
            if perm is None:
                out_store[key] = index_select(value, index, dim=dim)
            else:
                if isinstance(value, (Tensor, TensorFrame)):
                    perm = perm.astype(value.dtype)
                elif isinstance(value, np.ndarray):
                    perm = perm.cpu()
                out_store[key] = index_select(
                    value,
                    perm[index.astype('int64')],
                    dim=dim,
                )


def filter_data(data: Data, node: Tensor, row: Tensor, col: Tensor,
                edge: OptTensor, perm: OptTensor = None) -> Data:
    out = copy.copy(data)
    filter_node_store_(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


def filter_hetero_data(
    data: HeteroData,
    node_dict: Dict[NodeType, Tensor],
    row_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    edge_dict: Dict[EdgeType, OptTensor],
    perm_dict: Optional[Dict[EdgeType, OptTensor]] = None,
) -> HeteroData:
    out = copy.copy(data)

    for node_type in out.node_types:
        if node_type not in node_dict:
            node_dict[node_type] = paddle.empty([0], dtype='int64')

        filter_node_store_(data[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in out.edge_types:
        if edge_type not in row_dict:
            row_dict[edge_type] = paddle.empty([0], dtype='int64')
        if edge_type not in col_dict:
            col_dict[edge_type] = paddle.empty([0], dtype='int64')
        if edge_type not in edge_dict:
            edge_dict[edge_type] = paddle.empty([0], dtype='int64')

        filter_edge_store_(
            data[edge_type],
            out[edge_type],
            row_dict[edge_type],
            col_dict[edge_type],
            edge_dict[edge_type],
            perm_dict.get(edge_type, None) if perm_dict else None,
        )

    return out


def filter_custom_store(
    feature_store: FeatureStore,
    graph_store: GraphStore,
    node: Tensor,
    row: Tensor,
    col: Tensor,
    edge: OptTensor,
    custom_cls: Optional[Data] = None,
) -> Data:
    data = custom_cls() if custom_cls is not None else Data()

    data.edge_index = paddle.concat([row, col], axis=0)

    required_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        attr.index = node  # TODO Support edge features.
        required_attrs.append(attr)
        data.num_nodes = attr.index.shape[0]

    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.attr_name] = tensors[i]

    return data


def filter_custom_hetero_store(
    feature_store: FeatureStore,
    graph_store: GraphStore,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, OptTensor],
    custom_cls: Optional[HeteroData] = None,
) -> HeteroData:
    data = custom_cls() if custom_cls is not None else HeteroData()

    for attr in graph_store.get_all_edge_attrs():
        key = attr.edge_type
        if key in row_dict and key in col_dict:
            edge_index = paddle.concat([row_dict[key], col_dict[key]], axis=0)
            data[attr.edge_type].edge_index = edge_index

    required_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        if attr.group_name in node_dict:
            attr.index = node_dict[attr.group_name]
            required_attrs.append(attr)
            data[attr.group_name].num_nodes = attr.index.shape[0]

    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.group_name][attr.attr_name] = tensors[i]

    return data


def get_input_nodes(
    data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
    input_nodes: Union[InputNodes, TensorAttr],
    input_id: Optional[Tensor] = None,
) -> Tuple[Optional[str], Tensor, Optional[Tensor]]:
    def to_index(nodes, input_id) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(nodes, Tensor) and nodes.dtype == paddle.bool:
            nodes = nodes.nonzero(as_tuple=False).view(-1)
            if input_id is not None:
                assert input_id.shape[0] == nodes.shape[0]
            else:
                input_id = nodes
            return nodes, input_id

        if not isinstance(nodes, Tensor):
            nodes = paddle.to_tensor(nodes, dtype='int64')

        if input_id is not None:
            assert input_id.shape[0] == nodes.shape[0]

        return nodes, input_id

    if isinstance(data, Data):
        if input_nodes is None:
            return None, paddle.arange(data.num_nodes), None
        return None, *to_index(input_nodes, input_id)

    elif isinstance(data, HeteroData):
        assert input_nodes is not None

        if isinstance(input_nodes, str):
            return input_nodes, paddle.arange(data[input_nodes].num_nodes), None

        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)

        node_type, input_nodes = input_nodes
        if input_nodes is None:
            return node_type, paddle.arange(data[node_type].num_nodes), None
        return node_type, *to_index(input_nodes, input_id)

    else:  # Tuple[FeatureStore, GraphStore]
        feature_store, graph_store = data
        assert input_nodes is not None

        if isinstance(input_nodes, Tensor):
            return None, *to_index(input_nodes, input_id)

        if isinstance(input_nodes, str):
            num_nodes = remote_backend_utils.num_nodes(  #
                feature_store, graph_store, input_nodes)
            return input_nodes, paddle.arange(num_nodes), None

        if isinstance(input_nodes, (list, tuple)):
            assert len(input_nodes) == 2
            assert isinstance(input_nodes[0], str)

            node_type, input_nodes = input_nodes
            if input_nodes is None:
                num_nodes = remote_backend_utils.num_nodes(  #
                    feature_store, graph_store, input_nodes)
                return node_type, paddle.arange(num_nodes), None

            return node_type, *to_index(input_nodes, input_id)


def get_edge_label_index(
    data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
    edge_label_index: InputEdges,
) -> Tuple[Optional[str], Tensor]:
    edge_type = None
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    assert edge_label_index is not None
    assert isinstance(edge_label_index, (list, tuple))

    if isinstance(data, HeteroData):
        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            edge_type = data._to_canonical(*edge_type)
            assert edge_type in data.edge_types
            return edge_type, data[edge_type].edge_index

        assert len(edge_label_index) == 2

        edge_type, edge_label_index = edge_label_index
        edge_type = data._to_canonical(*edge_type)

        if edge_label_index is None:
            return edge_type, data[edge_type].edge_index

        return edge_type, edge_label_index

    else:  # Tuple[FeatureStore, GraphStore]
        _, graph_store = data

        # Need the edge index in COO for LinkNeighborLoader:
        def _get_edge_index(edge_type):
            row_dict, col_dict, _ = graph_store.coo([edge_type])
            row = list(row_dict.values())[0]
            col = list(col_dict.values())[0]
            return paddle.stack((row, col), axis=0)

        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            return edge_type, _get_edge_index(edge_type)

        assert len(edge_label_index) == 2
        edge_type, edge_label_index = edge_label_index

        if edge_label_index is None:
            return edge_type, _get_edge_index(edge_type)

        return edge_type, edge_label_index


def infer_filter_per_worker(data: Any) -> bool:
    out = True
    if isinstance(data, (Data, HeteroData)) and data.is_cuda:
        out = False
    logging.debug(f"Inferred 'filter_per_worker={out}' option for feature "
                  f"fetching routines of the data loader")
    return out
