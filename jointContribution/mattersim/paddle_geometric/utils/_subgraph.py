from typing import List, Literal, Optional, Tuple, Union, overload

import paddle
from paddle import Tensor

from paddle_geometric.typing import OptTensor, PairTensor
from paddle_geometric.utils import scatter
from paddle_geometric.utils.map import map_index
from paddle_geometric.utils.mask import index_to_mask
from paddle_geometric.utils.num_nodes import maybe_num_nodes

def get_num_hops(model: paddle.nn.Layer) -> int:
    r"""Returns the number of hops the model is aggregating information
    from.

    Example:
        >>> class GNN(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = GCNConv(3, 16)
        ...         self.conv2 = GCNConv(16, 16)
        ...         self.lin = Linear(16, 2)
        ...
        ...     def forward(self, x, edge_index):
        ...         x = self.conv1(x, edge_index).relu()
        ...         x = self.conv2(x, edge_index).relu()
        ...         return self.lin(x)
        >>> get_num_hops(GNN())
        2
    """
    from paddle_geometric.nn.conv import MessagePassing
    num_hops = 0
    for module in model.sublayers():
        if isinstance(module, MessagePassing):
            num_hops += 1
    return num_hops


@overload
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[False],
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[True],
) -> Tuple[Tensor, OptTensor, Tensor]:
    pass


def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = paddle.to_tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """
    # device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = paddle.to_tensor(subset, dtype=paddle.int64)

    if subset.dtype != paddle.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.shape[0]
        node_mask = subset
        subset = node_mask.nonzero().flatten()

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index, _ = map_index(
            edge_index.flatten(),
            subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index.reshape([2, -1])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    size: Optional[Tuple[int, int]] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`."""

    device = edge_index.device

    src_subset, dst_subset = subset
    if not isinstance(src_subset, Tensor):
        src_subset = paddle.to_tensor(src_subset, dtype=paddle.int64)
    if not isinstance(dst_subset, Tensor):
        dst_subset = paddle.to_tensor(dst_subset, dtype=paddle.int64)

    if src_subset.dtype != paddle.bool:
        src_size = int(edge_index[0].max()) + 1 if size is None else size[0]
        src_node_mask = index_to_mask(src_subset, size=src_size)
    else:
        src_size = src_subset.shape[0]
        src_node_mask = src_subset
        src_subset = src_subset.nonzero().flatten()

    if dst_subset.dtype != paddle.bool:
        dst_size = int(edge_index[1].max()) + 1 if size is None else size[1]
        dst_node_mask = index_to_mask(dst_subset, size=dst_size)
    else:
        dst_size = dst_subset.shape[0]
        dst_node_mask = dst_subset
        dst_subset = dst_subset.nonzero().flatten()

    edge_mask = src_node_mask[edge_index[0]] & dst_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        src_index, _ = map_index(edge_index[0], src_subset, max_index=src_size,
                                 inclusive=True)
        dst_index, _ = map_index(edge_index[1], dst_subset, max_index=dst_size,
                                 inclusive=True)
        edge_index = paddle.stack([src_index, dst_index], axis=0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops."""

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = paddle.full([num_nodes], False, dtype=paddle.bool)
    edge_mask = paddle.full([row.shape[0]], False, dtype=paddle.bool)

    if isinstance(node_idx, int):
        node_idx = paddle.to_tensor([node_idx], dtype=paddle.int64)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = paddle.to_tensor(node_idx, dtype=paddle.int64)
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask = paddle.full([num_nodes], False, dtype='bool')
        node_mask[subsets[-1]] = True
        # paddle.assign(paddle.index_select(node_mask, axis=0, index=row), edge_mask)
        node_mask_int = paddle.cast(node_mask, 'int32')
        edge_mask = paddle.index_select(node_mask_int, axis=0, index=row)
        edge_mask = paddle.cast(edge_mask, 'bool')
        subsets.append(col[edge_mask])

    subset, inv = paddle.concat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = paddle.full([num_nodes], -1, dtype=paddle.int64)
        mapping[subset] = paddle.arange(subset.shape[0], dtype=paddle.int64)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask

@overload
def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[False],
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[True],
) -> Tuple[Tensor, OptTensor, Tensor]:
    pass


def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
    r"""Returns the induced subgraph of the hyper graph of
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (paddle.Tensor or [int]): The nodes to keep.
        edge_index (LongTensor): Hyperedge tensor
            with shape :obj:`[2, num_edges*num_nodes_per_edge]`, where
            :obj:`edge_index[1]` denotes the hyperedge index and
            :obj:`edge_index[0]` denotes the node indices that are connected
            by the hyperedge.
        edge_attr (paddle.Tensor, optional): Edge weights or multi-dimensional
            edge features of shape :obj:`[num_edges, *]`.
            (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the
            resulting :obj:`edge_index` will be relabeled to hold
            consecutive indices starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index[0]) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = paddle.to_tensor([[0, 1, 2, 1, 2, 3, 0, 2, 3],
        ...                                [0, 0, 0, 1, 1, 1, 2, 2, 2]])
        >>> edge_attr = paddle.to_tensor([3, 2, 6])
        >>> subset = paddle.to_tensor([0, 3])
        >>> hyper_subgraph(subset, edge_index, edge_attr)
        (tensor([[0, 3],
                 [0, 0]]),
         tensor([ 6.]))

        >>> hyper_subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[0, 3],
                 [0, 0]]),
         tensor([ 6.]),
         tensor([False, False, True]))
    """
    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = paddle.to_tensor(subset, dtype=paddle.int64)

    if subset.dtype != paddle.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.shape[0]
        node_mask = subset

    # Mask all connections that contain a node not in the subset
    hyper_edge_connection_mask = node_mask[edge_index[0]]  # num_edges*num_nodes_per_edge

    # Mask hyperedges that contain one or less nodes from the subset
    edge_mask = scatter(hyper_edge_connection_mask.astype(paddle.int64),
                        edge_index[1], reduce='sum') > 1

    # Mask connections if hyperedge contains one or less nodes from the subset
    # or is connected to a node not in the subset
    hyper_edge_connection_mask = hyper_edge_connection_mask & edge_mask[edge_index[1]]

    edge_index = edge_index[:, hyper_edge_connection_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # Relabel edges
    edge_idx = paddle.zeros_like(edge_mask, dtype=paddle.int64)
    edge_idx[edge_mask] = paddle.arange(edge_mask.sum().item())
    edge_index = paddle.concat([edge_index[0].unsqueeze(0), edge_idx[edge_index[1]].unsqueeze(0)], axis=0)

    if relabel_nodes:
        node_idx = paddle.zeros_like(node_mask, dtype=paddle.int64)
        node_idx[subset] = paddle.arange(node_mask.sum().item())
        edge_index = paddle.concat([node_idx[edge_index[0]].unsqueeze(0), edge_index[1].unsqueeze(0)], axis=0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr