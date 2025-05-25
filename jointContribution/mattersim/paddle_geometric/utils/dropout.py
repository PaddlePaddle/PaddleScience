from typing import Optional, Tuple

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.deprecation import deprecated
from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import cumsum, degree, sort_edge_index, subgraph
from paddle_geometric.utils.num_nodes import maybe_num_nodes


def filter_adj(row: Tensor, col: Tensor, edge_attr: OptTensor,
               mask: Tensor) -> Tuple[Tensor, Tensor, OptTensor]:
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


@deprecated("use 'dropout_edge' instead")
def dropout_adj(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, OptTensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index

    mask = paddle.rand([row.shape[0]]) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = paddle.stack(
            [paddle.concat([row, col], axis=0),
             paddle.concat([col, row], axis=0)], axis=0)
        if edge_attr is not None:
            edge_attr = paddle.concat([edge_attr, edge_attr], axis=0)
    else:
        edge_index = paddle.stack([row, col], axis=0)

    return edge_index, edge_attr


def dropout_node(
    edge_index: Tensor,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    training: bool = True,
    relabel_nodes: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = paddle.ones([num_nodes], dtype=paddle.bool)
        edge_mask = paddle.ones([edge_index.shape[1]], dtype=paddle.bool)
        return edge_index, edge_mask, node_mask

    prob = paddle.rand([num_nodes])
    node_mask = prob > p
    edge_index, _, edge_mask = subgraph(
        node_mask,
        edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        return_edge_mask=True,
    )
    return edge_index, edge_mask, node_mask


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = paddle.ones([edge_index.shape[1]], dtype=paddle.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = paddle.rand([row.shape[0]]) >= p

    if force_undirected:
        edge_mask = edge_mask.astype(paddle.int32)

        row, col = edge_index
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = paddle.concat([edge_index, paddle.flip(edge_index, [0])], axis=1)
        edge_mask = paddle.nonzero(edge_mask).tile([2, 1]).squeeze()

    return edge_index, edge_mask


def dropout_path(edge_index: Tensor, p: float = 0.2, walks_per_node: int = 1,
                 walk_length: int = 3, num_nodes: Optional[int] = None,
                 is_sorted: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    num_edges = edge_index.shape[1]
    edge_mask = paddle.ones([num_edges], dtype=paddle.bool)
    if not training or p == 0.0:
        return edge_index, edge_mask

    if not paddle_geometric.typing.WITH_PADDLE_CLUSTER or is_compiling():
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_orders = None
    ori_edge_index = edge_index
    if not is_sorted:
        edge_orders = paddle.arange(num_edges)
        edge_index, edge_orders = sort_edge_index(edge_index, edge_orders,
                                                  num_nodes=num_nodes)

    row, col = edge_index
    sample_mask = paddle.rand([row.shape[0]]) <= p
    start = row[sample_mask].repeat(walks_per_node)

    rowptr = cumsum(degree(row, num_nodes=num_nodes, dtype=paddle.int64))
    n_id, e_id = paddle.ops.torch_cluster.random_walk(rowptr, col, start,
                                                     walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].reshape([-1])  # filter illegal edges

    if edge_orders is not None:  # Permute edge indices:
        e_id = paddle.index_select(edge_orders, e_id)
    edge_mask[e_id] = False
    edge_index = ori_edge_index[:, edge_mask]

    return edge_index, edge_mask
