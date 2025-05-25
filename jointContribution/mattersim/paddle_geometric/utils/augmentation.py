from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.utils import cumsum, negative_sampling, scatter


def shuffle_node(
    x: Tensor,
    batch: Optional[Tensor] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    if not training:
        perm = paddle.arange(x.shape[0])
        return x, perm
    if batch is None:
        perm = paddle.randperm(x.shape[0])
        return paddle.index_select(x, perm), perm
    num_nodes = scatter(paddle.ones([x.shape[0]]), batch, reduce='sum')
    ptr = cumsum(num_nodes)
    perm = paddle.concat([
        paddle.randperm(n) + offset
        for offset, n in zip(ptr[:-1], num_nodes)
    ])
    return paddle.index_select(x, perm), perm


def mask_feature(
    x: Tensor,
    p: float = 0.5,
    mode: str = 'col',
    fill_value: float = 0.,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 (got {p})')
    if not training or p == 0.0:
        return x, paddle.ones_like(x, dtype='bool')
    assert mode in ['row', 'col', 'all']

    if mode == 'row':
        mask = paddle.rand([x.shape[0]]) >= p
        mask = mask.unsqueeze(1)
    elif mode == 'col':
        mask = paddle.rand([x.shape[1]]) >= p
        mask = mask.unsqueeze(0)
    else:
        mask = paddle.rand(x.shape) >= p

    x = paddle.where(mask, x, paddle.full_like(x, fill_value))
    return x, mask


def add_random_edge(
    edge_index: Tensor,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f"Ratio of added edges has to be between 0 and 1 (got '{p}')")
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise RuntimeError("'force_undirected' is not supported for bipartite graphs")

    device = edge_index.place
    if not training or p == 0.0:
        edge_index_to_add = paddle.empty([2, 0], dtype='int64')
        return edge_index, edge_index_to_add

    edge_index_to_add = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=round(edge_index.shape[1] * p),
        force_undirected=force_undirected,
    )

    edge_index = paddle.concat([edge_index, edge_index_to_add], axis=1)

    return edge_index, edge_index_to_add
