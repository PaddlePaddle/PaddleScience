from inspect import isfunction
from typing import List
from typing import Union

import paddle


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32):
    num_nodes = x.shape[0]
    if batch is None:
        batch = paddle.zeros(shape=[num_nodes], dtype=paddle.int64)

    dist_matrix = paddle.norm(x.unsqueeze(1) - x.unsqueeze(0), axis=-1, p=2)

    adj_matrix = dist_matrix < r

    if not loop:
        adj_matrix = adj_matrix * (1 - paddle.eye(num_nodes, dtype=paddle.bool))

    mask = batch.unsqueeze(1) == batch.unsqueeze(0)
    adj_matrix = adj_matrix * mask

    degree = adj_matrix.sum(axis=-1)
    if max_num_neighbors < degree.max():
        idx = degree.argsort(descending=True)
        idx = idx[:max_num_neighbors]
        adj_matrix = adj_matrix[:, idx]

    return adj_matrix


def k_hop_subgraph(
    edge_index: paddle.Tensor,
    num_hops: int,
    node_idx: Union[int, List[int], paddle.Tensor],
    relabel_nodes: bool = False,
) -> paddle.Tensor:
    if not isinstance(node_idx, paddle.Tensor):
        node_idx = paddle.to_tensor(node_idx, dtype="int64")

    visited = paddle.zeros([edge_index.max() + 1], dtype="bool")
    queue = node_idx.tolist() if isinstance(node_idx, paddle.Tensor) else node_idx
    visited[queue] = True
    sub_edge_index = []

    current_hop = 0

    while queue and current_hop < num_hops:
        current_hop += 1
        next_queue = []

        for node in queue:
            neighbors = edge_index[1] == node
            neighbors = edge_index[0][neighbors]
            neighbors = neighbors[~visited[neighbors]]

            next_queue.extend(neighbors.tolist())
            visited[neighbors] = True

            for neighbor in neighbors:
                if relabel_nodes:
                    original_idx = (
                        paddle.nonzero(node_idx == node)[0].item()
                        if isinstance(node_idx, paddle.Tensor)
                        else node_idx.index(node)
                    )
                    sub_edge_index.append([original_idx, len(sub_edge_index) // 2 + 1])
                else:
                    sub_edge_index.append([node, neighbor])

        queue = next_queue

    sub_edge_index = paddle.to_tensor(sub_edge_index, dtype="int64")
    if relabel_nodes:
        return sub_edge_index.reshape([-1, 2])[:, 1]
    else:
        return sub_edge_index.reshape([-1, 2])
