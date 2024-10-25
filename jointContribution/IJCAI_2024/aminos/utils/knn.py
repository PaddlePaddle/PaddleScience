from typing import List
from typing import Union

import numpy as np
import paddle
from scipy.spatial import cKDTree


def knn_scipy_batched(x, y, k, batch_x=None, batch_y=None):
    assert batch_x is not None and batch_y is not None, "Batch information is required."

    unique_batches = np.unique(batch_x)
    all_distances = np.full((x.shape[0], k), np.inf)
    all_indices = np.full((x.shape[0], k), -1)

    for batch in unique_batches:
        mask_x = batch_x == batch
        mask_y = batch_y == batch
        batch_x_points = x[mask_x]
        batch_y_points = y[mask_y]

        if batch_x_points.size == 0 or batch_y_points.size == 0:
            continue

        tree = cKDTree(batch_y_points)
        distances, indices = tree.query(batch_x_points, k=k)

        true_indices = np.where(mask_y)[0][indices]

        all_distances[mask_x] = distances
        all_indices[mask_x] = true_indices

    return all_distances, all_indices


def knn_graph(pos_tensor, k):
    dist = paddle.cdist(pos_tensor, pos_tensor)

    nn_indices = []

    for i in range(pos_tensor.shape[0]):
        distances = dist[i].numpy()
        distances[i] = np.inf

        indices = np.argsort(distances)[:k]
        nn_indices.append(indices)

    nn_indices_tensor = paddle.to_tensor(nn_indices)

    return nn_indices_tensor


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
