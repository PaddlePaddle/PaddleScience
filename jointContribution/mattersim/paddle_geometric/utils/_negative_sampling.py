import random
from typing import Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor


def negative_sampling(
        edge_index: Tensor,
        num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
        num_neg_samples: Optional[int] = None,
        method: str = "sparse",
        force_undirected: bool = False,
) -> Tensor:
    """
    Samples random negative edges for a given graph represented by `edge_index`.

    Args:
        edge_index (Tensor): The edge indices of the graph.
        num_nodes (int or Tuple[int, int], optional): The number of nodes, i.e.,
            max value + 1 of `edge_index`. If a tuple is provided, `edge_index`
            is treated as a bipartite graph with shape `(num_src_nodes, num_dst_nodes)`.
        num_neg_samples (int, optional): The number of negative samples to generate.
            If set to None, generates a negative edge for each positive edge.
        method (str, optional): Sampling method, "sparse" or "dense". Controls
            memory/runtime trade-offs. "sparse" works with any graph size, while
            "dense" performs faster checks for true negatives.
        force_undirected (bool, optional): If True, sampled negative edges will be
            undirected.

    Returns:
        Tensor: Negative edge indices.

    Examples:
        edge_index = paddle.to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        negative_sampling(edge_index)
    """
    assert method in ['sparse', 'dense']
    if num_nodes is None:
        num_nodes = max(edge_index.flatten().numpy()) + 1

    size = (num_nodes, num_nodes) if isinstance(num_nodes, int) else num_nodes
    bipartite = not isinstance(num_nodes, int)
    force_undirected = False if bipartite else force_undirected

    idx, population = edge_index_to_vector(edge_index, size, bipartite, force_undirected)

    if idx.shape[0] >= population:
        return paddle.empty((2, 0), dtype="int64")

    num_neg_samples = num_neg_samples or edge_index.shape[1]
    if force_undirected:
        num_neg_samples //= 2

    prob = 1.0 - idx.shape[0] / population
    sample_size = int(1.1 * num_neg_samples / prob)

    neg_idx = None
    if method == 'dense':
        mask = paddle.ones([population], dtype=paddle.bool)
        mask[idx] = False
        for _ in range(3):
            rnd = sample(population, sample_size, edge_index.place)
            rnd = rnd[mask[rnd]]
            neg_idx = paddle.concat([neg_idx, rnd]) if neg_idx is not None else rnd
            if neg_idx.shape[0] >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:
        idx = idx.cpu()
        for _ in range(3):
            rnd = sample(population, sample_size, 'cpu')
            mask = np.isin(rnd.numpy(), idx.numpy()) | (neg_idx is not None and np.isin(rnd, neg_idx.cpu()))
            rnd = rnd[~mask].to(edge_index.place)
            neg_idx = paddle.concat([neg_idx, rnd]) if neg_idx is not None else rnd
            if neg_idx.shape[0] >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def batched_negative_sampling(
        edge_index: Tensor,
        batch: Union[Tensor, Tuple[Tensor, Tensor]],
        num_neg_samples: Optional[int] = None,
        method: str = "sparse",
        force_undirected: bool = False,
) -> Tensor:
    """
    Samples random negative edges for multiple graphs based on `edge_index` and `batch`.

    Args:
        edge_index (Tensor): The edge indices of the graph.
        batch (Tensor or Tuple[Tensor, Tensor]): Batch vector to assign each node to a specific example.
        num_neg_samples (int, optional): The number of negative samples to return.
        method (str, optional): Sampling method, "sparse" or "dense". Controls memory/runtime trade-offs.
        force_undirected (bool, optional): If True, sampled negative edges will be undirected.

    Returns:
        Tensor: Batched negative edge indices.
    """
    src_batch, dst_batch = (batch, batch) if isinstance(batch, Tensor) else batch
    src_cumsum = paddle.cumsum(paddle.to_tensor([len(src_batch)]), axis=0)[:-1]

    num_nodes = paddle.unique(paddle.concat([src_batch, dst_batch]))
    ptr = src_cumsum if isinstance(batch, Tensor) else paddle.concat([src_cumsum, src_cumsum + len(dst_batch)])

    neg_edge_indices = []
    for i, edge_index_part in enumerate(paddle.split(edge_index, len(src_batch))):
        neg_edge_index = negative_sampling(edge_index_part - ptr[i], [len(src_batch), len(dst_batch)][i],
                                           num_neg_samples, method, force_undirected)
        neg_edge_indices.append(neg_edge_index + ptr[i])

    return paddle.concat(neg_edge_indices, axis=1)


def structured_negative_sampling(edge_index: Tensor, num_nodes: Optional[int] = None,
                                 contains_neg_self_loops: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Samples a negative edge for each positive edge in `edge_index` and returns as a tuple `(i, j, k)`.

    Args:
        edge_index (Tensor): The edge indices of the graph.
        num_nodes (int, optional): The number of nodes.
        contains_neg_self_loops (bool, optional): If False, sampled negative edges will not contain self loops.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Positive and negative edges `(i, j, k)`.
    """
    num_nodes = max(edge_index.flatten().numpy()) + 1 if num_nodes is None else num_nodes
    row, col = edge_index[:, 0], edge_index[:, 1]
    pos_idx = row * num_nodes + col

    rand = paddle.randint(0, num_nodes, (len(row),), dtype=paddle.int64)
    neg_idx = row * num_nodes + rand

    mask = paddle.to_tensor(np.isin(neg_idx.numpy(), pos_idx.numpy()), dtype=paddle.bool)
    rest = paddle.nonzero(mask).squeeze()
    while rest.numel() > 0:
        tmp = paddle.randint(0, num_nodes, (len(rest),), dtype=paddle.int64)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp
        mask = paddle.to_tensor(np.isin(neg_idx.numpy(), pos_idx.numpy()), dtype=paddle.bool)
        rest = rest[mask]

    return row, col, rand

def structured_negative_sampling_feasible(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    """
    Returns True if structured_negative_sampling is feasible
    on the graph given by edge_index.
    structured_negative_sampling is infeasible if at least one node
    is connected to all other nodes.

    Args:
        edge_index (Tensor): The edge indices.
        num_nodes (int, optional): The number of nodes, i.e.,
            max_val + 1 of edge_index. (default: None)
        contains_neg_self_loops (bool, optional): If set to False, sampled
            negative edges will not contain self loops. (default: True)

    Returns:
        bool: Whether structured negative sampling is feasible.

    Examples:
        >>> edge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False
        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    def maybe_num_nodes(edge_index, num_nodes=None):
        if num_nodes is not None:
            return num_nodes
        return int(paddle.max(edge_index) + 1)

    def coalesce(edge_index, num_nodes):
        sorted_indices = paddle.argsort(edge_index[0] * num_nodes + edge_index[1])
        edge_index = edge_index[:, sorted_indices]
        unique_indices = paddle.unique(edge_index, axis=1)
        return unique_indices

    def remove_self_loops(edge_index):
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        return edge_index

    def degree(src, num_nodes):
        return paddle.bincount(src, minlength=num_nodes)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if not contains_neg_self_loops:
        edge_index = remove_self_loops(edge_index)
        max_num_neighbors -= 1  # Reduce number of valid neighbors

    deg = degree(edge_index[0], num_nodes)
    # True if there exists no node that is connected to all other nodes.
    return bool(paddle.all(deg < max_num_neighbors))


def sample(population: int, k: int, device: Optional[str] = None) -> Tensor:
    """
    Samples `k` unique elements from the range `[0, population)`.

    Args:
        population (int): Total population size.
        k (int): Number of samples to draw.
        device (Optional[str]): Device to store the sampled elements.

    Returns:
        Tensor: A tensor of sampled elements.
    """
    return paddle.arange(population) if population <= k else paddle.to_tensor(random.sample(range(population), k))


def edge_index_to_vector(edge_index: Tensor, size: Tuple[int, int], bipartite: bool, force_undirected: bool = False) -> \
Tuple[Tensor, int]:
    """
    Converts an `edge_index` tensor to a flattened vector of unique indices.

    Args:
        edge_index (Tensor): Edge indices of the graph.
        size (Tuple[int, int]): Size of the graph.
        bipartite (bool): If True, treats the graph as bipartite.
        force_undirected (bool): If True, treats the edges as undirected.

    Returns:
        Tuple[Tensor, int]: Flattened indices and the total population.
    """
    row, col = edge_index[:, 0], edge_index[:, 1]

    if bipartite:
        idx = row * size[1] + col
        population = size[0] * size[1]
    elif force_undirected:
        num_nodes = size[0]
        mask = row < col
        offset = paddle.cumsum(paddle.ones([num_nodes - 1], dtype=row.dtype))[:-1]
        idx = (row[mask] * num_nodes + col[mask] - offset).astype(row.dtype)
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
    else:
        num_nodes = size[0]
        mask = row != col
        col[row < col] -= 1
        idx = row[mask] * (num_nodes - 1) + col[mask]
        population = num_nodes * (num_nodes - 1)

    return idx, population


def vector_to_edge_index(idx: Tensor, size: Tuple[int, int], bipartite: bool, force_undirected: bool = False) -> Tensor:
    """
    Converts a flattened index vector back to an `edge_index` format.

    Args:
        idx (Tensor): Flattened index vector.
        size (Tuple[int, int]): Size of the graph.
        bipartite (bool): If True, treats the graph as bipartite.
        force_undirected (bool): If True, treats the edges as undirected.

    Returns:
        Tensor: Edge index tensor.
    """
    if bipartite:
        row = idx // size[1]
        col = idx % size[1]
    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = paddle.cumsum(paddle.ones([num_nodes - 1], dtype=idx.dtype))[:-1]
        end = paddle.arange(num_nodes - 1, idx.shape[0] + 1, dtype=idx.dtype)
        row = paddle.bucketize(idx, end - offset, right=True)
        col = offset[row] + idx % num_nodes
        row, col = paddle.concat([row, col]), paddle.concat([col, row])
    else:
        num_nodes = size[0]
        row = idx // (num_nodes - 1)
        col = idx % (num_nodes - 1)
        col[row <= col] += 1

    return paddle.stack([row, col], axis=0)
