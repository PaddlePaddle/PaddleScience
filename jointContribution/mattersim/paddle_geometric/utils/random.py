from itertools import chain
from typing import List, Union

import numpy as np
import paddle

from paddle_geometric.utils import remove_self_loops, to_undirected


def erdos_renyi_graph(
    num_nodes: int,
    edge_prob: float,
    directed: bool = False,
) -> paddle.Tensor:
    r"""Returns the :obj:`edge_index` of a random Erdos-Renyi graph.

    Args:
        num_nodes (int): The number of nodes.
        edge_prob (float): Probability of an edge.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)
    """
    if directed:
        idx = paddle.arange((num_nodes - 1) * num_nodes)
        idx = idx.reshape([num_nodes - 1, num_nodes])
        idx = idx + paddle.arange(1, num_nodes).reshape([-1, 1])
        idx = idx.reshape([-1])
    else:
        idx = paddle.combinations(paddle.arange(num_nodes), r=2)

    # Filter edges.
    mask = paddle.rand([idx.shape[0]]) < edge_prob
    idx = idx[mask]

    if directed:
        row = idx // num_nodes
        col = idx % num_nodes
        edge_index = paddle.stack([row, col], axis=0)
    else:
        edge_index = to_undirected(idx.T, num_nodes=num_nodes)

    return edge_index


def stochastic_blockmodel_graph(
    block_sizes: Union[List[int], paddle.Tensor],
    edge_probs: Union[List[List[float]], paddle.Tensor],
    directed: bool = False,
) -> paddle.Tensor:
    r"""Returns the :obj:`edge_index` of a stochastic blockmodel graph.

    Args:
        block_sizes ([int] or paddle.Tensor): The sizes of blocks.
        edge_probs ([[float]] or paddle.Tensor): The density of edges going
            from each block to each other block. Must be symmetric if the
            graph is undirected.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)
    """
    size, prob = block_sizes, edge_probs

    if not isinstance(size, paddle.Tensor):
        size = paddle.to_tensor(size, dtype='int64')
    if not isinstance(prob, paddle.Tensor):
        prob = paddle.to_tensor(prob, dtype='float32')

    assert size.ndim == 1
    assert prob.ndim == 2 and prob.shape[0] == prob.shape[1]
    assert size.shape[0] == prob.shape[0]
    if not directed:
        assert paddle.allclose(prob, prob.T)

    node_idx = paddle.concat([paddle.full([b], i, dtype='int64') for i, b in enumerate(size)])
    num_nodes = node_idx.shape[0]

    if directed:
        idx = paddle.arange((num_nodes - 1) * num_nodes)
        idx = idx.reshape([num_nodes - 1, num_nodes])
        idx = idx + paddle.arange(1, num_nodes).reshape([-1, 1])
        idx = idx.reshape([-1])
        row = idx // num_nodes
        col = idx % num_nodes
    else:
        row, col = paddle.combinations(paddle.arange(num_nodes), r=2).T

    mask = paddle.bernoulli(prob[node_idx[row], node_idx[col]]).astype('bool')
    edge_index = paddle.stack([row[mask], col[mask]], axis=0)

    if not directed:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index


def barabasi_albert_graph(num_nodes: int, num_edges: int) -> paddle.Tensor:
    r"""Returns the :obj:`edge_index` of a Barabasi-Albert preferential
    attachment model, where a graph of :obj:`num_nodes` nodes grows by
    attaching new nodes with :obj:`num_edges` edges that are preferentially
    attached to existing nodes with high degree.

    Args:
        num_nodes (int): The number of nodes.
        num_edges (int): The number of edges from a new node to existing nodes.
    """
    assert num_edges > 0 and num_edges < num_nodes

    row = paddle.arange(num_edges, dtype='int64')
    col = paddle.randperm(num_edges)

    for i in range(num_edges, num_nodes):
        row = paddle.concat([row, paddle.full([num_edges], i, dtype='int64')])
        choice = np.random.choice(paddle.concat([row, col]).numpy(), num_edges)
        col = paddle.concat([col, paddle.to_tensor(choice)])

    edge_index = paddle.stack([row, col], axis=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index
