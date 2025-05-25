from typing import Optional

from paddle import Tensor

import paddle_geometric.typing

# Check if the paddle equivalent for graph clustering is available

graclus_cluster = None


def graclus(edge_index: Tensor, weight: Optional[Tensor] = None,
            num_nodes: Optional[int] = None):
    r"""A greedy clustering algorithm from the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper of picking an unmarked
    vertex and matching it with one of its unmarked neighbors (that maximizes
    its edge weight).
    The GPU algorithm is adapted from the `"A GPU Algorithm for Greedy Graph
    Matching" <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`_
    paper.

    Args:
        edge_index (paddle.Tensor): The edge indices.
        weight (paddle.Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`paddle.Tensor`
    """
    if graclus_cluster is None:
        raise ImportError('`graclus` requires `paddle-cluster`.')

    return graclus_cluster(edge_index[0], edge_index[1], weight, num_nodes)
