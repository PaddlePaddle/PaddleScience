from typing import Optional

from paddle import Tensor

from paddle_geometric.utils import degree


def normalized_cut(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Computes the normalized cut :math:`\mathbf{e}_{i,j} \cdot
    \left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)` of a weighted graph
    given by edge indices and edge attributes.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor): Edge weights or multi-dimensional edge features.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Example:
        >>> edge_index = paddle.to_tensor([[1, 1, 2, 3],
        ...                                [3, 3, 1, 2]], dtype="int64")
        >>> edge_attr = paddle.to_tensor([1., 1., 1., 1.])
        >>> normalized_cut(edge_index, edge_attr)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [1.5, 1.5, 2.0, 1.5])
    """
    row, col = edge_index[0], edge_index[1]
    deg = 1. / degree(col, num_nodes, dtype=edge_attr.dtype)
    deg = deg[row] + deg[col]
    cut = edge_attr * deg
    return cut
