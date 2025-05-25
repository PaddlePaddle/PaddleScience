import paddle
from paddle import Tensor

from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import coalesce, degree
from paddle_geometric.utils._to_dense_adj import to_dense_adj


def assortativity(edge_index: Adj) -> float:
    r"""The degree assortativity coefficient from the
    `"Mixing patterns in networks"
    <https://arxiv.org/abs/cond-mat/0209450>`_ paper.
    Assortativity in a network refers to the tendency of nodes to
    connect with other similar nodes over dissimilar nodes.
    It is computed from Pearson correlation coefficient of the node degrees.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.

    Returns:
        The value of the degree assortativity coefficient for the input
        graph :math:`\in [-1, 1]`

    Example:
        >>> edge_index = paddle.to_tensor([[0, 1, 2, 3, 2],
        ...                                [1, 2, 0, 1, 3]])
        >>> assortativity(edge_index)
        -0.666667640209198
    """
    if isinstance(edge_index, SparseTensor):
        adj: SparseTensor = edge_index
        row, col, _ = adj.coo()
    else:
        assert isinstance(edge_index, Tensor)
        row, col = edge_index

    device = row.place
    out_deg = degree(row, dtype='int64')
    in_deg = degree(col, dtype='int64')
    degrees = paddle.unique(paddle.concat([out_deg, in_deg]))
    mapping = paddle.zeros([degrees.max().item() + 1], dtype=row.dtype)
    mapping = paddle.scatter(mapping, degrees, paddle.arange(degrees.shape[0], dtype=row.dtype))

    # Compute degree mixing matrix (joint probability distribution) `M`
    num_degrees = degrees.shape[0]
    src_deg = paddle.gather(mapping, out_deg[row])
    dst_deg = paddle.gather(mapping, in_deg[col])

    pairs = paddle.stack([src_deg, dst_deg], axis=0)
    occurrence = paddle.ones([pairs.shape[1]], dtype=row.dtype)
    pairs, occurrence = coalesce(pairs, occurrence)
    M = to_dense_adj(pairs, edge_attr=occurrence, max_num_nodes=num_degrees)[0]
    # normalization
    M /= M.sum()

    # Numeric assortativity coefficient, computed by Pearson correlation coefficient of the node degrees
    x = y = degrees.astype('float32')
    a, b = M.sum(axis=0), M.sum(axis=1)

    vara = (a * x**2).sum() - ((a * x).sum())**2
    varb = (b * x**2).sum() - ((b * x).sum())**2
    xy = paddle.outer(x, y)
    ab = paddle.outer(a, b)
    out = (xy * (M - ab)).sum() / (vara * varb).sqrt()
    return out.item()
