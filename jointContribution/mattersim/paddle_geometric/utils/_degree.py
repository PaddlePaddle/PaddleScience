from typing import Optional
import paddle
from paddle import Tensor

from paddle_geometric.utils.num_nodes import maybe_num_nodes


def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[paddle.dtype] = None) -> Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (Tensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`paddle.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = paddle.to_tensor([0, 1, 0, 2, 0], dtype='int64')
        >>> degree(row, dtype='int64')
        Tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = paddle.zeros((N, ), dtype=dtype if dtype is not None else index.dtype)
    one = paddle.ones((index.shape[0], ), dtype=out.dtype)
    return out.put_along_axis_(indices=index, values=one, axis=0, reduce='add', include_self=True)
