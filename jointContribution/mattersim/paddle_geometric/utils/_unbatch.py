from typing import List, Optional

import paddle
from paddle import Tensor

from paddle_geometric.utils import cumsum, degree


def unbatch(
    src: Tensor,
    batch: Tensor,
    dim: int = 0,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (Tensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> src = paddle.arange(7)
        >>> batch = paddle.to_tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (Tensor([0, 1, 2]), Tensor([3, 4]), Tensor([5, 6]))
    """
    sizes = degree(batch, batch_size, dtype='int64').tolist()
    return paddle.split(src, num_or_sections=sizes, axis=dim)


def unbatch_edge_index(
    edge_index: Tensor,
    batch: Tensor,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (Tensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                                 [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (Tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        Tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, batch_size, dtype='int64')
    ptr = cumsum(deg)

    edge_batch = paddle.gather(batch, edge_index[0])
    edge_index = edge_index - paddle.gather(ptr, edge_batch)
    sizes = degree(edge_batch, batch_size, dtype='int64').numpy().tolist()
    return paddle.split(edge_index, sizes, axis=1)
