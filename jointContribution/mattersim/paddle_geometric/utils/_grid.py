from typing import Optional, Tuple
import paddle
from paddle import Tensor

from paddle_geometric.utils import coalesce


def grid(
    height: int,
    width: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the edge indices of a two-dimensional grid graph with height
    :attr:`height` and width :attr:`width` and its node positions.

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        dtype (paddle.dtype, optional): The desired data type of the returned
            position tensor. (default: :obj:`None`)
        device (str, optional): The desired device of the returned
            tensors. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """
    edge_index = grid_index(height, width, device)
    pos = grid_pos(height, width, dtype, device)
    return edge_index, pos


def grid_index(
    height: int,
    width: int,
    device: Optional[str] = None,
) -> Tensor:

    w = width
    kernel = paddle.to_tensor(
        [-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1],
        place=device,
    )

    row = paddle.arange(height * width, dtype="int64")
    row = row.reshape([-1, 1]).tile([1, kernel.shape[0]])
    col = row + kernel.reshape([1, -1])
    row, col = row.reshape([height, -1]), col.reshape([height, -1])
    index = paddle.arange(3, row.shape[1] - 3, dtype="int64")
    row, col = row[:, index].reshape([-1]), col[:, index].reshape([-1])

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = paddle.stack([row, col], axis=0)
    edge_index = coalesce(edge_index, num_nodes=height * width)
    return edge_index


def grid_pos(
    height: int,
    width: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tensor:

    dtype = paddle.float32 if dtype is None else dtype
    x = paddle.arange(width, dtype=dtype)
    y = (height - 1) - paddle.arange(height, dtype=dtype)

    x = x.tile([height])
    y = y.unsqueeze(-1).tile([1, width]).reshape([-1])

    return paddle.stack([x, y], axis=-1)
