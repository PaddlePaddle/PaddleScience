from typing import Any, List, Union

import paddle
from paddle import Tensor

from paddle_geometric.typing import TensorFrame
from paddle_geometric.utils.mask import mask_select
from paddle_geometric.utils.sparse import is_paddle_sparse_tensor


def select(
    src: Union[Tensor, List[Any], TensorFrame],
    index_or_mask: Tensor,
    dim: int,
) -> Union[Tensor, List[Any]]:
    r"""Selects the input tensor or input list according to a given index or
    mask vector.

    Args:
        src (paddle.Tensor or list): The input tensor or list.
        index_or_mask (paddle.Tensor): The index or mask vector.
        dim (int): The dimension along which to select.
    """
    if isinstance(src, Tensor):
        if index_or_mask.dtype == paddle.bool:
            return mask_select(src, dim, index_or_mask)
        return paddle.index_select(src, index=index_or_mask, axis=dim)

    if isinstance(src, (tuple, list)):
        if dim != 0:
            raise ValueError("Cannot select along dimension other than 0")
        if index_or_mask.dtype == paddle.bool:
            return [src[i] for i, m in enumerate(index_or_mask.numpy()) if m]
        return [src[i] for i in index_or_mask.numpy()]

    if isinstance(src, TensorFrame):
        assert dim == 0
        if index_or_mask.dtype == paddle.bool:
            return mask_select(src, dim, index_or_mask)
        return src[index_or_mask.numpy()]

    raise ValueError(f"Encountered invalid input type (got '{type(src)}')")


def narrow(src: Union[Tensor, List[Any]], dim: int, start: int,
           length: int) -> Union[Tensor, List[Any]]:
    r"""Narrows the input tensor or input list to the specified range.

    Args:
        src (paddle.Tensor or list): The input tensor or list.
        dim (int): The dimension along which to narrow.
        start (int): The starting dimension.
        length (int): The distance to the ending dimension.
    """
    if isinstance(src, Tensor) and is_paddle_sparse_tensor(src):
        # Paddle currently does not fully support sparse tensor narrowing.
        index = paddle.arange(start, start + length, dtype='int64')
        return paddle.index_select(src, index=index, axis=dim)

    if isinstance(src, Tensor):
        return src.slice(dim, start, start + length)

    if isinstance(src, list):
        if dim != 0:
            raise ValueError("Cannot narrow along dimension other than 0")
        return src[start:start + length]

    raise ValueError(f"Encountered invalid input type (got '{type(src)}')")
