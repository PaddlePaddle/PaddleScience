import warnings
from typing import Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor


def is_integer_dtype(dtype):
    """
    Checks if the given dtype is an integer type.

    Args:
        dtype (paddle.dtype): The data type to check.

    Returns:
        bool: True if the dtype is an integer type, False otherwise.
    """
    return dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]


def map_index(
    src: Tensor,
    index: Tensor,
    max_index: Optional[Union[int, Tensor]] = None,
    inclusive: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Maps indices in `src` to the positional value of their
    corresponding occurrence in `index`.
    Indices must be strictly positive.

    Args:
        src (paddle.Tensor): The source tensor to map.
        index (paddle.Tensor): The index tensor that denotes the new mapping.
        max_index (int, optional): The maximum index value.
            (default :obj:`None`)
        inclusive (bool, optional): If set to True, it is assumed that
            every entry in `src` has a valid entry in `index`.
            Can speed-up computation. (default: `False`)

    Returns:
        Tuple[paddle.Tensor, Optional[paddle.Tensor]]

    Example:
        >>> src = paddle.to_tensor([2, 0, 1, 0, 3], dtype='int64')
        >>> index = paddle.to_tensor([3, 2, 0, 1], dtype='int64')
        >>> map_index(src, index)
        (Tensor([1, 2, 3, 2, 0], dtype=int64), Tensor([True, True, True, True, True]))

        >>> src = paddle.to_tensor([2, 0, 1, 0, 3], dtype='int64')
        >>> index = paddle.to_tensor([3, 2, 0], dtype='int64')
        >>> map_index(src, index)
        (Tensor([1, 2, -1, 2, 0], dtype=int64), Tensor([True, True, False, True, True]))
    """
    if not is_integer_dtype(src.dtype) or not is_integer_dtype(index.dtype):
        raise ValueError("Expected 'src' and 'index' to be integer tensors.")

    # if src.place != index.place:
    #     raise ValueError(f"'src' and 'index' must be on the same device. all in gpu:0")

    if max_index is None:
        max_index = max(src.max(), index.max()).item()

    # Memory-efficient method if `max_index` is within threshold
    THRESHOLD = 40_000_000 if src.place.is_gpu_place() else 10_000_000
    if max_index <= THRESHOLD:
        assoc = paddle.full((max_index + 1,), -1, dtype=src.dtype)
        assoc = paddle.scatter(assoc, index, paddle.arange(index.shape[0], dtype=src.dtype))

        out = paddle.gather(assoc, src)
        if inclusive:
            if paddle.any(out == -1):
                raise ValueError("Found invalid entries in 'src' that do not have a corresponding entry in 'index'.")
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    # CPU-based fallback using pandas
    try:
        import pandas as pd
        left_ser = pd.Series(src.numpy(), name='left_ser')
        right_ser = pd.Series(
            index.numpy(),
            index=np.arange(index.shape[0]),
            name='right_ser',
        )
        result = pd.merge(left_ser, right_ser, how='left', left_on='left_ser', right_index=True)
        out = paddle.to_tensor(result['right_ser'].fillna(-1).values, place=src.place, dtype=src.dtype)

        if inclusive:
            if paddle.any(out == -1):
                raise ValueError("Found invalid entries in 'src' that do not have a corresponding entry in 'index'.")
            return out, None
        else:
            mask = out != -1
            return out[mask], mask
    except ImportError:
        warnings.warn("Install 'pandas' for better performance.")
        raise
