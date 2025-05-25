from typing import List

import numpy as np
import paddle
from paddle import Tensor


def lexsort(
    keys: List[Tensor],
    axis: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([paddle.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        axis (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    # Convert tensors to numpy arrays for `np.lexsort` functionality
    keys = [k.numpy() for k in keys]
    if descending:
        keys = [-k for k in keys]

    # Perform lexicographical sort using numpy
    out = np.lexsort(keys[::-1], axis=axis)

    return paddle.to_tensor(out, dtype='int64')
