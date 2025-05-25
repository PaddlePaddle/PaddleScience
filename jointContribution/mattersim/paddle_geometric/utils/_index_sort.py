from typing import Optional, Tuple

import paddle
from paddle import Tensor


def index_sort(
    inputs: paddle.Tensor,
    max_value: Optional[int] = None,
    stable: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Sorts the elements of the :obj:`inputs` tensor in ascending order.
    It is expected that :obj:`inputs` is one-dimensional and that it only
    contains positive integer values. If :obj:`max_value` is given, it can
    be used by the underlying algorithm for better performance.

    Args:
        inputs (Tensor): A vector with positive integer values.
        max_value (int, optional): The maximum value stored inside
            :obj:`inputs`. This value can be an estimation, but needs to be
            greater than or equal to the real maximum.
            (default: :obj:`None`)
        stable (bool, optional): Makes the sorting routine stable, which
            guarantees that the order of equivalent elements is preserved.
            (default: :obj:`False`)
    """
    if stable:
        # Perform stable sort if requested
        indices = paddle.argsort(inputs, axis=0, descending=False, stable=True)
        sorted_inputs = paddle.gather(inputs, indices)
    else:
        # Perform regular sort
        indices = paddle.argsort(inputs, axis=0, descending=False)
        sorted_inputs = paddle.gather(inputs, indices)

    return sorted_inputs, indices

