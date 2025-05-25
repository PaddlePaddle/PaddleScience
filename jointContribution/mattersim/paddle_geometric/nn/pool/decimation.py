from typing import Union, Tuple, Any

import paddle
from paddle import Tensor
from paddle_geometric.utils import cumsum


def decimation_indices(
    ptr: Any,
    decimation_factor: Union[int, float],
) -> Tuple[Tensor, Any]:
    """Gets indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have at least
    one node after decimation.

    Args:
        ptr (LongTensor): The indices of samples in the batch.
        decimation_factor (int or float): The value to divide number of nodes
            with. Should be higher than (or equal to) :obj:`1` for
            downsampling.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`): The indices and
        updated :obj:`ptr` after downsampling.
    """
    if decimation_factor < 1:
        raise ValueError(
            f"The argument `decimation_factor` should be higher than (or "
            f"equal to) 1 for downsampling. (got {decimation_factor})")

    batch_size = ptr.size(0) - 1
    count = ptr[1:] - ptr[:-1]
    decim_count = paddle.floor(count.astype('float') / decimation_factor).astype('int')
    decim_count = paddle.maximum(decim_count, paddle.to_tensor([1]))  # Prevent empty examples.

    decim_indices = [
        ptr[i] + paddle.argsort(paddle.rand(count[i], device=ptr.device))[:decim_count[i]]
        for i in range(batch_size)
    ]
    decim_indices = paddle.concat(decim_indices, axis=0)

    # Get updated ptr (e.g., for future decimations):
    decim_ptr = cumsum(decim_count)

    return decim_indices, decim_ptr
