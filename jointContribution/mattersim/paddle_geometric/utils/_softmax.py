from typing import Optional

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.typing import pyg_lib
from paddle_geometric.utils import scatter, segment
from paddle_geometric.utils.num_nodes import maybe_num_nodes


def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    axis: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (Tensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (Tensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        axis (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    if (ptr is not None and src.place.is_cpu_place()
            and paddle_geometric.typing.WITH_SOFTMAX
            and not is_compiling()):  # pragma: no cover
        return pyg_lib.ops.softmax_csr(src, ptr, axis)

    if (ptr is not None and
        (ptr.dim() == 1 or (ptr.dim() > 1 and index is None) or
         (paddle_geometric.typing.WITH_PADDLE_SCATTER and not is_compiling()))):

        axis = axis + src.dim() if axis < 0 else axis
        size = ([1] * axis) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.reshape(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = paddle.repeat_interleave(src_max, repeats=count, axis=axis)
        out = paddle.exp(src - src_max)
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = paddle.repeat_interleave(out_sum, repeats=count, axis=axis)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, axis, dim_size=N, reduce='max')
        out = src - paddle.index_select(src_max, index=index, axis=axis)
        out = paddle.exp(out)
        out_sum = scatter(out, index, axis, dim_size=N, reduce='sum') + 1e-16
        out_sum = paddle.index_select(out_sum, index=index, axis=axis)
    else:
        raise NotImplementedError("'softmax' requires 'index' to be specified")

    return out / out_sum
