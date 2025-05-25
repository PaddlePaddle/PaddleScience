import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.typing import paddle_scatter


def segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    r"""Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. See the `documentation
    <https://paddle-scatter.readthedocs.io/en/latest/functions/
    segment_csr.html>`__ of the :obj:`paddle_scatter` package for more
    information.

    Args:
        src (paddle.Tensor): The source tensor.
        ptr (paddle.Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.size(0)`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    if not paddle_geometric.typing.WITH_PADDLE_SCATTER or is_compiling():
        return _paddle_segment(src, ptr, reduce)

    if (ptr.ndim == 1 and paddle_geometric.typing.WITH_PADDLE2 and src.is_gpu()
            and reduce == 'mean'):
        return _paddle_segment(src, ptr, reduce)

    return paddle_scatter.segment_csr(src, ptr, reduce=reduce)


def _paddle_segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    if not paddle_geometric.typing.WITH_PADDLE2:
        raise ImportError("'segment' requires the 'paddle-scatter' package")
    if ptr.ndim > 1:
        raise ImportError("'segment' in an arbitrary dimension "
                          "requires the 'paddle-scatter' package")

    if reduce in ['min', 'max']:
        reduce_func = paddle.min if reduce == 'min' else paddle.max
        initial = None
    elif reduce == 'mean':
        reduce_func = paddle.mean
        initial = 0
    else:
        reduce_func = paddle.sum
        initial = 0

    segments = []
    for i in range(ptr.shape[0] - 1):
        start, end = ptr[i].item(), ptr[i + 1].item()
        segment = src[start:end]
        if initial is not None:
            segment = paddle.where(segment.isinf(), paddle.to_tensor(initial, dtype=segment.dtype), segment)
        segments.append(reduce_func(segment, axis=0))

    return paddle.stack(segments)
