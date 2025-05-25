from typing import Optional

import paddle
from paddle import Tensor


def one_hot(
    index: Tensor,
    num_classes: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
) -> Tensor:
    r"""Takes a one-dimensional :obj:`index` tensor and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input tensor, in which case it will be :obj:`1`.

    Args:
        index (paddle.Tensor): The one-dimensional input tensor.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input tensor.
            (default: :obj:`None`)
        dtype (paddle.dtype, optional): The :obj:`dtype` of the output tensor.
    """
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    index = paddle.to_tensor(index)
    out = paddle.zeros((index.shape[0], num_classes), dtype=dtype)

    return out.put_along_axis_(indices=index.unsqueeze(1), values=paddle.ones([index.shape[0], 1], dtype=dtype), axis=1)
    # return out.scatter_(paddle.to_tensor(1), index.unsqueeze(1), paddle.ones([index.shape[0]], dtype=dtype))
