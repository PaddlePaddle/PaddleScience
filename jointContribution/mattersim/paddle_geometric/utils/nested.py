from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.utils import scatter


def to_nested_tensor(
    x: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Given a contiguous batch of tensors
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`
    (with :math:`N_i` indicating the number of elements in example :math:`i`),
    creates a `nested Paddle tensor`.
    Reverse operation of :meth:`from_nested_tensor`.

    Args:
        x (paddle.Tensor): The input tensor
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`.
        batch (paddle.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            element to a specific example. Must be ordered.
            (default: :obj:`None`)
        ptr (paddle.Tensor, optional): Alternative representation of
            :obj:`batch` in compressed format. (default: :obj:`None`)
        batch_size (int, optional): The batch size :math:`B`.
            (default: :obj:`None`)
    """
    if ptr is not None:
        offsets = ptr[1:] - ptr[:-1]
        sizes = offsets.tolist()
        xs = list(paddle.split(x, sizes, axis=0))
    elif batch is not None:
        offsets = scatter(paddle.ones_like(batch), batch, dim_size=batch_size)
        sizes = offsets.tolist()
        xs = list(paddle.split(x, sizes, axis=0))
    else:
        xs = [x]

    # This currently copies the data, although `x` is already contiguous.
    # Sadly, there does not exist any (public) API to prevent this :(
    return paddle.to_tensor(xs)


def from_nested_tensor(
    x: Tensor,
    return_batch: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Given a `nested Paddle tensor`, creates a contiguous
    batch of tensors
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`, and
    optionally a batch vector which assigns each element to a specific example.
    Reverse operation of :meth:`to_nested_tensor`.

    Args:
        x (paddle.Tensor): The nested input tensor. The size of nested tensors
            need to match except for the first dimension.
        return_batch (bool, optional): If set to :obj:`True`, will also return
            the batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`.
            (default: :obj:`False`)
    """
    if not isinstance(x, list):
        raise ValueError("Input tensor in 'from_nested_tensor' is not nested")

    sizes = paddle.to_tensor([t.shape for t in x])

    for dim, (a, b) in enumerate(zip(sizes[0, 1:], sizes[:, 1:].t())):
        if not paddle.all(paddle.equal(a.expand(b.shape), b)):
            raise ValueError(f"Not all nested tensors have the same size "
                             f"in dimension {dim + 1} "
                             f"(expected size {a.item()} for all tensors)")

    out = paddle.concat([t.flatten() for t in x])
    out = out.reshape([-1] + sizes[0, 1:].tolist())

    if not return_batch:
        return out

    batch = paddle.arange(len(x), dtype='int64')
    batch = batch.repeat_interleave(sizes[:, 0])

    return out, batch
