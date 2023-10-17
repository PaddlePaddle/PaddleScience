from typing import Optional, Tuple

import paddle

def broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_add_(dim, index, src, x):
    # for i in range(len(index)):
    #     x[index[i]] += src[i]
    if x.dim()==1:
        output = paddle.scatter_nd_add(x.unsqueeze(-1), index.unsqueeze(-1), src.unsqueeze(-1)).squeeze(-1)
    else:
        i, j = index.shape
        grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
        # 若 PyTorch 的 dim 取 0
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
        # 若 PyTorch 的 dim 取 1
        # index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
        # PaddlePaddle updates 的 shape 大小必须与 index 对应
        updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        updates = paddle.gather_nd(src, index=updates_index)
        output = paddle.scatter_nd_add(x, index, updates)
    return output


def scatter_sum(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                out: Optional[paddle.Tensor] = None,
                dim_size: Optional[int] = None) -> paddle.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = paddle.zeros(size, dtype=src.dtype)
        return scatter_add_(0, index, src, out)
    else:
        return scatter_add_(0, index, src, out)


def scatter_add(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                out: Optional[paddle.Tensor] = None,
                dim_size: Optional[int] = None) -> paddle.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                out: Optional[paddle.Tensor] = None,
                dim_size: Optional[int] = None) -> paddle.Tensor:
    return paddle.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                 out: Optional[paddle.Tensor] = None,
                 dim_size: Optional[int] = None) -> paddle.Tensor:

    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = paddle.ones(index.size(), dtype=src.dtype, place=src.place)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)
    count = broadcast(count, out, dim)
    if paddle.is_floating_point(out):
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out


def scatter_min(
        src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
        out: Optional[paddle.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[paddle.Tensor, paddle.Tensor]:
    return paddle.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
        out: Optional[paddle.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[paddle.Tensor, paddle.Tensor]:
    return paddle.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
            out: Optional[paddle.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> paddle.Tensor:
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1` in ascending order.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import scatter

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])

        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError
