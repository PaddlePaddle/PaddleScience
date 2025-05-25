import warnings

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import EdgeIndex
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import scatter

def spmm(
    src: Adj,
    other: Tensor,
    reduce: str = 'sum',
) -> Tensor:
    r"""Matrix product of sparse matrix with dense matrix.

    Args:
        src (paddle.Tensor or paddle_sparse.SparseTensor or EdgeIndex):
            The input sparse matrix which can be a
            :pyg:`paddle_geometric` :class:`paddle_sparse.SparseTensor`,
            a :paddle:`Paddle` :class:`paddle.sparse.Tensor` or
            a :pyg:`paddle_geometric` :class:`EdgeIndex`.
        other (paddle.Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    reduce = 'sum' if reduce == 'add' else reduce

    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError(f"`reduce` argument '{reduce}' not supported")

    if isinstance(src, EdgeIndex):
        return src.matmul(other=other, reduce=reduce)

    if isinstance(src, SparseTensor):
        if src.nnz() == 0:
            return paddle.zeros([src.shape[0], other.shape[1]], dtype=other.dtype)

        # Use Paddle's sparse mm if available
        if paddle.version.full_version >= "2.0.0" and other.ndim == 2:
            return paddle.sparse.matmul(src, other)

        return paddle.sparse.matmul(src, other)

    if not isinstance(src, paddle.sparse.coo_tensor):
        raise ValueError("'src' must be a 'paddle.sparse.SparseTensor' or a 'paddle.sparse.coo_tensor'")

    # If reducing by "sum"
    if reduce == 'sum':
        return paddle.sparse.matmul(src, other)

    # Handle "mean" reduction by dividing by the degree:
    if reduce == 'mean':
        if isinstance(src, paddle.sparse.csr_tensor):
            ptr = src.crow_indices()
            deg = ptr[1:] - ptr[:-1]
        else:  # Assuming COO format
            src = src.coalesce()
            ones = paddle.ones_like(src.values())
            index = src.indices()[0]
            deg = scatter(ones, index, 0, dim_size=src.shape[0], reduce='sum')

        return paddle.sparse.matmul(src, other) / deg.reshape([-1, 1]).clip(min=1)

    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"'paddle.sparse.Tensor' on device '{src.place}'")
