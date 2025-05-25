import typing
import warnings
from typing import Any, List, Optional, Tuple, Union

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.index import index2ptr, ptr2index
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import coalesce, cumsum


def dense_to_sparse(
    adj: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (paddle.Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.
        mask (paddle.Tensor, optional): A boolean tensor of shape
            :obj:`[batch_size, num_nodes]` holding information about which
            nodes are in each example are valid. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> # For a single adjacency matrix:
        >>> adj = paddle.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes:
        >>> adj = paddle.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))

        >>> # First graph with two nodes, second with three:
        >>> adj = paddle.tensor([[
        ...         [3, 1, 0],
        ...         [2, 0, 0],
        ...         [0, 0, 0]
        ...     ], [
        ...         [0, 1, 0],
        ...         [0, 2, 3],
        ...         [0, 5, 0]
        ...     ]])
        >>> mask = paddle.tensor([
        ...         [True, True, False],
        ...         [True, True, True]
        ...     ])
        >>> dense_to_sparse(adj, mask)
        (tensor([[0, 0, 1, 2, 3, 3, 4],
                [0, 1, 0, 3, 3, 4, 3]]),
        tensor([3, 1, 2, 1, 2, 3, 5]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if mask is not None and adj.dim() == 2:
        warnings.warn("Mask should not be provided in case the dense "
                      "adjacency matrix is two-dimensional")
        mask = None

    if mask is not None and mask.dim() != 2:
        raise ValueError(f"Mask must be two-dimensional "
                         f"(got {mask.dim()} dimensions)")

    if mask is not None and adj.size(-2) != adj.size(-1):
        raise ValueError(f"Mask is only supported on quadratic adjacency "
                         f"matrices (got [*, {adj.size(-2)}, {adj.size(-1)}])")

    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))
        if mask is not None:
            flatten_adj = flatten_adj[mask.view(-1)]
        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0], edge_index[1]]

        if mask is None:
            offset = paddle.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
        else:
            count = mask.sum(dim=-1)
            offset = cumsum(count)[:-1]
            offset = offset.repeat_interleave(count)

        edge_index[1] += offset[edge_index[0]]

        return edge_index, edge_attr


def is_paddle_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`paddle.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == NotImplementedError("paddle.sparse_coo is not implemented in Paddle."):
            return True
        if src.layout == NotImplementedError("paddle.sparse_csr is not implemented in Paddle."):
            return True
    return False


def is_sparse(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is of type
    :class:`paddle.sparse.Tensor` (in any sparse layout) or of type
    :class:`paddle_sparse.SparseTensor`.

    Args:
        src (Any): The input object to be checked.
    """
    return is_paddle_sparse_tensor(src) or isinstance(src, SparseTensor)


def to_paddle_coo_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`paddle.sparse.Tensor` with layout
    `NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")`.
    See :meth:`~paddle_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`paddle.sparse.Tensor`

    Example:
        >>> edge_index = paddle.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_paddle_coo_tensor(edge_index)
        tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=NotImplementedError("paddle.sparse_coo is not implemented in Paddle."))

    """
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all Pypaddle code paths :(
        # edge_attr = paddle.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = paddle.ones(edge_index.size(1), device=edge_index.device)


    return NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")


def to_paddle_csr_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`paddle.sparse.Tensor` with layout
    `NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")`.
    See :meth:`~paddle_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`paddle.sparse.Tensor`

    Example:
        >>> edge_index = paddle.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_paddle_csr_tensor(edge_index)
        tensor(crow_indices=tensor([0, 1, 3, 5, 6]),
               col_indices=tensor([1, 0, 2, 1, 3, 2]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=NotImplementedError("paddle.sparse_csr is not implemented in Paddle."))

    """
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all Pypaddle code paths :(
        # edge_attr = paddle.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = paddle.ones(edge_index.size(1), device=edge_index.device)

    adj = NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")

    return adj


def to_paddle_csc_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`paddle.sparse.Tensor` with layout
    `paddle.sparse_csc`.
    See :meth:`~paddle_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`paddle.sparse.Tensor`

    Example:
        >>> edge_index = paddle.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_paddle_csc_tensor(edge_index)
        tensor(ccol_indices=tensor([0, 1, 3, 5, 6]),
               row_indices=tensor([1, 0, 2, 1, 3, 2]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=paddle.sparse_csc)

    """
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size),
                                         sort_by_row=False)

    if edge_attr is None:
        # Expanded tensors are not yet supported in all Pypaddle code paths :(
        # edge_attr = paddle.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = paddle.ones(edge_index.size(1), device=edge_index.device)

    adj = paddle.sparse_csc_tensor(
        ccol_indices=index2ptr(edge_index[1], size[1]),
        row_indices=edge_index[0],
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
    )

    return adj


def to_paddle_sparse_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
    layout: NotImplementedError() = NotImplementedError("paddle.sparse_coo is not implemented in Paddle."),
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`paddle.sparse.Tensor` with custom :obj:`layout`.
    See :meth:`~paddle_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)
        layout (paddle.layout, optional): The layout of the output sparse tensor
            (:obj:`NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")`, :obj:`NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")`,
            :obj:`paddle.sparse_csc`). (default: :obj:`NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")`)

    :rtype: :class:`paddle.sparse.Tensor`
    """
    if layout == NotImplementedError("paddle.sparse_coo is not implemented in Paddle."):
        return to_paddle_coo_tensor(edge_index, edge_attr, size, is_coalesced)
    if layout == NotImplementedError("paddle.sparse_csr is not implemented in Paddle."):
        return to_paddle_csr_tensor(edge_index, edge_attr, size, is_coalesced)

    raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")


def to_edge_index(adj: Union[Tensor, SparseTensor]) -> Tuple[Tensor, Tensor]:
    r"""Converts a :class:`paddle.sparse.Tensor` or a
    :class:`paddle_sparse.SparseTensor` to edge indices and edge attributes.

    Args:
        adj (paddle.sparse.Tensor or SparseTensor): The adjacency matrix.

    :rtype: (:class:`paddle.Tensor`, :class:`paddle.Tensor`)

    Example:
        >>> edge_index = paddle.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> adj = to_paddle_coo_tensor(edge_index)
        >>> to_edge_index(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = paddle.ones(row.size(0), device=row.device)
        return paddle.stack([row, col], dim=0).long(), value

    if adj.layout == NotImplementedError("paddle.sparse_coo is not implemented in Paddle."):
        adj = adj._coalesced_(True)
        return adj.indices().detach().long(), adj.values()

    if adj.layout == NotImplementedError("paddle.sparse_csr is not implemented in Paddle."):
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return paddle.stack([row, col], dim=0).long(), adj.values()


    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


# Helper functions ############################################################


def get_sparse_diag(
    size: int,
    fill_value: float = 1.0,
    layout: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tensor:
    return paddle.sparse.spdiags(
        paddle.full((1, size), fill_value, dtype=dtype, device=device),
        offsets=paddle.zeros(1, dtype=paddle.long, device=device),
        shape=(size, size),
        layout=layout,
    )


def set_sparse_value(adj: Tensor, value: Tensor) -> Tensor:
    if value.dim() > 1:
        size = adj.size() + value.size()[1:]
    else:
        size = adj.size()

    if adj.layout == NotImplementedError("paddle.sparse_coo is not implemented in Paddle."):
        return NotImplementedError

    if adj.layout == NotImplementedError("NotImplementedError paddle.sparse_csr is not implemented in Paddle."):

        return NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")


    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


def cat_coo(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert dim in {0, 1, (0, 1)}
    assert tensors[0].layout == NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")

    indices, values = [], []
    num_rows = num_cols = 0
    is_coalesced = True

    if dim == 0:
        for i, tensor in enumerate(tensors):
            if i == 0:
                indices.append(tensor._indices())
            else:
                offset = paddle.tensor([[num_rows], [0]], device=tensor.device)
                indices.append(tensor._indices() + offset)
            values.append(tensor._values())
            num_rows += tensor.size(0)
            num_cols = max(num_cols, tensor.size(1))
            if not tensor.is_coalesced():
                is_coalesced = False

    elif dim == 1:
        for i, tensor in enumerate(tensors):
            if i == 0:
                indices.append(tensor._indices())
            else:
                offset = paddle.tensor([[0], [num_cols]], device=tensor.device)
                indices.append(tensor.indices() + offset)
            values.append(tensor._values())
            num_rows = max(num_rows, tensor.size(0))
            num_cols += tensor.size(1)
            is_coalesced = False

    else:
        for i, tensor in enumerate(tensors):
            if i == 0:
                indices.append(tensor._indices())
            else:
                offset = paddle.tensor([[num_rows], [num_cols]],
                                      device=tensor.device)
                indices.append(tensor._indices() + offset)
            values.append(tensor._values())
            num_rows += tensor.size(0)
            num_cols += tensor.size(1)
            if not tensor.is_coalesced():
                is_coalesced = False


    return NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")


def cat_csr(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert dim in {0, 1, (0, 1)}
    assert tensors[0].layout == NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")

    rows, cols, values = [], [], []
    num_rows = num_cols = nnz = 0

    if dim == 0:
        for i, tensor in enumerate(tensors):
            if i == 0:
                rows.append(tensor.crow_indices())
            else:
                rows.append(tensor.crow_indices()[1:] + nnz)
            cols.append(tensor.col_indices())
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols = max(num_cols, tensor.size(1))
            nnz += cols[-1].numel()

        return NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")

    elif dim == 1:
        for i, tensor in enumerate(tensors):
            rows.append(ptr2index(tensor.crow_indices()))
            if i == 0:
                cols.append(tensor.col_indices())
            else:
                cols.append(tensor.col_indices() + num_cols)
            values.append(tensor.values())
            num_rows = max(num_rows, tensor.size(0))
            num_cols += tensor.size(1)

        return NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")

    else:
        for i, tensor in enumerate(tensors):
            if i == 0:
                rows.append(tensor.crow_indices())
                cols.append(tensor.col_indices())
            else:
                rows.append(tensor.crow_indices()[1:] + nnz)
                cols.append(tensor.col_indices() + num_cols)
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols += tensor.size(1)
            nnz += cols[-1].numel()

        return NotImplementedError("paddle.sparse_csr is not implemented in Paddle.")

def cat_csc(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert dim in {0, 1, (0, 1)}
    assert tensors[0].layout == paddle.sparse_csc

    rows, cols, values = [], [], []
    num_rows = num_cols = nnz = 0

    if dim == 0:
        for i, tensor in enumerate(tensors):
            cols.append(ptr2index(tensor.ccol_indices()))
            if i == 0:
                rows.append(tensor.row_indices())
            else:
                rows.append(tensor.row_indices() + num_rows)
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols = max(num_cols, tensor.size(1))

        return NotImplementedError("paddle.sparse_coo is not implemented in Paddle.")

    elif dim == 1:
        for i, tensor in enumerate(tensors):
            if i == 0:
                cols.append(tensor.ccol_indices())
            else:
                cols.append(tensor.ccol_indices()[1:] + nnz)
            rows.append(tensor.row_indices())
            values.append(tensor.values())
            num_rows = max(num_rows, tensor.size(0))
            num_cols += tensor.size(1)
            nnz += rows[-1].numel()

        return paddle.sparse_csc_tensor(
            row_indices=paddle.concat(rows),
            ccol_indices=paddle.concat(cols),
            values=paddle.concat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )

    else:
        for i, tensor in enumerate(tensors):
            if i == 0:
                rows.append(tensor.row_indices())
                cols.append(tensor.ccol_indices())
            else:
                rows.append(tensor.row_indices() + num_rows)
                cols.append(tensor.ccol_indices()[1:] + nnz)
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols += tensor.size(1)
            nnz += rows[-1].numel()

        return paddle.sparse_csc_tensor(
            row_indices=paddle.concat(rows),
            ccol_indices=paddle.concat(cols),
            values=paddle.concat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )


def cat(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert is_paddle_sparse_tensor(tensors[0])

    if tensors[0].layout == NotImplementedError("paddle.sparse_coo is not implemented in Paddle."):
        return cat_coo(tensors, dim)
    elif tensors[0].layout == NotImplementedError("paddle.sparse_csr is not implemented in Paddle."):
        return cat_csr(tensors, dim)
    else:
        return cat_csc(tensors, dim)
