import paddle
from typing import Optional
# from custom_setup_ops import custom_radius


# def radius(
#     x: paddle.Tensor,
#     y: paddle.Tensor,
#     r: float,
#     batch_x: Optional[paddle.Tensor] = None,
#     batch_y: Optional[paddle.Tensor] = None,
#     max_num_neighbors: int = 32,
#     num_workers: int = 32,
#     batch_size: Optional[int] = None,
# ) -> paddle.Tensor:
#     r"""Finds for each element in :obj:`y` all points in :obj:`x` within
#     distance :obj:`r`.
#
#     Args:
#         x (Tensor): Node feature matrix
#             :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
#         y (Tensor): Node feature matrix
#             :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
#         r (float): The radius.
#         batch_x (LongTensor, optional): Batch vector
#             :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
#             node to a specific example. :obj:`batch_x` needs to be sorted.
#             (default: :obj:`None`)
#         batch_y (LongTensor, optional): Batch vector
#             :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
#             node to a specific example. :obj:`batch_y` needs to be sorted.
#             (default: :obj:`None`)
#         max_num_neighbors (int, optional): The maximum number of neighbors to
#             return for each element in :obj:`y`.
#             If the number of actual neighbors is greater than
#             :obj:`max_num_neighbors`, returned neighbors are picked randomly.
#             (default: :obj:`32`)
#         num_workers (int): Number of workers to use for computation. Has no
#             effect in case :obj:`batch_x` or :obj:`batch_y` is not
#             :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
#         batch_size (int, optional): The number of examples :math:`B`.
#             Automatically calculated if not given. (default: :obj:`None`)
#
#     .. code-block:: python
#
#         import paddle
#         from paddle_cluster import radius
#
#         x = paddle.to_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype='float32')
#         batch_x = paddle.to_tensor([0, 0, 0, 0], dtype='int64')
#         y = paddle.to_tensor([[-1, 0], [1, 0]], dtype='float32')
#         batch_y = paddle.to_tensor([0, 0], dtype='int64')
#         assign_index = radius(x, y, 1.5, batch_x, batch_y)
#     """
#
#     if x.numel() == 0 or y.numel() == 0:
#         return paddle.empty(shape=[2, 0], dtype='int64')
#
#     x = x.reshape([-1, 1]) if x.dim() == 1 else x
#     y = y.reshape([-1, 1]) if y.dim() == 1 else y
#     x, y = x.contiguous(), y.contiguous()
#
#     if batch_size is None:
#         batch_size = 1
#         if batch_x is not None:
#             assert x.shape[0] == batch_x.numel()
#             batch_size = int(batch_x.max()) + 1
#         if batch_y is not None:
#             assert y.shape[0] == batch_y.numel()
#             batch_size = max(batch_size, int(batch_y.max()) + 1)
#     assert batch_size > 0
#
#     ptr_x: Optional[paddle.Tensor] = None
#     ptr_y: Optional[paddle.Tensor] = None
#
#     if batch_size > 1:
#         assert batch_x is not None
#         assert batch_y is not None
#         arange = paddle.arange(batch_size + 1, dtype='int64', device=x.place)
#         ptr_x = paddle.bucketize(arange, batch_x)
#         ptr_y = paddle.bucketize(arange, batch_y)
#
#     out = custom_radius(x, y, r, max_num_neighbors, ignore_same_index=False)
#
#     # 在 Python 端进行转置
#     out_transposed = paddle.transpose(out, [1, 0])
#
#     # 交换两行
#     out_swapped = paddle.concat([out_transposed[1].unsqueeze(0), out_transposed[0].unsqueeze(0)], axis=0)
#
#     return out_swapped
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def radius(
        x: paddle.Tensor,
        y: paddle.Tensor,
        r: float,
        batch_x: Optional[paddle.Tensor] = None,
        batch_y: Optional[paddle.Tensor] = None,
        max_num_neighbors: int = 32,
        num_workers: int = 32,
        batch_size: Optional[int] = None,
) -> paddle.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return paddle.empty([2, 0], dtype='int64', place=x.place)

    x = x.reshape([-1, 1]) if x.ndim == 1 else x
    y = y.reshape([-1, 1]) if y.ndim == 1 else y

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.shape[0] == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.shape[0] == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    x = paddle.concat([x, 2 * r * batch_x.reshape([-1, 1])], axis=-1) if batch_x is not None else x
    y = paddle.concat([y, 2 * r * batch_y.reshape([-1, 1])], axis=-1) if batch_y is not None else y

    # 使用 cKDTree 创建 KD 树（只支持 CPU）
    tree = cKDTree(x.numpy())

    # 执行多线程查询
    def query_neighbors(idx):
        _, indices = tree.query(y[idx].numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
        row = [idx] * len(indices)
        return row, indices

    rows, cols = [], []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(query_neighbors, range(y.shape[0]))
        for row, col in results:
            rows.extend(row)
            cols.extend(col)

    row_tensor = paddle.to_tensor(rows, dtype='int64')
    col_tensor = paddle.to_tensor(cols, dtype='int64')
    mask = col_tensor < tree.n

    return paddle.stack([row_tensor[mask], col_tensor[mask]], axis=0)


def radius_graph(
    x: paddle.Tensor,
    r: float,
    batch: Optional[paddle.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 32,
    batch_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import paddle
        from paddle_cluster import radius_graph

        x = paddle.to_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype='float32')
        batch = paddle.to_tensor([0, 0, 0, 0], dtype='int64')
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors if loop else max_num_neighbors + 1,
                        num_workers, batch_size)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return paddle.stack([row, col], axis=0)