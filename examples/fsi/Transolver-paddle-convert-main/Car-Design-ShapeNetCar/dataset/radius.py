# import paddle
# import numpy as np
# from scipy.spatial import cKDTree
# from typing import Optional
#
# def radius(
#         x: paddle.Tensor,
#         y: paddle.Tensor,
#         r: float,
#         batch_x: Optional[paddle.Tensor] = None,
#         batch_y: Optional[paddle.Tensor] = None,
#         max_num_neighbors: int = 32,
#         min_num_neighbors: int = 1,
#         num_workers: int = 32  # 添加线程数参数，默认 32
# ) -> paddle.Tensor:
#     # 默认在 CPU 上运行，不需要指定设备
#
#     if batch_x is None:
#         batch_x = paddle.zeros([x.shape[0]], dtype='int64')
#     if batch_y is None:
#         batch_y = paddle.zeros([y.shape[0]], dtype='int64')
#
#     x = x.reshape([-1, 1]) if x.ndim == 1 else x
#     y = y.reshape([-1, 1]) if y.ndim == 1 else y
#
#     assert x.ndim == 2 and batch_x.ndim == 1
#     assert y.ndim == 2 and batch_y.ndim == 1
#     assert x.shape[1] == y.shape[1]
#     assert x.shape[0] == batch_x.shape[0]
#     assert y.shape[0] == batch_y.shape[0]
#
#     # 拼接批次维度信息
#     x = paddle.concat([x, (2 * r * batch_x.reshape([-1, 1])).astype(x.dtype)], axis=-1)
#     y = paddle.concat([y, (2 * r * batch_y.reshape([-1, 1])).astype(y.dtype)], axis=-1)
#
#     # 构建 KD 树并查询，使用多线程
#     tree = cKDTree(x.numpy())  # cKDTree 只支持 CPU 计算
#     distances, col = tree.query(
#         y.numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8, workers=num_workers
#     )
#
#     # 保证最小邻居数
#     valid_indices = [i for i in range(len(col)) if len(col[i]) >= min_num_neighbors]
#     col = [col[i] for i in valid_indices]
#     distances = [distances[i] for i in valid_indices]
#
#     # 将结果转换为张量
#     col = [paddle.to_tensor(c, dtype='int64') for c in col]
#     row = [paddle.full_like(c, i, dtype='int64') for i, c in enumerate(col)]
#     row, col = paddle.concat(row, axis=0), paddle.concat(col, axis=0)
#     mask = col < tree.n
#
#     return paddle.stack([row[mask], col[mask]], axis=0)
#
# def radius_graph(
#         x: paddle.Tensor,
#         r: float,
#         batch: Optional[paddle.Tensor] = None,
#         loop: bool = False,
#         max_num_neighbors: int = 32,
#         min_num_neighbors: int = 1,
#         flow: str = 'source_to_target',
#         num_workers: int = 32  # 添加线程数参数，默认 32
# ) -> paddle.Tensor:
#     if batch is not None:
#         batch = batch
#
#     assert flow in ['source_to_target', 'target_to_source']
#     row, col = radius(x, x, r, batch, batch, max_num_neighbors + 1, min_num_neighbors, num_workers)
#     row, col = (col, row) if flow == 'source_to_target' else (row, col)
#
#     if not loop:
#         mask = row != col
#         row, col = row[mask], col[mask]
#
#     return paddle.stack([row, col], axis=0)

import paddle
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional
from concurrent.futures import ThreadPoolExecutor


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
