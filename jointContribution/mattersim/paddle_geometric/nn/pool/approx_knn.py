from typing import Optional, Tuple

import paddle
from paddle import Tensor

# Replacing 'pynndescent' with an approximation in Paddle
def approx_knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Tensor = None,
    batch_y: Tensor = None,
) -> Tensor:
    """Finds for each element in :obj:`y` the :obj:`k` approximated nearest
    points in :obj:`x`."""
    from sklearn.neighbors import NearestNeighbors

    if batch_x is None:
        batch_x = x.new_zeros(x.shape[0], dtype=paddle.int64)
    if batch_y is None:
        batch_y = y.new_zeros(y.shape[0], dtype=paddle.int64)

    x = x.unsqueeze(1) if x.ndim == 1 else x
    y = y.unsqueeze(1) if y.ndim == 1 else y

    assert x.ndim == 2 and batch_x.ndim == 1
    assert y.ndim == 2 and batch_y.ndim == 1
    assert x.shape[1] == y.shape[1]
    assert x.shape[0] == batch_x.shape[0]
    assert y.shape[0] == batch_y.shape[0]

    min_xy = min(x.min(), y.min())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max(), y.max())
    x, y = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist:
    x = paddle.concat([x, 2 * x.shape[1] * batch_x.unsqueeze(-1).astype(x.dtype)], axis=-1)
    y = paddle.concat([y, 2 * y.shape[1] * batch_y.unsqueeze(-1).astype(y.dtype)], axis=-1)

    # Using sklearn's NearestNeighbors for kNN
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(x.numpy())
    col, dist = nn.kneighbors(y.numpy())
    dist = paddle.to_tensor(dist).view(-1).to(x.device, x.dtype)
    col = paddle.to_tensor(col).view(-1).to(x.device, paddle.int64)
    row = paddle.arange(y.shape[0], device=x.device, dtype=paddle.int64)
    row = row.tile([k])
    mask = ~paddle.isinf(dist)
    row, col = row[mask], col[mask]

    return paddle.stack([row, col], axis=0)


def approx_knn_graph(
    x: Tensor,
    k: int,
    batch: Tensor = None,
    loop: bool = False,
    flow: str = 'source_to_target',
) -> Tensor:
    """Computes graph edges to the nearest approximated :obj:`k` points."""
    assert flow in ['source_to_target', 'target_to_source']
    row, col = approx_knn(x, x, k if loop else k + 1, batch, batch)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return paddle.stack([row, col], axis=0)
