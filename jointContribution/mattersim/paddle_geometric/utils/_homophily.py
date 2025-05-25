from typing import Optional, Tuple, Union, overload

import paddle
from paddle import Tensor

from paddle_geometric.typing import Adj, OptTensor
from paddle_geometric.utils import degree, scatter


@overload
def homophily(
    edge_index: Adj,
    y: Tensor,
    batch: None = ...,
    method: str = ...,
) -> float:
    pass


@overload
def homophily(
    edge_index: Adj,
    y: Tensor,
    batch: Tensor,
    method: str = ...,
) -> Tensor:
    pass


def homophily(
    edge_index: Adj,
    y: Tensor,
    batch: OptTensor = None,
    method: str = 'edge',
) -> Union[float, Tensor]:
    assert method in {'edge', 'node', 'edge_insensitive'}
    y = y.squeeze(-1) if y.ndim > 1 else y

    if isinstance(edge_index, paddle.sparse.SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        out = paddle.zeros([row.shape[0]], dtype='float32', place=row.place)
        out = paddle.where(y[row] == y[col], paddle.ones_like(out), out)
        if batch is None:
            return float(out.mean().item())
        else:
            dim_size = int(batch.max().item()) + 1
            return scatter(out, batch[col], 0, dim_size, reduce='mean')

    elif method == 'node':
        out = paddle.zeros([row.shape[0]], dtype='float32', place=row.place)
        out = paddle.where(y[row] == y[col], paddle.ones_like(out), out)
        out = scatter(out, col, 0, dim_size=y.shape[0], reduce='mean')
        if batch is None:
            return float(out.mean().item())
        else:
            return scatter(out, batch, dim=0, reduce='mean')

    elif method == 'edge_insensitive':
        num_classes = int(y.max().item()) + 1
        assert num_classes >= 2
        batch = paddle.zeros_like(y) if batch is None else batch
        num_nodes = degree(batch, dtype='int64')
        num_graphs = num_nodes.shape[0]
        batch = num_classes * batch + y

        h = homophily(edge_index, y, batch, method='edge')
        h = h.reshape([num_graphs, num_classes])

        counts = paddle.bincount(batch, minlength=num_classes * num_graphs)
        counts = counts.reshape([num_graphs, num_classes])
        proportions = counts / num_nodes.reshape([-1, 1])

        out = (h - proportions).clip(min=0).sum(axis=-1)
        out /= (num_classes - 1)
        return out if out.shape[0] > 1 else float(out.item())

    else:
        raise NotImplementedError
