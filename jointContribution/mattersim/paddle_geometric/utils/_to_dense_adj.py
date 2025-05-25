from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import cumsum, scatter


def to_dense_adj(
    edge_index: Tensor,
    batch: OptTensor = None,
    edge_attr: OptTensor = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (Tensor): The edge indices.
        batch (Tensor, optional): Batch vector that assigns each node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge features.
        max_num_nodes (int, optional): The size of the output node dimension. (default: :obj:`None`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        max_index = int(paddle.max(edge_index)) + 1 if edge_index.numel() > 0 else 0
        batch = paddle.zeros([max_index], dtype='int64', place=edge_index.place)

    if batch_size is None:
        batch_size = int(paddle.max(batch)) + 1 if batch.numel() > 0 else 1

    one = paddle.ones([batch.size], dtype='int64', place=batch.place)
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = cumsum(num_nodes)

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(paddle.max(num_nodes))

    elif ((idx1.numel() > 0 and paddle.max(idx1) >= max_num_nodes)
          or (idx2.numel() > 0 and paddle.max(idx2) >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = paddle.ones([idx0.numel()], dtype=edge_index.dtype, place=edge_index.place)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.shape)[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
    adj = adj.reshape(size)

    return adj
