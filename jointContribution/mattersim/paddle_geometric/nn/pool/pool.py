from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.utils import coalesce, remove_self_loops, scatter


def pool_edge(
    cluster,
    edge_index,
    edge_attr: Optional[paddle.Tensor] = None,
    reduce: Optional[str] = 'sum',
):
    num_nodes = cluster.shape[0]
    edge_index = cluster[edge_index.flatten()].reshape([2, -1])
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         reduce=reduce)
    return edge_index, edge_attr


def pool_batch(perm, batch):
    return batch[perm]


def pool_pos(cluster, pos):
    return scatter(pos, cluster, dim=0, reduce='mean')
