from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.nn.pool.connect import Connect, ConnectOutput
from paddle_geometric.nn.pool.select import SelectOutput
from paddle_geometric.utils.num_nodes import maybe_num_nodes


def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    node_index: Tensor,
    cluster_index: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if cluster_index is None:
        cluster_index = paddle.arange(node_index.shape[0], device=node_index.device)

    mask = node_index.new_full((num_nodes, ), -1)
    mask[node_index] = cluster_index

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return paddle.stack([row, col], axis=0), edge_attr


class FilterEdges(Connect):
    r"""Filters out edges if their incident nodes are not in any cluster.

    .. math::
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where :math:`\mathbf{i}` denotes the set of retained nodes.
    It is assumed that each cluster contains only one node.
    """
    def forward(
        self,
        select_output: SelectOutput,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> ConnectOutput:

        if (not paddle.jit.is_scripting() and select_output.num_clusters
                != select_output.cluster_index.shape[0]):
            raise ValueError(f"'{self.__class__.__name__}' requires each "
                             f"cluster to contain only one node")

        edge_index, edge_attr = filter_adj(
            edge_index,
            edge_attr,
            select_output.node_index,
            select_output.cluster_index,
            num_nodes=select_output.num_nodes,
        )
        batch = self.get_pooled_batch(select_output, batch)

        return ConnectOutput(edge_index, edge_attr, batch)
