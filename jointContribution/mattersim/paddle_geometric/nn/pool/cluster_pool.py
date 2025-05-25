from typing import NamedTuple, Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import Tensor

from paddle_geometric.utils import (
    dense_to_sparse,
    one_hot,
    to_dense_adj,
    to_scipy_sparse_matrix,
)


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor


class ClusterPooling(paddle.nn.Layer):
    r"""The cluster pooling operator from the `"Edge-Based Graph Component
    Pooling" <paper url>`_ paper.

    :class:`ClusterPooling` computes a score for each edge.
    Based on the selected edges, graph clusters are calculated and compressed
    to one node using the injective :obj:`"sum"` aggregation function.
    Edges are remapped based on the nodes created by each cluster and the
    original edges.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (str, optional): The function to apply
            to compute the edge score from raw edge scores (:obj:`"tanh"`),
            :obj:`"sigmoid"`, :obj:`"log_softmax"`). (default: :obj:`"tanh"`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0.0`)
        threshold (float, optional): The threshold of edge scores. If set to
            :obj:`None`, will be automatically inferred depending on
            :obj:`edge_score_method`. (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        edge_score_method: str = 'tanh',
        dropout: float = 0.0,
        threshold: Optional[float] = None,
    ):
        super().__init__()
        assert edge_score_method in ['tanh', 'sigmoid', 'log_softmax']

        if threshold is None:
            threshold = 0.5 if edge_score_method == 'sigmoid' else 0.0

        self.in_channels = in_channels
        self.edge_score_method = edge_score_method
        self.dropout = dropout
        self.threshold = threshold

        self.lin = paddle.nn.Linear(2 * in_channels, 1)

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The node features.
            edge_index (paddle.Tensor): The edge indices.
            batch (paddle.Tensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(paddle.Tensor)* - The pooled node features.
            * **edge_index** *(paddle.Tensor)* - The coarsened edge indices.
            * **batch** *(paddle.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that can be consumed
              for unpooling.
        """
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

        edge_attr = paddle.concat(
            [x[edge_index[0]], x[edge_index[1]]],
            axis=-1,
        )
        edge_score = self.lin(edge_attr).flatten()
        edge_score = F.dropout(edge_score, p=self.dropout,
                               training=self.training)

        if self.edge_score_method == 'tanh':
            edge_score = edge_score.tanh()
        elif self.edge_score_method == 'sigmoid':
            edge_score = edge_score.sigmoid()
        else:
            assert self.edge_score_method == 'log_softmax'
            edge_score = F.log_softmax(edge_score, axis=0)

        return self._merge_edges(x, edge_index, batch, edge_score)

    def _merge_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_score: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:

        from scipy.sparse.csgraph import connected_components

        edge_contract = edge_index[:, edge_score > self.threshold]

        adj = to_scipy_sparse_matrix(edge_contract, num_nodes=x.size(0))
        _, cluster_np = connected_components(adj, directed=True,
                                             connection="weak")

        cluster = paddle.to_tensor(cluster_np, dtype=paddle.long, device=x.device)
        C = one_hot(cluster)
        A = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)
        S = to_dense_adj(edge_index, edge_attr=edge_score,
                         max_num_nodes=x.size(0)).squeeze(0)

        A_contract = to_dense_adj(edge_contract,
                                  max_num_nodes=x.size(0)).squeeze(0)
        nodes_single = ((A_contract.sum(axis=-1) +
                         A_contract.sum(axis=-2)) == 0).nonzero()
        S[nodes_single, nodes_single] = 1.0

        x_out = (S @ C).T @ x
        edge_index_out, _ = dense_to_sparse((C.T @ A @ C).fill_diagonal_(0))
        batch_out = batch.new_empty(x_out.size(0)).scatter_(0, cluster, batch)
        unpool_info = UnpoolInfo(edge_index, cluster, batch)

        return x_out, edge_index_out, batch_out, unpool_info

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
