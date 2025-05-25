from typing import Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Embedding, LayerList, Linear

from paddle_geometric.nn.conv import LGConv
from paddle_geometric.typing import Adj, OptTensor
from paddle_geometric.utils import is_sparse, to_edge_index

class _Loss(nn.Layer):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction = self._legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

    @staticmethod
    def _legacy_get_string(size_average, reduce):
        if size_average is not None and reduce is not None:
            if size_average and reduce:
                return 'mean'
            elif not size_average and reduce:
                return 'sum'
            elif not reduce:
                return 'none'
        raise ValueError("Invalid combination of 'size_average' and 'reduce'.")

class LightGCN(nn.Layer):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    :class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
    propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

    where each layer's embedding is computed as

    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Two prediction heads and training objectives are provided:
    **link prediction** (via
    :meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
    **recommendation** (via
    :meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.recommend`).

    .. note::

        Embeddings are propagated according to the graph connectivity specified
        by :obj:`edge_index` while rankings or link probabilities are computed
        according to the edges specified by :obj:`edge_label_index`.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        alpha (float or paddle.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.shape[0] == num_layers + 1
        else:
            alpha = paddle.full([num_layers + 1], alpha, dtype=paddle.float32)
        self.alpha = self.create_parameter(shape=alpha.shape, default_initializer=paddle.nn.initializer.Assign(alpha))

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = LayerList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        paddle.nn.initializer.XavierUniform()(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (paddle.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (paddle.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (paddle.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
        """
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(axis=-1)

    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`."""

        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
        sorted: bool = True,
    ) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`."""

        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = paddle.matmul(out_src, out_dst.t())
        top_index = pred.topk(k, axis=-1, sorted=sorted).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.reshape([-1])].reshape(top_index.shape)

        return top_index

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective."""

        loss_fn = paddle.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.astype(pred.dtype))

    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: Optional[Tensor] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss."""

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss."""
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / positives.shape[0]

        return -log_prob + regularization
