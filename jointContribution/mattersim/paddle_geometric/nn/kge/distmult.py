import paddle
import paddle.nn.functional as F
from paddle import Tensor

from paddle_geometric.nn.kge import KGEModel


class DistMult(KGEModel):
    r"""The DistMult model from the `"Embedding Entities and Relations for
    Learning and Inference in Knowledge Bases"
    <https://arxiv.org/abs/1412.6575>`_ paper.

    :class:`DistMult` models relations as diagonal matrices, which simplifies
    the bi-linear interaction between the head and tail entities to the score
    function:

    .. math::
        d(h, r, t) = < \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t >

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.node_emb.weight)
        paddle.nn.initializer.XavierUniform()(self.rel_emb.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        return (head * rel * tail).sum(axis=-1)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            label=paddle.ones_like(pos_score),
            margin=self.margin,
        )
