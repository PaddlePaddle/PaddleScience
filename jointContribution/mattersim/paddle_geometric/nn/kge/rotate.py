import math

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Embedding

from paddle_geometric.nn.kge import KGEModel


class RotatE(KGEModel):
    r"""The RotatE model from the `"RotatE: Knowledge Graph Embedding by
    Relational Rotation in Complex Space" <https://arxiv.org/abs/
    1902.10197>`_ paper.

    :class:`RotatE` models relations as a rotation in complex space
    from head to tail such that

    .. math::
        \mathbf{e}_t = \mathbf{e}_h \circ \mathbf{e}_r,

    resulting in the scoring function

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h \circ \mathbf{e}_r - \mathbf{e}_t \|}_p

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
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
        self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.node_emb.weight)
        paddle.nn.initializer.XavierUniform()(self.node_emb_im.weight)
        paddle.assign(paddle.uniform(self.rel_emb.weight.shape, min=0, max=2 * math.pi), self.rel_emb.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = paddle.cos(rel_theta), paddle.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = paddle.stack([re_score, im_score], axis=2)
        score = paddle.norm(complex_score, p=2, axis=(1, 2))

        return self.margin - score

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = paddle.concat([pos_score, neg_score], axis=0)

        pos_target = paddle.ones_like(pos_score)
        neg_target = paddle.zeros_like(neg_score)
        target = paddle.concat([pos_target, neg_target], axis=0)

        return F.binary_cross_entropy_with_logits(scores, target)
