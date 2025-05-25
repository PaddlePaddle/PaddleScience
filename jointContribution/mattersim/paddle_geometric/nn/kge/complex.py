import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Embedding

from paddle_geometric.nn.kge import KGEModel


class ComplEx(KGEModel):
    r"""The ComplEx model from the `"Complex Embeddings for Simple Link
    Prediction" <https://arxiv.org/abs/1606.06357>`_ paper.

    :class:`ComplEx` models relations as complex-valued bilinear mappings
    between head and tail entities using the Hermetian dot product.
    The entities and relations are embedded in different dimensional spaces,
    resulting in the scoring function:

    .. math::
        d(h, r, t) = Re(< \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t>)

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb_im = Embedding(num_relations, hidden_channels, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.node_emb.weight)
        paddle.nn.initializer.XavierUniform()(self.node_emb_im.weight)
        paddle.nn.initializer.XavierUniform()(self.rel_emb.weight)
        paddle.nn.initializer.XavierUniform()(self.rel_emb_im.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        return (triple_dot(head_re, rel_re, tail_re) +
                triple_dot(head_im, rel_re, tail_im) +
                triple_dot(head_re, rel_im, tail_im) -
                triple_dot(head_im, rel_im, tail_re))

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


def triple_dot(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return (x * y * z).sum(axis=-1)
