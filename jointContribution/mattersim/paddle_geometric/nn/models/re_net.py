import math
from typing import Callable, List, Tuple

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import GRU, Linear

from paddle_geometric.data import Data
from paddle_geometric.utils import scatter


class RENet(paddle.nn.Layer):
    r"""The Recurrent Event Network model from the `"Recurrent Event Network
    for Reasoning over Temporal Knowledge Graphs"
    <https://arxiv.org/abs/1904.05530>`_ paper.

    .. math::
        f_{\mathbf{\Theta}}(\mathbf{e}_s, \mathbf{e}_r,
        \mathbf{h}^{(t-1)}(s, r))

    based on a RNN encoder

    .. math::
        \mathbf{h}^{(t)}(s, r) = \textrm{RNN}(\mathbf{e}_s, \mathbf{e}_r,
        g(\mathcal{O}^{(t)}_r(s)), \mathbf{h}^{(t-1)}(s, r))

    where :math:`\mathbf{e}_s` and :math:`\mathbf{e}_r` denote entity and
    relation embeddings, and :math:`\mathcal{O}^{(t)}_r(s)` represents the set
    of objects interacted with subject :math:`s` under relation :math:`r` at
    timestamp :math:`t`.
    This model implements :math:`g` as the **Mean Aggregator** and
    :math:`f_{\mathbf{\Theta}}` as a linear projection.

    Args:
        num_nodes (int): The number of nodes in the knowledge graph.
        num_rels (int): The number of relations in the knowledge graph.
        hidden_channels (int): Hidden size of node and relation embeddings.
        seq_len (int): The sequence length of past events.
        num_layers (int, optional): The number of recurrent layers.
            (default: :obj:`1`)
        dropout (float): If non-zero, introduces a dropout layer before the
            final prediction. (default: :obj:`0.`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        hidden_channels: int,
        seq_len: int,
        num_layers: int = 1,
        dropout: float = 0.,
        bias: bool = True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = dropout

        self.ent = paddle.create_parameter(
            shape=[num_nodes, hidden_channels],
            dtype='float32'
        )

        self.rel = paddle.create_parameter(
            shape=[num_rels, hidden_channels],
            dtype='float32'
        )

        self.sub_gru = GRU(3 * hidden_channels, hidden_channels, num_layers,
                           batch_first=True, bias=bias)
        self.obj_gru = GRU(3 * hidden_channels, hidden_channels, num_layers,
                           batch_first=True, bias=bias)

        self.sub_lin = Linear(3 * hidden_channels, num_nodes, bias=bias)
        self.obj_lin = Linear(3 * hidden_channels, num_nodes, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.ent, gain=math.sqrt(2.0))
        paddle.nn.initializer.XavierUniform()(self.rel, gain=math.sqrt(2.0))

        self.sub_gru.reset_parameters()
        self.obj_gru.reset_parameters()
        self.sub_lin.reset_parameters()
        self.obj_lin.reset_parameters()

    @staticmethod
    def pre_transform(seq_len: int) -> Callable:
        r"""Precomputes history objects."""

        class PreTransform:
            def __init__(self, seq_len: int):
                self.seq_len = seq_len
                self.inc = 5000
                self.t_last = 0
                self.sub_hist = self.increase_hist_node_size([])
                self.obj_hist = self.increase_hist_node_size([])

            def increase_hist_node_size(self, hist: List[int]) -> List[int]:
                hist_inc = paddle.zeros((self.inc, self.seq_len + 1, 0))
                return hist + hist_inc.tolist()

            def get_history(
                self,
                hist: List[int],
                node: int,
                rel: int,
            ) -> Tuple[Tensor, Tensor]:
                hists, ts = [], []
                for s in range(self.seq_len):
                    h = hist[node][s]
                    hists += h
                    ts.append(paddle.full((len(h), ), s, dtype=paddle.int64))
                node, r = paddle.tensor(hists, dtype=paddle.int64).view(
                    -1, 2).T.contiguous()
                node = node[r == rel]
                t = paddle.concat(ts, axis=0)[r == rel]
                return node, t

            def step(self, hist: List[int]) -> List[int]:
                for i in range(len(hist)):
                    hist[i] = hist[i][1:]
                    hist[i].append([])
                return hist

            def __call__(self, data: Data) -> Data:
                sub, rel, obj, t = data.sub, data.rel, data.obj, data.t

                if max(sub, obj) + 1 > len(self.sub_hist):
                    self.sub_hist = self.increase_hist_node_size(self.sub_hist)
                    self.obj_hist = self.increase_hist_node_size(self.obj_hist)

                if t > self.t_last:
                    self.sub_hist = self.step(self.sub_hist)
                    self.obj_hist = self.step(self.obj_hist)
                    self.t_last = t

                data.h_sub, data.h_sub_t = self.get_history(
                    self.sub_hist, sub, rel)
                data.h_obj, data.h_obj_t = self.get_history(
                    self.obj_hist, obj, rel)

                self.sub_hist[sub][-1].append([obj, rel])
                self.obj_hist[obj][-1].append([sub, rel])

                return data

            def __repr__(self) -> str:
                return f'{self.__class__.__name__}(seq_len={self.seq_len})'

        return PreTransform(seq_len)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Given a :obj:`data` batch, computes the forward pass."""

        assert 'h_sub_batch' in data and 'h_obj_batch' in data
        batch_size, seq_len = data.sub.shape[0], self.seq_len

        h_sub_t = data.h_sub_t + data.h_sub_batch * seq_len
        h_obj_t = data.h_obj_t + data.h_obj_batch * seq_len

        h_sub = scatter(self.ent[data.h_sub], h_sub_t, dim_size=batch_size * seq_len,
                        reduce='mean').view(batch_size, seq_len, -1)
        h_obj = scatter(self.ent[data.h_obj], h_obj_t, dim_size=batch_size * seq_len,
                        reduce='mean').view(batch_size, seq_len, -1)

        sub = self.ent[data.sub].unsqueeze(1).repeat(1, seq_len, 1)
        rel = self.rel[data.rel].unsqueeze(1).repeat(1, seq_len, 1)
        obj = self.ent[data.obj].unsqueeze(1).repeat(1, seq_len, 1)

        _, h_sub = self.sub_gru(paddle.concat([sub, h_sub, rel], axis=-1))
        _, h_obj = self.obj_gru(paddle.concat([obj, h_obj, rel], axis=-1))
        h_sub, h_obj = h_sub.squeeze(0), h_obj.squeeze(0)

        h_sub = paddle.concat([self.ent[data.sub], h_sub, self.rel[data.rel]],
                              axis=-1)
        h_obj = paddle.concat([self.ent[data.obj], h_obj, self.rel[data.rel]],
                              axis=-1)

        h_sub = F.dropout(h_sub, p=self.dropout, training=self.training)
        h_obj = F.dropout(h_obj, p=self.dropout, training=self.training)

        log_prob_obj = F.log_softmax(self.sub_lin(h_sub), axis=1)
        log_prob_sub = F.log_softmax(self.obj_lin(h_obj), axis=1)

        return log_prob_obj, log_prob_sub

    def test(self, logits: Tensor, y: Tensor) -> Tensor:
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10.
        """
        _, perm = logits.argsort(axis=1, descending=True)
        mask = (y.unsqueeze(1) == perm)

        nnz = mask.nonzero(as_tuple=False)
        mrr = (1 / (nnz[:, -1] + 1).to(paddle.float32)).mean().item()
        hits1 = mask[:, :1].sum().item() / y.shape[0]
        hits3 = mask[:, :3].sum().item() / y.shape[0]
        hits10 = mask[:, :10].sum().item() / y.shape[0]

        return paddle.to_tensor([mrr, hits1, hits3, hits10])


