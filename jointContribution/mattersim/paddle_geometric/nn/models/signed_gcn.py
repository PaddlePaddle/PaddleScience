from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Embedding, Linear, LayerList
from paddle_geometric.nn import SignedConv
from paddle_geometric.utils import coalesce, negative_sampling, structured_negative_sampling

class SignedGCN(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        lamb: float = 5,
        bias: bool = True,
    ):
        super(SignedGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb

        self.conv1 = SignedConv(in_channels, hidden_channels // 2, first_aggr=True)
        self.convs = LayerList()
        for _ in range(num_layers - 1):
            self.convs.append(SignedConv(hidden_channels // 2, hidden_channels // 2, first_aggr=False))

        self.lin = Linear(2 * hidden_channels, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def split_edges(
        self,
        edge_index: Tensor,
        test_ratio: float = 0.2,
    ) -> Tuple[Tensor, Tensor]:
        mask = paddle.ones([edge_index.shape[1]], dtype=paddle.bool)
        mask[paddle.rand([mask.shape[0]]).argsort()[:int(test_ratio * mask.shape[0])]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def create_spectral_features(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        import scipy.sparse as sp
        from sklearn.decomposition import TruncatedSVD

        edge_index = paddle.concat([pos_edge_index, neg_edge_index], axis=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.numpy()

        pos_val = paddle.full([pos_edge_index.shape[1]], 2, dtype=paddle.float32)
        neg_val = paddle.full([neg_edge_index.shape[1]], 0, dtype=paddle.float32)
        val = paddle.concat([pos_val, neg_val], axis=0)

        row, col = edge_index
        edge_index = paddle.concat([edge_index, paddle.stack([col, row], axis=1)], axis=1)
        val = paddle.concat([val, val], axis=0)

        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1

        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = sp.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.in_channels, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return paddle.to_tensor(x, dtype=paddle.float32)

    def forward(
        self,
        x: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        value = paddle.concat([z[edge_index[0]], z[edge_index[1]]], axis=1)
        value = self.lin(value)
        return F.log_softmax(value, axis=1)

    def nll_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        edge_index = paddle.concat([pos_edge_index, neg_edge_index], axis=1)
        none_edge_index = negative_sampling(edge_index, z.shape[0])

        nll_loss = 0
        nll_loss += F.nll_loss(self.discriminate(z, pos_edge_index),
                               paddle.full([pos_edge_index.shape[1]], 0, dtype=paddle.long))
        nll_loss += F.nll_loss(self.discriminate(z, neg_edge_index),
                               paddle.full([neg_edge_index.shape[1]], 1, dtype=paddle.long))
        nll_loss += F.nll_loss(self.discriminate(z, none_edge_index),
                               paddle.full([none_edge_index.shape[1]], 2, dtype=paddle.long))
        return nll_loss / 3.0

    def pos_embedding_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
    ) -> Tensor:
        i, j, k = structured_negative_sampling(pos_edge_index, z.shape[0])

        out = (z[i] - z[j]).pow(2).sum(axis=1) - (z[i] - z[k]).pow(2).sum(axis=1)
        return paddle.clip(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        i, j, k = structured_negative_sampling(neg_edge_index, z.shape[0])

        out = (z[i] - z[k]).pow(2).sum(axis=1) - (z[i] - z[j]).pow(2).sum(axis=1)
        return paddle.clip(out, min=0).mean()

    def loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[float, float]:
        from sklearn.metrics import f1_score, roc_auc_score

        with paddle.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].argmax(axis=1)
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].argmax(axis=1)

        pred = (1 - paddle.concat([pos_p, neg_p])).cpu()
        y = paddle.concat(
            [pred.new_ones([pos_p.shape[0]]),
             pred.new_zeros([neg_p.shape[0]])])

        auc = roc_auc_score(y.numpy(), pred.numpy())
        f1 = f1_score(y.numpy(), pred.numpy(), average='binary') if pred.sum() > 0 else 0

        return auc, f1

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, num_layers={self.num_layers})')
