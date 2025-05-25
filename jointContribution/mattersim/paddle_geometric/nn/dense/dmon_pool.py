from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor

EPS = 1e-15


def _rank3_trace(tensor: Tensor) -> Tensor:
    return paddle.sum(paddle.diagonal(tensor, axis1=-2, axis2=-1), axis=-1)


class DMoNPooling(paddle.nn.Layer):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper.
    """
    def __init__(self, channels: Union[int, List[int]], k: int, dropout: float = 0.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        from paddle.nn import Sequential, Linear, ReLU
        layers = [Linear(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        layers.append(Linear(channels[-1], k))
        self.mlp = Sequential(*layers)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, Linear):
                layer.weight.set_value(paddle.nn.initializer.XavierUniform()(layer.weight.shape))

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = x.unsqueeze(0) if x.ndim == 2 else x
        adj = adj.unsqueeze(0) if adj.ndim == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = F.softmax(s, axis=-1)

        batch_size, num_nodes, _ = x.shape
        C = s.shape[-1]

        if mask is None:
            mask = paddle.ones([batch_size, num_nodes], dtype='bool')

        mask = mask.cast(x.dtype).reshape([batch_size, num_nodes, 1])
        x, s = x * mask, s * mask

        out = F.selu(paddle.matmul(s.transpose([0, 2, 1]), x))
        out_adj = paddle.matmul(paddle.matmul(s.transpose([0, 2, 1]), adj), s)

        # Spectral loss
        degrees = paddle.sum(adj, axis=-1, keepdim=True) * mask
        m = paddle.sum(degrees, axis=[-2, -1]) / 2
        m_expand = m.reshape([-1, 1, 1]).expand([-1, C, C])
        ca = paddle.matmul(s.transpose([0, 2, 1]), degrees)
        cb = paddle.matmul(degrees.transpose([0, 2, 1]), s)
        normalizer = paddle.matmul(ca, cb) / (2 * m_expand)
        spectral_loss = -_rank3_trace(out_adj - normalizer) / (2 * m)
        spectral_loss = paddle.mean(spectral_loss)

        # Orthogonality regularization
        ss = paddle.matmul(s.transpose([0, 2, 1]), s)
        i_s = paddle.eye(C, dtype=ss.dtype)
        ortho_loss = paddle.norm(ss / paddle.norm(ss, p=2) - i_s / paddle.norm(i_s), p='fro', axis=[-2, -1])
        ortho_loss = paddle.mean(ortho_loss)

        # Cluster loss
        cluster_size = paddle.sum(s, axis=1)
        cluster_loss = paddle.norm(cluster_size, p=2, axis=1) / paddle.sum(mask, axis=1) * paddle.norm(i_s, p=2) - 1
        cluster_loss = paddle.mean(cluster_loss)

        # Fix and normalize coarsened adjacency matrix
        out_adj = paddle.where(out_adj - paddle.eye(C, dtype=out_adj.dtype) * paddle.diagonal(out_adj, axis1=-2, axis2=-1), out_adj, paddle.zeros_like(out_adj))
        d = paddle.sqrt(paddle.sum(out_adj, axis=-1, keepdim=True)) + EPS
        out_adj = out_adj / d / d.transpose([0, 2, 1])

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(channels={self.mlp[0].weight.shape[1]}, num_clusters={self.mlp[-1].out_features})'
