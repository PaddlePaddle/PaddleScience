from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, GRUCell, Linear

from paddle_geometric.nn import GATConv, MessagePassing, global_add_pool
from paddle_geometric.nn.inits import glorot, zeros
from paddle_geometric.typing import Adj, OptTensor
from paddle_geometric.utils import softmax
from paddle_geometric.utils import negative_sampling


EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(paddle.nn.Layer):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(axis=1)
        return paddle.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.
        """
        adj = paddle.matmul(z, z.T)
        return paddle.sigmoid(adj) if sigmoid else adj


class GAE(paddle.nn.Layer):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    """
    def __init__(self, encoder: Layer, decoder: Optional[Layer] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, *args, **kwargs) -> Tensor:
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        pos_loss = -paddle.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.shape[0])
        neg_loss = -paddle.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.shape[1])
        neg_y = z.new_zeros(neg_edge_index.shape[1])
        y = paddle.concat([pos_y, neg_y], axis=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = paddle.concat([pos_pred, neg_pred], axis=0)

        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.
    """
    def __init__(self, encoder: Layer, decoder: Optional[Layer] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + paddle.randn_like(logstd) * paddle.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = paddle.clip(self.__logstd__, max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else paddle.clip(logstd, max=MAX_LOGSTD)
        return -0.5 * paddle.mean(
            paddle.sum(1 + 2 * logstd - mu**2 - paddle.exp(logstd)**2, axis=1))


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    """
    def __init__(
        self,
        encoder: Layer,
        discriminator: Layer,
        decoder: Optional[Layer] = None,
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.discriminator.reset_parameters()

    def reg_loss(self, z: Tensor) -> Tensor:
        real = paddle.sigmoid(self.discriminator(z))
        real_loss = -paddle.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z: Tensor) -> Tensor:
        real = paddle.sigmoid(self.discriminator(paddle.randn_like(z)))
        fake = paddle.sigmoid(self.discriminator(z.detach()))
        real_loss = -paddle.log(real + EPS).mean()
        fake_loss = -paddle.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    """
    def __init__(
        self,
        encoder: Layer,
        discriminator: Layer,
        decoder: Optional[Layer] = None,
    ):
        super().__init__(encoder, discriminator, decoder)
        self.VGAE = VGAE(encoder, decoder)

    @property
    def __mu__(self) -> Tensor:
        return self.VGAE.__mu__

    @property
    def __logstd__(self) -> Tensor:
        return self.VGAE.__logstd__

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        return self.VGAE.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs) -> Tensor:
        return self.VGAE.encode(*args, **kwargs)

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
    ) -> Tensor:
        return self.VGAE.kl_loss(mu, logstd)
