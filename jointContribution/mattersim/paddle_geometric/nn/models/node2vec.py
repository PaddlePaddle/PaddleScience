from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle_geometric.index import index2ptr
from paddle_geometric.utils import sort_edge_index
from paddle_geometric.typing import Adj
from paddle_geometric.utils.num_nodes import maybe_num_nodes
from paddle.nn import Embedding

EPS = 1e-15


class Node2Vec(nn.Layer):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.
    """
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()

        # Determine the random walk function based on available libraries
        if p == 1.0 and q == 1.0:
            # This is a simplified example, you would replace the logic
            # to implement or import your own random walk function
            self.random_walk_fn = self.random_walk
        else:
            raise ImportError(f"Node2Vec requires custom random walk function")

        # Get number of nodes
        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        # Define embedding layer
        self.embedding = Embedding(self.num_nodes, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs):
        """Returns a DataLoader to sample walks."""
        return paddle.io.DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch, self.walk_length, self.p, self.q)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return paddle.concat(walks, axis=0)

    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = paddle.randint(0, self.num_nodes, shape=(batch.shape[0], self.walk_length), dtype=batch.dtype)
        rw = paddle.concat([batch.unsqueeze(-1), rw], axis=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return paddle.concat(walks, axis=0)

    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = paddle.to_tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).reshape([pos_rw.shape[0], 1, self.embedding_dim])
        h_rest = self.embedding(rest.reshape([-1])).reshape([pos_rw.shape[0], -1, self.embedding_dim])

        out = (h_start * h_rest).sum(axis=-1).reshape([-1])
        pos_loss = -paddle.log(paddle.sigmoid(out) + EPS).mean()

        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).reshape([neg_rw.shape[0], 1, self.embedding_dim])
        h_rest = self.embedding(rest.reshape([-1])).reshape([neg_rw.shape[0], -1, self.embedding_dim])

        out = (h_start * h_rest).sum(axis=-1).reshape([-1])
        neg_loss = -paddle.log(1 - paddle.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z: Tensor, train_y: Tensor, test_z: Tensor, test_y: Tensor, solver: str = 'lbfgs', *args, **kwargs) -> float:
        """Evaluates latent space quality via a logistic regression downstream task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args, **kwargs).fit(train_z.detach().cpu().numpy(),
                                                                   train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.embedding.weight.shape[0]}, {self.embedding.weight.shape[1]})'


