from typing import Dict, List, Optional, Tuple

import math
import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import Embedding
from paddle.io import DataLoader

from paddle_geometric.index import index2ptr
from paddle_geometric.typing import EdgeType, NodeType, OptTensor
from paddle_geometric.utils import sort_edge_index

EPS = 1e-15


class MetaPath2Vec(nn.Layer):
    r"""The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
    Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper where random walks based
    on a given :obj:`metapath` are sampled in a heterogeneous graph, and node
    embeddings are learned via negative sampling optimization.
    """
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        self.rowptr_dict, self.col_dict, self.rowcount_dict = {}, {}, {}
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = sort_edge_index(edge_index, num_nodes=max(sizes)).cpu()
            rowptr = index2ptr(row, size=sizes[0])
            self.rowptr_dict[keys] = rowptr
            self.col_dict[keys] = col
            self.rowcount_dict[keys] = rowptr[1:] - rowptr[:-1]

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "Found invalid metapath. Ensure that the destination node "
                    "type matches with the source node type across all "
                    "consecutive edge types.")

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = {x[0] for x in metapath} | {x[-1] for x in metapath}
        types = sorted(list(types))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath
                   ] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = paddle.to_tensor(offset)

        # + 1 denotes a dummy node used to link to for isolated nodes.
        self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        self.dummy_idx = count

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type`.
        """
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph.

        Args:
            **kwargs (optional): Arguments of
                :class:`paddle.io.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            batch = sample(
                self.rowptr_dict[edge_type],
                self.col_dict[edge_type],
                self.rowcount_dict[edge_type],
                batch,
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
            ).view(-1)
            rws.append(batch)

        rw = paddle.stack(rws, axis=-1)
        rw.add_(self.offset.view(1, -1))
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return paddle.concat(walks, axis=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = paddle.randint(0, self.num_nodes_dict[keys[-1]],
                                  shape=[batch.shape[0]], dtype='int64')
            rws.append(batch)

        rw = paddle.stack(rws, axis=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return paddle.concat(walks, axis=0)

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = paddle.to_tensor(batch, dtype='int64')
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).reshape([pos_rw.shape[0], 1,
                                                 self.embedding_dim])
        h_rest = self.embedding(rest.reshape([-1])).reshape([pos_rw.shape[0], -1,
                                                            self.embedding_dim])

        out = (h_start * h_rest).sum(axis=-1).reshape([-1])
        pos_loss = -paddle.log(paddle.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).reshape([neg_rw.shape[0], 1,
                                                 self.embedding_dim])
        h_rest = self.embedding(rest.reshape([-1])).reshape([neg_rw.shape[0], -1,
                                                            self.embedding_dim])

        out = (h_start * h_rest).sum(axis=-1).reshape([-1])
        neg_loss = -paddle.log(1 - paddle.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z: Tensor, train_y: Tensor, test_z: Tensor,
             test_y: Tensor, solver: str = "lbfgs", *args, **kwargs) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.embedding.weight.shape[0] - 1}, '
                f'{self.embedding.weight.shape[1]})')


def sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, subset: Tensor,
           num_neighbors: int, dummy_idx: int) -> Tensor:

    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = paddle.rand([subset.shape[0], num_neighbors], dtype=subset.dtype)
    rand *= count.to(rand.dtype).reshape([-1, 1])
    rand = rand.astype('int64') + rowptr[subset].reshape([-1, 1])
    rand = rand.clip(max=col.numel() - 1)  # If last node is isolated.

    col = col[rand] if col.numel() > 0 else rand
    col[mask | (count == 0)] = dummy_idx
    return col
