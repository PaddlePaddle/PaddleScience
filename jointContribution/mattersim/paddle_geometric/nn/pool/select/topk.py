from typing import Callable, Optional, Union

import paddle
from paddle import Tensor

from paddle_geometric.nn.inits import uniform
from paddle_geometric.nn.pool.select import Select, SelectOutput
from paddle_geometric.nn.resolver import activation_resolver
from paddle_geometric.utils import cumsum, scatter, softmax


# TODO (matthias) Document this method.
def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clip(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.shape[0]), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.shape[0], ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.astype(x.dtype)).ceil().to(paddle.int64)

        x, x_perm = paddle.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = paddle.sort(batch, descending=False, stable=True)

        arange = paddle.arange(x.shape[0], dtype=paddle.int64, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")


class SelectTopK(Select):
    r"""Selects the top-:math:`k` nodes with highest projection scores from the
    `"Graph U-Nets" <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}
            \right)

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

    where :math:`\mathbf{p}` is the learnable projection vector.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.act = activation_resolver(act)

        self.weight = paddle.create_parameter(shape=[1, in_channels], dtype='float32')

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> SelectOutput:
        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=paddle.int64)

        x = x.view(-1, 1) if x.dim() == 1 else x
        score = (x * self.weight).sum(axis=-1)

        if self.min_score is None:
            score = self.act(score / self.weight.norm(p=2, axis=-1))
        else:
            score = softmax(score, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)

        return SelectOutput(
            node_index=node_index,
            num_nodes=x.shape[0],
            cluster_index=paddle.arange(node_index.shape[0], device=x.device),
            num_clusters=node_index.shape[0],
            weight=score[node_index],
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f'ratio={self.ratio}'
        else:
            arg = f'min_score={self.min_score}'
        return f'{self.__class__.__name__}({self.in_channels}, {arg})'
