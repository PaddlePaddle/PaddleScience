from dataclasses import dataclass
from typing import Optional

import paddle
from paddle import Tensor
import paddle_geometric.typing
from paddle_geometric.nn.pool.select import SelectOutput


@dataclass(init=False)
class ConnectOutput:
    r"""The output of the :class:`Connect` method, which holds the coarsened
    graph structure, and optional pooled edge features and batch vectors.

    Args:
        edge_index (paddle.Tensor): The edge indices of the coarsened graph.
        edge_attr (paddle.Tensor, optional): The pooled edge features of the
            coarsened graph. (default: :obj:`None`)
        batch (paddle.Tensor, optional): The pooled batch vector of the
            coarsened graph. (default: :obj:`None`)
    """
    edge_index: Tensor
    edge_attr: Optional[Tensor] = None
    batch: Optional[Tensor] = None

    def __init__(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ):
        if edge_index.ndim != 2:
            raise ValueError(f"Expected 'edge_index' to be two-dimensional "
                             f"(got {edge_index.ndim} dimensions)")

        if edge_index.shape[0] != 2:
            raise ValueError(f"Expected 'edge_index' to have size '2' in the "
                             f"first dimension (got '{edge_index.shape[0]}')")

        if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(f"Expected 'edge_index' and 'edge_attr' to "
                             f"hold the same number of edges (got "
                             f"{edge_index.shape[1]} and {edge_attr.shape[0]} "
                             f"edges)")

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch




class Connect(paddle.nn.Layer):
    r"""An abstract base class for implementing custom edge connection
    operators as described in the `"Understanding Pooling in Graph Neural
    Networks" <https://arxiv.org/abs/1905.05178>`_ paper.

    Specifically, :class:`Connect` determines for each pair of supernodes the
    presence or absence of an edge based on the existing edges between the
    nodes in the two supernodes.
    The operator also computes pooled edge features and batch vectors
    (if present).
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(
        self,
        select_output: SelectOutput,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> ConnectOutput:
        r"""Forward pass.

        Args:
            select_output (SelectOutput): The output of :class:`Select`.
            edge_index (paddle.Tensor): The edge indices.
            edge_attr (paddle.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        raise NotImplementedError

    @staticmethod
    def get_pooled_batch(
        select_output: SelectOutput,
        batch: Optional[Tensor],
    ) -> Optional[Tensor]:
        r"""Returns the batch vector of the coarsened graph.

        Args:
            select_output (SelectOutput): The output of :class:`Select`.
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
        """
        if batch is None:
            return batch

        out = paddle.arange(select_output.num_clusters, device=batch.device)
        return out.scatter_(0, select_output.cluster_index,
                            batch[select_output.node_index])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
