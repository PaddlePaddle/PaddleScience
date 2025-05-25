from typing import Optional, Tuple

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import Layer


class MetaLayer(nn.Layer):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity).
    The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
    as well as graph-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the modules :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    To allow for batch-wise graph processing, all callable functions take an
    additional argument :obj:`batch`, which determines the assignment of
    edges or nodes to their specific graphs.

    Args:
        edge_model (paddle.nn.Layer, optional): A callable which updates a
            graph's edge features based on its source and target node features,
            its current edge features and its global features.
            (default: :obj:`None`)
        node_model (paddle.nn.Layer, optional): A callable which updates a
            graph's node features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (paddle.nn.Layer, optional): A callable which updates a
            graph's global features based on its node features, its graph
            connectivity, its edge features and its current global features.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        edge_model: Optional[nn.Layer] = None,
        node_model: Optional[nn.Layer] = None,
        global_model: Optional[nn.Layer] = None,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The node features.
            edge_index (paddle.Tensor): The edge indices.
            edge_attr (paddle.Tensor, optional): The edge features.
                (default: :obj:`None`)
            u (paddle.Tensor, optional): The global graph features.
                (default: :obj:`None`)
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')
