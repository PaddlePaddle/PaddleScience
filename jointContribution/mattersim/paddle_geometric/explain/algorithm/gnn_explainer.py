from math import sqrt
from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor, nn

from paddle_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from paddle_geometric.explain.algorithm import ExplainerAlgorithm
from paddle_geometric.explain.algorithm.utils import clear_masks, set_masks
from paddle_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::
        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~paddle_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
        self,
        model: nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = paddle.optimizer.Adam(parameters, learning_rate=self.lr)

        for i in range(self.epochs):
            optimizer.clear_grad()

            h = x if self.node_mask is None else x * paddle.nn.functional.sigmoid(self.node_mask)
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            if i == 0:
                if self.node_mask is not None and self.node_mask.grad is not None:
                    self.hard_node_mask = self.node_mask.grad != 0.0
                if self.edge_mask is not None and self.edge_mask.grad is not None:
                    self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.place
        (N, F), E = x.shape, edge_index.shape[1]

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = paddle.create_parameter(
                shape=[N, 1], dtype=x.dtype, default_initializer=paddle.nn.initializer.Normal(std=std)
            )
        elif node_mask_type == MaskType.attributes:
            self.node_mask = paddle.create_parameter(
                shape=[N, F], dtype=x.dtype, default_initializer=paddle.nn.initializer.Normal(std=std)
            )
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = paddle.create_parameter(
                shape=[1, F], dtype=x.dtype, default_initializer=paddle.nn.initializer.Normal(std=std)
            )

        if edge_mask_type == MaskType.object:
            std = paddle.nn.initializer.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = paddle.create_parameter(
                shape=[E], dtype=x.dtype, default_initializer=paddle.nn.initializer.Normal(std=std)
            )

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        if self.hard_edge_mask is not None:
            m = paddle.nn.functional.sigmoid(self.edge_mask[self.hard_edge_mask])
            edge_reduce = getattr(paddle, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * paddle.log(m + self.coeffs['EPS']) - (
                1 - m) * paddle.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            m = paddle.nn.functional.sigmoid(self.node_mask[self.hard_node_mask])
            node_reduce = getattr(paddle, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * paddle.log(m + self.coeffs['EPS']) - (
                1 - m) * paddle.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
