import logging
from typing import Optional, Union

import paddle
from paddle import Tensor
from paddle.nn import ReLU, Sequential
import paddle.nn.functional as F

from paddle_geometric.explain import Explanation
from paddle_geometric.explain.algorithm import ExplainerAlgorithm
from paddle_geometric.explain.algorithm.utils import clear_masks, set_masks
from paddle_geometric.explain.config import ExplanationType, ModelMode, ModelTaskLevel
from paddle_geometric.nn import Linear
from paddle_geometric.nn.inits import reset
from paddle_geometric.utils import get_embeddings


class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~paddle_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.01,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.optimizer = paddle.optimizer.Adam(parameters=self.mlp.parameters(), learning_rate=lr)
        self._curr_epoch = -1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
        self,
        epoch: int,
        model: paddle.nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (paddle.nn.Layer): The model to explain.
            x (paddle.Tensor): The input node features of a homogeneous graph.
            edge_index (paddle.Tensor): The input edge indices of a homogeneous graph.
            target (paddle.Tensor): The target of the model.
            index (int or paddle.Tensor, optional): The index of the model output to explain.
                Needs to be a single index.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in '{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided in '{self.__class__.__name__}' for node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' argument in '{self.__class__.__name__}'")

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        self.optimizer.clear_grad()
        temperature = self._get_temperature(epoch)

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).flatten()
        edge_mask = self._concrete_sample(logits, temperature)
        set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index, num_nodes=x.shape[0])
            edge_mask = edge_mask[hard_edge_mask]

        y_hat, y = model(x, edge_index, **kwargs), target

        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss = self._loss(y_hat, y, edge_mask)
        loss.backward()
        self.optimizer.step()

        clear_masks(model)
        self._curr_epoch = epoch

        return float(loss.numpy())

    def forward(
        self,
        model: paddle.nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in '{self.__class__.__name__}'")

        if self._curr_epoch < self.epochs - 1:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully trained (got {self._curr_epoch + 1} epochs from {self.epochs} epochs). Please first train the underlying explainer model by running `explainer.algorithm.train(...)`.")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided in '{self.__class__.__name__}' for node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' argument in '{self.__class__.__name__}'")

            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index, num_nodes=x.shape[0])

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).flatten()

        edge_mask = self._post_process_mask(logits, hard_edge_mask, apply_sigmoid=True)

        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports phenomenon explanations got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports node-level or graph-level explanations got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support explaining input node features got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor, index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].reshape([1, -1]).expand([zs[0].shape[0], -1]))
        return paddle.concat(zs, axis=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * (temp[1] / temp[0]) ** (epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * paddle.rand(logits.shape) + bias
        return (paddle.log(eps) - paddle.log(1 - eps) + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = F.sigmoid(edge_mask)
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * paddle.log(mask + 1e-15) - (1 - mask) * paddle.log(1 - mask + 1e-15)
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss
