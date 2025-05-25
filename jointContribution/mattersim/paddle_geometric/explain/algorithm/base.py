from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor

from paddle_geometric.explain import Explanation, HeteroExplanation
from paddle_geometric.explain.config import (
    ExplainerConfig,
    ModelConfig,
    ModelReturnType,
)
from paddle_geometric.nn import MessagePassing
from paddle_geometric.utils import k_hop_subgraph


class ExplainerAlgorithm(paddle.nn.Layer):
    r"""An abstract base class for implementing explainer algorithms."""
    @abstractmethod
    def forward(
        self,
        model: paddle.nn.Layer,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[str, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        r"""Computes the explanation."""

    @abstractmethod
    def supports(self) -> bool:
        r"""Checks if the explainer supports the user-defined settings provided
        in :obj:`self.explainer_config`, :obj:`self.model_config`.
        """

    ###########################################################################

    @property
    def explainer_config(self) -> ExplainerConfig:
        if not hasattr(self, '_explainer_config'):
            raise ValueError(
                f"The explanation algorithm '{self.__class__.__name__}' is "
                f"not yet connected to any explainer configuration. Please "
                f"call `{self.__class__.__name__}.connect(...)` before "
                f"proceeding.")
        return self._explainer_config

    @property
    def model_config(self) -> ModelConfig:
        if not hasattr(self, '_model_config'):
            raise ValueError(
                f"The explanation algorithm '{self.__class__.__name__}' is "
                f"not yet connected to any model configuration. Please call "
                f"`{self.__class__.__name__}.connect(...)` before "
                f"proceeding.")
        return self._model_config

    def connect(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ):
        self._explainer_config = ExplainerConfig.cast(explainer_config)
        self._model_config = ModelConfig.cast(model_config)

        if not self.supports():
            raise ValueError(
                f"The explanation algorithm '{self.__class__.__name__}' does "
                f"not support the given explanation settings.")

    # Helper functions ########################################################

    @staticmethod
    def _post_process_mask(
        mask: Optional[Tensor],
        hard_mask: Optional[Tensor] = None,
        apply_sigmoid: bool = True,
    ) -> Optional[Tensor]:
        if mask is None:
            return mask

        mask = mask.detach()

        if apply_sigmoid:
            mask = F.sigmoid(mask)

        if hard_mask is not None and mask.shape[0] == hard_mask.shape[0]:
            mask = paddle.where(hard_mask, mask, paddle.zeros_like(mask))

        return mask

    @staticmethod
    def _get_hard_masks(
        model: paddle.nn.Layer,
        node_index: Optional[Union[int, Tensor]],
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if node_index is None:
            return None, None  # Consider all nodes and edges.

        index, _, _, edge_mask = k_hop_subgraph(
            node_index,
            num_hops=ExplainerAlgorithm._num_hops(model),
            edge_index=edge_index,
            num_nodes=num_nodes,
            flow=ExplainerAlgorithm._flow(model),
        )

        node_mask = paddle.zeros([num_nodes], dtype='bool')
        node_mask[index] = True

        return node_mask, edge_mask

    @staticmethod
    def _num_hops(model: paddle.nn.Layer) -> int:
        num_hops = 0
        for module in model.sublayers():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    @staticmethod
    def _flow(model: paddle.nn.Layer) -> str:
        for module in model.sublayers():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def _loss_binary_classification(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.return_type == ModelReturnType.raw:
            loss_fn = F.binary_cross_entropy_with_logits
        elif self.model_config.return_type == ModelReturnType.probs:
            loss_fn = F.binary_cross_entropy
        else:
            raise ValueError("Invalid ModelReturnType for binary classification")

        return loss_fn(y_hat.reshape(y.shape), y.astype('float32'))

    def _loss_multiclass_classification(
        self,
        y_hat: Tensor,
        y: Tensor,
    ) -> Tensor:
        if self.model_config.return_type == ModelReturnType.raw:
            loss_fn = F.cross_entropy
        elif self.model_config.return_type == ModelReturnType.probs:
            loss_fn = F.cross_entropy
        elif self.model_config.return_type == ModelReturnType.log_probs:
            loss_fn = F.nll_loss
        else:
            raise ValueError("Invalid ModelReturnType for multiclass classification")

        return loss_fn(y_hat, y)

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        assert self.model_config.return_type == ModelReturnType.raw
        return F.mse_loss(y_hat, y)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
