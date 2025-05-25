import logging
from typing import List, Optional, Union

import paddle
from paddle import Tensor

from paddle_geometric.explain import Explanation
from paddle_geometric.explain.algorithm import ExplainerAlgorithm
from paddle_geometric.explain.config import ExplanationType, ModelTaskLevel
from paddle_geometric.nn.conv.message_passing import MessagePassing


class AttentionExplainer(ExplainerAlgorithm):
    r"""An explainer that uses the attention coefficients produced by an
    attention-based GNN (*e.g.*,
    :class:`~paddle_geometric.nn.conv.GATConv`,
    :class:`~paddle_geometric.nn.conv.GATv2Conv`, or
    :class:`~paddle_geometric.nn.conv.TransformerConv`) as edge explanation.
    Attention scores across layers and heads will be aggregated according to
    the :obj:`reduce` argument.

    Args:
        reduce (str, optional): The method to reduce the attention scores
            across layers and heads. (default: :obj:`"max"`)
    """
    def __init__(self, reduce: str = 'max'):
        super().__init__()
        self.reduce = reduce

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
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.shape[0])

        alphas: List[Tensor] = []

        def hook(layer, msg_kwargs, out):
            if 'alpha' in msg_kwargs[0]:
                alphas.append(msg_kwargs[0]['alpha'].detach())
            elif getattr(layer, '_alpha', None) is not None:
                alphas.append(layer._alpha.detach())

        hook_handles = []
        for layer in model.sublayers():  # Register hooks for attention layers
            if (isinstance(layer, MessagePassing) and layer.explain is not False):
                hook_handles.append(layer.register_message_forward_hook(hook))

        model(x, edge_index, **kwargs)

        for handle in hook_handles:  # Remove hooks
            handle.remove()

        if len(alphas) == 0:
            raise ValueError("Could not collect any attention coefficients. "
                             "Please ensure that your model is using "
                             "attention-based GNN layers.")

        for i, alpha in enumerate(alphas):
            alpha = alpha[:edge_index.shape[1]]  # Account for potential self-loops.
            if alpha.ndim == 2:
                alpha = getattr(paddle, self.reduce)(alpha, axis=-1)
                if isinstance(alpha, tuple):  # Handle `paddle.max` tuple output
                    alpha = alpha[0]
            elif alpha.ndim > 2:
                raise ValueError(f"Cannot reduce attention coefficients of "
                                 f"shape {list(alpha.shape)}")
            alphas[i] = alpha

        if len(alphas) > 1:
            alpha = paddle.stack(alphas, axis=-1)
            alpha = getattr(paddle, self.reduce)(alpha, axis=-1)
            if isinstance(alpha, tuple):  # Handle `paddle.max` tuple output
                alpha = alpha[0]
        else:
            alpha = alphas[0]

        alpha = self._post_process_mask(alpha, hard_edge_mask,
                                        apply_sigmoid=False)

        return Explanation(edge_mask=alpha)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.model:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"model explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True
