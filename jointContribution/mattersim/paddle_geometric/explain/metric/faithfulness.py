from typing import Optional

import paddle
import paddle.nn.functional as F

from paddle_geometric.explain import Explainer, Explanation
from paddle_geometric.explain.config import MaskType, ModelMode, ModelReturnType


def unfaithfulness(
    explainer: Explainer,
    explanation: Explanation,
    top_k: Optional[int] = None,
) -> float:
    r"""Evaluates how faithful an :class:`~torch_geometric.explain.Explanation`
    is to an underyling GNN predictor, as described in the
    `"Evaluating Explainability for Graph Neural Networks"
    <https://arxiv.org/abs/2208.09339>`_ paper.

    In particular, the graph explanation unfaithfulness metric is defined as

    .. math::
        \textrm{GEF}(y, \hat{y}) = 1 - \exp(- \textrm{KL}(y || \hat{y}))

    where :math:`y` refers to the prediction probability vector obtained from
    the original graph, and :math:`\hat{y}` refers to the prediction
    probability vector obtained from the masked subgraph.
    Finally, the Kullback-Leibler (KL) divergence score quantifies the distance
    between the two probability distributions.

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
        top_k (int, optional): If set, will only keep the original values of
            the top-:math:`k` node features identified by an explanation.
            If set to :obj:`None`, will use :obj:`explanation.node_mask` as it
            is for masking node features. (default: :obj:`None`)
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    if top_k is not None and explainer.node_mask_type == MaskType.object:
        raise ValueError("Cannot apply top-k feature selection based on a "
                         "node mask of type 'object'")

    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    x, edge_index = explanation.x, explanation.edge_index
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.get('prediction')
    if y is None:  # == ExplanationType.phenomenon
        y = explainer.get_prediction(x, edge_index, **kwargs)

    if node_mask is not None and top_k is not None:
        feat_importance = node_mask.sum(axis=0)
        _, top_k_index = paddle.topk(feat_importance, top_k)
        node_mask = paddle.zeros_like(node_mask)
        node_mask[:, top_k_index] = 1.0

    y_hat = explainer.get_masked_prediction(x, edge_index, node_mask,
                                            edge_mask, **kwargs)

    if explanation.get('index') is not None:
        y, y_hat = y[explanation['index']], y_hat[explanation['index']]

    if explainer.model_config.return_type == ModelReturnType.raw:
        y, y_hat = F.softmax(y, axis=-1), F.softmax(y_hat, axis=-1)
    elif explainer.model_config.return_type == ModelReturnType.log_probs:
        y, y_hat = paddle.exp(y), paddle.exp(y_hat)

    kl_div = F.kl_div(paddle.log(y), y_hat, reduction='batchmean')
    return 1 - float(paddle.exp(-kl_div))
