from typing import Any, Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.utils import scatter


class SparseCrossEntropy(paddle.autograd.PyLayer):
    # We implement our own custom autograd function for this to avoid the
    # double gradient computation to `inputs`.
    @staticmethod
    def forward(
            ctx: Any,
            inputs: Tensor,
            edge_label_index: Tensor,
            edge_label_weight: Optional[Tensor],
    ) -> Tensor:
        assert inputs.ndim == 2

        logsumexp = paddle.logsumexp(inputs, axis=-1)
        ctx.save_for_backward(inputs, edge_label_index, edge_label_weight,
                              logsumexp)

        out = paddle.index_select(inputs, edge_label_index[1], axis=0)
        out = paddle.index_select(out, edge_label_index[0], axis=1)
        out = -out + paddle.index_select(logsumexp, edge_label_index[0], axis=0)

        if edge_label_weight is not None:
            out *= edge_label_weight

        return out.sum() / inputs.shape[0]

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[Tensor, None, None]:
        inputs, edge_label_index, edge_label_weight, logsumexp = (
            ctx.saved_tensor())

        grad_out = grad_out / inputs.shape[0]
        grad_out = grad_out.expand([edge_label_index.shape[1]])

        if edge_label_weight is not None:
            grad_out = grad_out * edge_label_weight

        grad_logsumexp = scatter(grad_out, edge_label_index[0], dim=0,
                                 dim_size=inputs.shape[0], reduce='sum')

        # Gradient computation of `logsumexp`: `grad * (self - result).exp()`
        grad_input = inputs - logsumexp.unsqueeze(-1)
        grad_input = paddle.exp(grad_input)
        grad_input *= grad_logsumexp.unsqueeze(-1)

        grad_input[edge_label_index[0], edge_label_index[1]] -= grad_out

        return grad_input, None, None


def sparse_cross_entropy(
        inputs: Tensor,
        edge_label_index: Tensor,
        edge_label_weight: Optional[Tensor] = None,
) -> Tensor:
    r"""A sparse-label variant of :func:`paddle.nn.functional.cross_entropy`.
    In particular, the binary target matrix is solely given by sparse indices
    :obj:`edge_label_index`.

    Args:
        inputs (Tensor): The predicted unnormalized logits of shape
            :obj:`[batch_size, num_classes]`.
        edge_label_index (Tensor): The sparse ground-truth indices with
            shape :obj:`[2, num_labels]`.
        edge_label_weight (Tensor, optional): The weight of ground-truth
            indices with shape :obj:`[num_labels]`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Example:
        >>> inputs = paddle.randn([2, 3])
        >>> edge_label_index = paddle.to_tensor([
        ...     [0, 0, 1],
        ...     [0, 1, 2],
        ... ])
        >>> loss = sparse_cross_entropy(inputs, edge_label_index)
        tensor(1.2919)
    """
    if edge_label_weight is not None:
        assert not edge_label_weight.stop_gradient

    return SparseCrossEntropy.apply(
        inputs,
        edge_label_index,
        edge_label_weight,
    )
