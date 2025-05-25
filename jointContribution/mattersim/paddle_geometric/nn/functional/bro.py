from typing import Union
import paddle


def bro(
        x: paddle.Tensor,
        batch: paddle.Tensor,
        p: Union[int, str] = 2,
) -> paddle.Tensor:
    r"""The Batch Representation Orthogonality penalty from the `"Improving
    Molecular Graph Neural Network Explainability with Orthonormalization
    and Induced Sparsity" <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization for each graph representation in a mini-batch
    according to

    .. math::
        \mathcal{L}_{\textrm{BRO}}^\mathrm{graph} =
          || \mathbf{HH}^T - \mathbf{I}||_p

    and returns an average over all graphs in the batch.

    Args:
        x (paddle.Tensor): The node feature matrix.
        batch (paddle.Tensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        p (int or str, optional): The norm order to use. (default: :obj:`2`)
    """
    _, counts = paddle.unique(batch, return_counts=True)

    # Prepare diagonal matrices for each graph in the batch
    sequences = paddle.ones_like(batch).split(counts.tolist())
    diags = paddle.stack([
        paddle.diag(x) for x in paddle.nn.utils.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )
    ])

    # Split and pad the input tensor `x` for batch processing
    x_split = x.split(counts.tolist())
    x_padded = paddle.nn.utils.pad_sequence(x_split, batch_first=True, padding_value=0.0)

    # Calculate the BRO loss
    return paddle.sum(
        paddle.norm(x_padded @ x_padded.transpose([0, 2, 1]) - diags, p=p, axis=(1, 2))
    ) / counts.shape[0]
