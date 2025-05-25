import paddle
import paddle.nn as nn
from paddle import Tensor


class MaskLabel(nn.Layer):
    r"""The label embedding and masking layer from the `"Masked Label
    Prediction: Unified Message Passing Model for Semi-Supervised
    Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    Here, node labels :obj:`y` are merged to the initial node features :obj:`x`
    for a subset of their nodes according to :obj:`mask`.

    .. note::

        For an example of using :class:`MaskLabel`, see
        `examples/unimp_arxiv.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/unimp_arxiv.py>`_.


    Args:
        num_classes (int): The number of classes.
        out_channels (int): Size of each output sample.
        method (str, optional): If set to :obj:`"add"`, label embeddings are
            added to the input. If set to :obj:`"concat"`, label embeddings are
            concatenated. In case :obj:`method="add"`, then :obj:`out_channels`
            needs to be identical to the input dimensionality of node features.
            (default: :obj:`"add"`)
    """
    def __init__(self, num_classes: int, out_channels: int,
                 method: str = "add"):
        super().__init__()

        self.method = method
        if method not in ["add", "concat"]:
            raise ValueError(
                f"'method' must be either 'add' or 'concat' (got '{method}')")

        self.emb = nn.Embedding(num_classes, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.emb.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """"""  # noqa: D419
        if self.method == "concat":
            out = paddle.zeros([y.shape[0], self.emb.weight.shape[-1]], dtype='float32')
            out[mask] = self.emb(y[mask])
            return paddle.concat([x, out], axis=-1)
        else:
            x = x.clone()
            x[mask] += self.emb(y[mask])
            return x

    @staticmethod
    def ratio_mask(mask: Tensor, ratio: float):
        r"""Modifies :obj:`mask` by setting :obj:`ratio` of :obj:`True`
        entries to :obj:`False`. Does not operate in-place.

        Args:
            mask (paddle.Tensor): The mask to re-mask.
            ratio (float): The ratio of entries to keep.
        """
        n = int(mask.sum())
        out = mask.clone()
        out[mask] = paddle.rand([n], dtype=mask.dtype, device=mask.device) < ratio
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
