from typing import Optional
import paddle
from paddle import Tensor
from paddle.nn import GRU

from paddle_geometric.nn.aggr import Aggregation


class GRUAggregation(Aggregation):
    r"""Performs GRU aggregation in which the elements to aggregate are
    interpreted as a sequence, as described in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    .. note::

        :class:`GRUAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~paddle_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices or by calling `Data.sort()`.

    .. warning::

        :class:`GRUAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of :class:`paddle.nn.GRU`.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = GRU(in_channels, out_channels, time_major=False, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gru.named_sublayers():
            if isinstance(layer, paddle.nn.GRUCell):
                layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=max_num_elements)

        output, _ = self.gru(x)
        return output[:, -1]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
