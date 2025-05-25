from typing import Optional
import paddle
from paddle import Tensor
from paddle_geometric.nn.aggr import Aggregation


class MLPAggregation(Aggregation):
    r"""Performs MLP aggregation in which the elements to aggregate are
    flattened into a single vectorial representation, and are then processed by
    a Multi-Layer Perceptron (MLP), as described in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    .. note::

        :class:`MLPAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~paddle_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices or by calling `Data.sort()`.

    .. warning::

        :class:`MLPAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        max_num_elements (int): The maximum number of elements to aggregate per
            group.
        **kwargs (optional): Additional arguments for `paddle.nn.Sequential`
            MLP layers.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_num_elements: int,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements

        # Define MLP with Paddle's Sequential API
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_channels * max_num_elements, out_channels),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(kwargs.get("dropout", 0.5)),
            paddle.nn.Linear(out_channels, out_channels)
        )

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, paddle.nn.Linear):
                paddle.nn.initializer.XavierUniform()(layer.weight)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=self.max_num_elements)

        x = x.reshape([-1, x.shape[1] * x.shape[2]])
        return self.mlp(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'max_num_elements={self.max_num_elements})')
