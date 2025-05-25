from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.nn.aggr import Aggregation
from paddle_geometric.utils import softmax


class Set2Set(Aggregation):
    r"""The Set2Set aggregation operator based on iterative content-based
    attention, as described in the `"Order Matters: Sequence to sequence for
    Sets" <https://arxiv.org/abs/1511.06391>`_ paper.

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        **kwargs (optional): Additional arguments of :class:`paddle.nn.LSTM`.
    """
    def __init__(self, in_channels: int, processing_steps: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.lstm = paddle.nn.LSTM(self.out_channels, in_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lstm.sublayers():
            if isinstance(layer, paddle.nn.Linear):
                paddle.nn.initializer.XavierUniform()(layer.weight)
                if layer.bias is not None:
                    paddle.nn.initializer.Constant(0.0)(layer.bias)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)

        h = (
            paddle.zeros((self.lstm.num_layers, dim_size, x.shape[-1]), dtype=x.dtype),
            paddle.zeros((self.lstm.num_layers, dim_size, x.shape[-1]), dtype=x.dtype)
        )
        q_star = paddle.zeros([dim_size, self.out_channels], dtype=x.dtype)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.squeeze(0).reshape([dim_size, self.in_channels])
            e = (x * q[index]).sum(axis=-1, keepdim=True)
            a = softmax(e, index, ptr, dim_size, dim)
            r = self.reduce(a * x, index, ptr, dim_size, dim, reduce='sum')
            q_star = paddle.concat([q, r], axis=-1)

        return q_star

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
