from typing import Callable, Optional, Union

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import reset
from paddle_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

try:
    from paddle_cluster import knn
except ImportError:
    knn = None


class EdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.

    Args:
        nn (paddle.nn.Layer): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`paddle.nn.Sequential`.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
    """

    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(paddle.concat([x_i, x_j - x_i], axis=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class DynamicEdgeConv(MessagePassing):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper,
    where the graph is dynamically constructed using nearest neighbors.

    Args:
        nn (paddle.nn.Layer): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`.
        k (int): Number of nearest neighbors.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            (default: :obj:`1`)
    """

    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        if knn is None:
            raise ImportError('`DynamicEdgeConv` requires `paddle-cluster`.')

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        batch: Union[OptTensor, Optional[PairTensor]] = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if x[0].ndim != 2:
            raise ValueError("Static graphs are not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])
        return self.propagate(edge_index, x=x)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(paddle.concat([x_i, x_j - x_i], axis=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'
