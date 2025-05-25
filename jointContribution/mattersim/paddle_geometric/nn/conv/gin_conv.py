from typing import Callable, Optional, Union, Tuple

import paddle
from paddle import Tensor
from paddle.nn import Layer

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.utils import spmm


class GINConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = self.create_parameter([1], default_initializer=paddle.nn.initializer.Constant(eps))
        else:
            self.eps = paddle.to_tensor(eps)
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.apply(lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None)
        if isinstance(self.eps, Tensor):
            self.eps.fill_(self.initial_eps)

    def forward(
            self,
            x: Union[Tensor, Tuple[Tensor, Tensor]],
            edge_index: Tensor,
            size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Tensor, x: Tuple[Tensor, Tensor]) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINEConv(MessagePassing):
    def __init__(self, nn: Layer, eps: float = 0.0, train_eps: bool = False, edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = self.create_parameter([1], default_initializer=paddle.nn.initializer.Constant(eps))
        else:
            self.eps = paddle.to_tensor(eps)
        if edge_dim is not None:
            if hasattr(self.nn[0], 'weight'):
                in_channels = self.nn[0].weight.shape[0]
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.apply(lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None)
        if isinstance(self.eps, Tensor):
            self.eps.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
            self,
            x: Union[Tensor, Tuple[Tensor, Tensor]],
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.shape[-1] != edge_attr.shape[-1]:
            raise ValueError("Node and edge feature dimensionalities do not match. Set 'edge_dim' for 'GINEConv'.")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return paddle.nn.functional.relu(x_j + edge_attr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
