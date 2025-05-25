from typing import Callable, List, Optional, Tuple

import paddle
from paddle import Tensor
from paddle_geometric.nn.aggr import Aggregation
from paddle_geometric.nn.inits import reset
from paddle_geometric.utils import scatter


class ResNetPotential(paddle.nn.Layer):
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int]):
        super().__init__()
        sizes = [in_channels] + num_layers + [out_channels]
        self.layers = paddle.nn.LayerList([
            paddle.nn.Sequential(
                paddle.nn.Linear(in_size, out_size),
                paddle.nn.LayerNorm(out_size),
                paddle.nn.Tanh()
            )
            for in_size, out_size in zip(sizes[:-2], sizes[1:-1])
        ])
        self.layers.append(paddle.nn.Linear(sizes[-2], sizes[-1]))

        self.res_trans = paddle.nn.LayerList([
            paddle.nn.Linear(in_channels, layer_size)
            for layer_size in num_layers + [out_channels]
        ])

    def forward(self, x: Tensor, y: Tensor, index: Optional[Tensor],
                dim_size: Optional[int] = None) -> Tensor:
        if index is None:
            inp = paddle.concat([x, y.expand([x.shape[0], -1])], axis=1)
        else:
            inp = paddle.concat([x, paddle.gather(y, index)], axis=1)

        h = inp
        for layer, res in zip(self.layers, self.res_trans):
            h = layer(h)
            h = res(inp) + h

        if index is None:
            return h.mean()

        if dim_size is None:
            dim_size = int(paddle.max(index).item() + 1)

        return scatter(h, index, 0, dim_size, reduce='mean').sum()


class MomentumOptimizer(paddle.nn.Layer):
    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9,
                 learnable: bool = True):
        super().__init__()

        self._initial_lr = learning_rate
        self._initial_mom = momentum
        self._lr = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(learning_rate))
        self._lr.stop_gradient = not learnable
        self._mom = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(momentum))
        self._mom.stop_gradient = not learnable
        self.softplus = paddle.nn.Softplus()
        self.sigmoid = paddle.nn.Sigmoid()

    def reset_parameters(self):
        self._lr.set_value(paddle.to_tensor(self._initial_lr))
        self._mom.set_value(paddle.to_tensor(self._initial_mom))

    @property
    def learning_rate(self):
        return self.softplus(self._lr)

    @property
    def momentum(self):
        return self.sigmoid(self._mom)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        index: Optional[Tensor],
        dim_size: Optional[int],
        func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        iterations: int = 5,
    ) -> Tuple[Tensor, float]:

        momentum_buffer = paddle.zeros_like(y)
        for _ in range(iterations):
            val = func(x, y, index, dim_size)
            grad = paddle.grad([val], [y], create_graph=True, retain_graph=True)[0]
            delta = self.learning_rate * grad
            momentum_buffer = self.momentum * momentum_buffer - delta
            y = y + momentum_buffer
        return y


class EquilibriumAggregation(Aggregation):
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int], grad_iter: int = 5, lamb: float = 0.1):
        super().__init__()

        self.potential = ResNetPotential(in_channels + out_channels, 1, num_layers)
        self.optimizer = MomentumOptimizer()
        self.initial_lamb = lamb
        self.lamb = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(lamb))
        self.softplus = paddle.nn.Softplus()
        self.grad_iter = grad_iter
        self.output_dim = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.lamb.set_value(paddle.to_tensor(self.initial_lamb))
        reset(self.optimizer)
        reset(self.potential)

    def init_output(self, dim_size: int) -> Tensor:
        return paddle.zeros([dim_size, self.output_dim], dtype=paddle.float32, stop_gradient=False)

    def reg(self, y: Tensor) -> Tensor:
        return self.softplus(self.lamb) * y.square().sum(axis=-1).mean()

    def energy(self, x: Tensor, y: Tensor, index: Optional[Tensor],
               dim_size: Optional[int] = None):
        return self.potential(x, y, index, dim_size) + self.reg(y)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_index_present(index)

        dim_size = int(paddle.max(index)) + 1 if dim_size is None else dim_size

        with paddle.set_grad_enabled(True):
            y = self.optimizer(x, self.init_output(dim_size), index, dim_size,
                               self.energy, iterations=self.grad_iter)

        return y

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')
