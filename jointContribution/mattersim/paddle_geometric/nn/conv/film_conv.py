import copy
from typing import Callable, Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle.nn import LayerList, ReLU, Linear
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import reset

from paddle_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    paddle_sparse,
)

class FiLMConv(MessagePassing):
    r"""The FiLM graph convolutional operator from the
    `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
    <https://arxiv.org/abs/1906.12192>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{r \in \mathcal{R}}
        \sum_{j \in \mathcal{N}(i)} \sigma \left(
        \boldsymbol{\gamma}_{r,i} \odot \mathbf{W}_r \mathbf{x}_j +
        \boldsymbol{\beta}_{r,i} \right)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        num_relations (int, optional): Number of relations. (default: :obj:`1`)
        nn (paddle.nn.Layer, optional): The neural network :math:`g` that
            maps node features :obj:`x_i` of shape
            :obj:`[-1, in_channels]` to shape :obj:`[-1, 2 * out_channels]`.
            If set to :obj:`None`, :math:`g` will be implemented as a single
            linear layer. (default: :obj:`None`)
        act (callable, optional): Activation function :math:`\sigma`.
            (default: :meth:`paddle.nn.ReLU()`)
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int = 1,
            nn: Optional[Callable] = None,
            act: Optional[Callable] = ReLU(),
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = max(num_relations, 1)
        self.act = act

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lins = LayerList()
        self.films = LayerList()
        for _ in range(num_relations):
            self.lins.append(Linear(in_channels[0], out_channels, bias_attr=False))
            if nn is None:
                film = Linear(in_channels[1], 2 * out_channels)
            else:
                film = copy.deepcopy(nn)
            self.films.append(film)

        self.lin_skip = Linear(in_channels[1], self.out_channels, bias_attr=False)
        if nn is None:
            self.film_skip = Linear(in_channels[1], 2 * self.out_channels, bias_attr=False)
        else:
            self.film_skip = copy.deepcopy(nn)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin, film in zip(self.lins, self.films):
            lin.weight.set_value(paddle.nn.initializer.XavierUniform()(lin.weight.shape))
            reset(film)
        self.lin_skip.weight.set_value(paddle.nn.initializer.XavierUniform()(self.lin_skip.weight.shape))
        reset(self.film_skip)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_type: OptTensor = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        beta, gamma = paddle.split(self.film_skip(x[1]), self.out_channels, axis=-1)
        out = gamma * self.lin_skip(x[1]) + beta
        if self.act is not None:
            out = self.act(out)

        # propagate_type: (x: Tensor, beta: Tensor, gamma: Tensor)
        if self.num_relations <= 1:
            beta, gamma = paddle.split(self.films[0](x[1]), self.out_channels, axis=-1)
            out = out + self.propagate(edge_index, x=self.lins[0](x[0]), beta=beta, gamma=gamma)
        else:
            for i, (lin, film) in enumerate(zip(self.lins, self.films)):
                beta, gamma = paddle.split(film(x[1]), self.out_channels, axis=-1)
                if isinstance(edge_index, SparseTensor):
                    _edge_type = edge_index.coo().values()
                    mask = _edge_type == i
                    adj_t = paddle_sparse.masked_select_nnz(edge_index, mask, layout='coo')
                    out = out + self.propagate(adj_t, x=lin(x[0]), beta=beta, gamma=gamma)
                else:
                    assert edge_type is not None
                    mask = edge_type == i
                    out = out + self.propagate(edge_index[:, mask], x=lin(x[0]), beta=beta, gamma=gamma)

        return out

    def message(self, x_j: Tensor, beta_i: Tensor, gamma_i: Tensor) -> Tensor:
        out = gamma_i * x_j + beta_i
        if self.act is not None:
            out = self.act(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
