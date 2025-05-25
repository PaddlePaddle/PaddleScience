from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer, Linear, BatchNorm1D, LayerNorm, InstanceNorm1D, ReLU, Dropout, Sequential
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from paddle_geometric.nn.aggr import Aggregation, MultiAggregation
from paddle_geometric.nn.norm import MessageNorm


class MLP(Sequential):
    def __init__(self, channels: list, norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.0):
        layers = []
        for i in range(1, len(channels)):
            layers.append(Linear(channels[i - 1], channels[i], bias_attr=bias))

            if i < len(channels) - 1:
                if norm == 'batch':
                    layers.append(BatchNorm1D(channels[i]))
                elif norm == 'layer':
                    layers.append(LayerNorm(channels[i]))
                elif norm == 'instance':
                    layers.append(InstanceNorm1D(channels[i]))
                elif norm:
                    raise NotImplementedError(f'Normalization layer "{norm}" not supported.')

                layers.append(ReLU())
                layers.append(Dropout(dropout))

        super().__init__(*layers)


class GENConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, list, Aggregation]] = 'softmax',
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: str = 'batch',
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):

        # Backward compatibility:
        semi_grad = True if aggr == 'softmax_sg' else False
        aggr = 'softmax' if aggr == 'softmax_sg' else aggr
        aggr = 'powermean' if aggr == 'power' else aggr

        if 'aggr_kwargs' not in kwargs:
            if aggr == 'softmax':
                kwargs['aggr_kwargs'] = dict(t=t, learn=learn_t, semi_grad=semi_grad)
            elif aggr == 'powermean':
                kwargs['aggr_kwargs'] = dict(p=p, learn=learn_p)

        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if in_channels[0] != out_channels:
            self.lin_src = Linear(in_channels[0], out_channels, bias_attr=bias)

        if edge_dim is not None and edge_dim != out_channels:
            self.lin_edge = Linear(edge_dim, out_channels, bias_attr=bias)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(out_channels)
        else:
            aggr_out_channels = out_channels

        if aggr_out_channels != out_channels:
            self.lin_aggr_out = Linear(aggr_out_channels, out_channels, bias_attr=bias)

        if in_channels[1] != out_channels:
            self.lin_dst = Linear(in_channels[1], out_channels, bias_attr=bias)

        channels = [out_channels]
        for i in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)
        self.mlp = MLP(channels, norm=norm, bias=bias)

        if msg_norm:
            self.msg_norm = MessageNorm(learn_msg_scale)

    def reset_parameters(self):
        self.mlp.apply(lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None)
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_edge'):
            self.lin_edge.reset_parameters()
        if hasattr(self, 'lin_aggr_out'):
            self.lin_aggr_out.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if hasattr(self, 'lin_src'):
            x = (self.lin_src(x[0]), x[1])

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if hasattr(self, 'lin_aggr_out'):
            out = self.lin_aggr_out(out)

        if hasattr(self, 'msg_norm'):
            h = x[1] if x[1] is not None else x[0]
            assert h is not None
            out = self.msg_norm(h, out)

        x_dst = x[1]
        if x_dst is not None:
            if hasattr(self, 'lin_dst'):
                x_dst = self.lin_dst(x_dst)
            out = out + x_dst

        return self.mlp(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        if edge_attr is not None:
            assert x_j.shape[-1] == edge_attr.shape[-1]

        msg = x_j if edge_attr is None else x_j + edge_attr
        return msg.relu(msg) + self.eps

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
