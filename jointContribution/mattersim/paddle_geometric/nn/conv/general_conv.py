from typing import Union, Tuple, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, Linear, Identity, LayerList
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.utils import softmax

class GeneralConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: Optional[int],
        in_edge_channels: Optional[int] = None,
        aggr: str = "add",
        skip_linear: str = False,
        directed_msg: bool = True,
        heads: int = 1,
        attention: bool = False,
        attention_type: str = "additive",
        l2_normalize: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels
        self.aggr = aggr
        self.skip_linear = skip_linear
        self.directed_msg = directed_msg
        self.heads = heads
        self.attention = attention
        self.attention_type = attention_type
        self.normalize_l2 = l2_normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.directed_msg:
            self.lin_msg = Linear(in_channels[0], out_channels * self.heads, bias_attr=bias)
        else:
            self.lin_msg = Linear(in_channels[0], out_channels * self.heads, bias_attr=bias)
            self.lin_msg_i = Linear(in_channels[0], out_channels * self.heads, bias_attr=bias)

        if self.skip_linear or self.in_channels != self.out_channels:
            self.lin_self = Linear(in_channels[1], out_channels, bias_attr=bias)
        else:
            self.lin_self = Identity()

        if self.in_edge_channels is not None:
            self.lin_edge = Linear(in_edge_channels, out_channels * self.heads, bias_attr=bias)

        # Attention parameters
        if self.attention:
            if self.attention_type == 'additive':
                self.att_msg = self.create_parameter(
                    shape=[1, self.heads, self.out_channels],
                    default_initializer=paddle.nn.initializer.XavierUniform())
            elif self.attention_type == 'dot_product':
                self.scaler = paddle.to_tensor(paddle.sqrt(paddle.to_tensor(out_channels, dtype='float32')))
            else:
                raise ValueError(f"Attention type '{self.attention_type}' not supported")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_msg.weight.set_value(paddle.nn.initializer.XavierUniform()(self.lin_msg.weight.shape))
        if hasattr(self.lin_self, 'reset_parameters'):
            self.lin_self.reset_parameters()
        if self.in_edge_channels is not None:
            self.lin_edge.weight.set_value(paddle.nn.initializer.XavierUniform()(self.lin_edge.weight.shape))
        if self.attention and self.attention_type == 'additive':
            paddle.nn.initializer.XavierUniform()(self.att_msg)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)
        x_self = x[1]

        # propagate_type: (x: Tuple[Tensor, Tensor], edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        out = out.mean(axis=1)  # Aggregating heads
        out = out + self.lin_self(x_self)
        if self.normalize_l2:
            out = F.normalize(out, p=2, axis=-1)
        return out

    def message_basic(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor]):
        if self.directed_msg:
            x_j = self.lin_msg(x_j)
        else:
            x_j = self.lin_msg(x_j) + self.lin_msg_i(x_i)
        if edge_attr is not None:
            x_j = x_j + self.lin_edge(edge_attr)
        return x_j

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor,
                size_i: Tensor, edge_attr: Tensor) -> Tensor:
        x_j_out = self.message_basic(x_i, x_j, edge_attr)
        x_j_out = x_j_out.reshape([-1, self.heads, self.out_channels])
        if self.attention:
            if self.attention_type == 'dot_product':
                x_i_out = self.message_basic(x_j, x_i, edge_attr)
                x_i_out = x_i_out.reshape([-1, self.heads, self.out_channels])
                alpha = paddle.sum(x_i_out * x_j_out, axis=-1) / self.scaler
            else:
                alpha = paddle.sum(x_j_out * self.att_msg, axis=-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            alpha = alpha.reshape([-1, self.heads, 1])
            return x_j_out * alpha
        else:
            return x_j_out
