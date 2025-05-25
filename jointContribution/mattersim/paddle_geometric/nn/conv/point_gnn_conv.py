import paddle
from paddle import Tensor
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import reset
from paddle_geometric.typing import Adj


class PointGNNConv(MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.

    Args:
        mlp_h (paddle.nn.Layer): A neural network that maps node features
            of size :math:`F_{in}` to three-dimensional coordination offsets.
        mlp_f (paddle.nn.Layer): A neural network that computes :math:`e_{j,i}`
            from the features of neighbors and the three-dimensional vector
            `pos_j - pos_i + Delta pos_i`.
        mlp_g (paddle.nn.Layer): A neural network that maps the aggregated edge
            features back to the original feature dimension.
        **kwargs (optional): Additional arguments of
            `paddle_geometric.nn.conv.MessagePassing`.

    """
    def __init__(
        self,
        mlp_h: paddle.nn.Layer,
        mlp_f: paddle.nn.Layer,
        mlp_g: paddle.nn.Layer,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.mlp_h = mlp_h
        self.mlp_f = mlp_f
        self.mlp_g = mlp_g

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_f)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor, pos: Tensor)
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.mlp_g(out)
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor,
                x_j: Tensor) -> Tensor:
        delta = self.mlp_h(x_i)
        e = paddle.concat([pos_j - pos_i + delta, x_j], axis=-1)
        return self.mlp_f(e)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  mlp_h={self.mlp_h},\n'
                f'  mlp_f={self.mlp_f},\n'
                f'  mlp_g={self.mlp_g},\n'
                f')')
