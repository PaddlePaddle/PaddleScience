# import numpy as np
import paddle
import paddle.nn as nn


def set_activation(activation):
    if activation == "identity":
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        print("WARNING: invalid activation function!")
        return -1


class FeedForward(nn.Layer):
    """An n-layer-feed-forward-layer module"""

    def __init__(self, in_dim=2, out_dim=1, depth=5, hidden_dim=50, activation="tanh"):
        super().__init__()
        self.depth = depth
        self.activation = set_activation(activation)
        self.ff_in = nn.Linear(in_dim, hidden_dim)
        self.linears = nn.LayerList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(self.depth - 2)]
        )
        self.ff_out = nn.Linear(hidden_dim, out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier Normal Initialization"""
        if isinstance(m, nn.Linear):
            init_xaiverNormal = paddle.nn.initializer.XavierNormal(gain=1.0)
            init_xaiverNormal(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant()
                init_constant(m.bias)

    def forward(self, x):
        x = self.ff_in(x)
        x = self.activation(x)
        for i in range(self.depth - 2):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.ff_out(x)
        return x


def ffn_pinns(params):
    return FeedForward(
        in_dim=params.in_dim,
        out_dim=params.out_dim,
        depth=params.depth,
        hidden_dim=params.hidden_dim,
        activation="tanh",
    )
