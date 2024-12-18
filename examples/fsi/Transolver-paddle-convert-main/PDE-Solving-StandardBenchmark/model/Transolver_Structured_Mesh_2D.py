import sys

from utils import paddle_aux
import paddle
from paddle.nn.initializer import TruncatedNormal, Constant
import numpy as np
from model.Embedding import timestep_embedding
from model.Physics_Attention import Physics_Attention_Structured_Mesh_2D

ACTIVATION = {'gelu': paddle.nn.GELU, 'tanh': paddle.nn.Tanh, 'sigmoid':
    paddle.nn.Sigmoid, 'relu': paddle.nn.ReLU, 'leaky_relu': paddle.nn.
    LeakyReLU(negative_slope=0.1), 'softplus': paddle.nn.Softplus, 'ELU':
                  paddle.nn.ELU, 'silu': paddle.nn.Silu}


class MLP(paddle.nn.Layer):

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu',
                 res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = paddle.nn.Sequential(paddle.nn.Linear(in_features
                                                                =n_input, out_features=n_hidden), act())
        self.linear_post = paddle.nn.Linear(in_features=n_hidden,
                                            out_features=n_output)
        self.linears = paddle.nn.LayerList(sublayers=[paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(paddle.nn.Layer):
    """Transformer encoder block."""

    def __init__(self, num_heads: int, hidden_dim: int, dropout: float, act
    ='gelu', mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32, H=
                 85, W=85):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=
        num_heads, dim_head=hidden_dim // num_heads, dropout=dropout,
                                                         slice_num=slice_num, H=H, W=W)
        self.ln_2 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = paddle.nn.Linear(in_features=hidden_dim,
                                         out_features=out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(paddle.nn.Layer):

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, Time_Input=False, act='gelu', mlp_ratio=1, fun_dim=1,
                 out_dim=1, slice_num=32, ref=8, unified_pos=False, H=85, W=85):
        super(Model, self).__init__()
        self.__name__ = 'Transolver_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden),
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden)
            )
        self.blocks = paddle.nn.LayerList([
            Transolver_block(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
                mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num, H=H,
                W=W, last_layer=_ == n_layers - 1
            ) for _ in range(n_layers)
        ])
        self.initialize_weights()
        self.placeholder = paddle.create_parameter(
            shape=[n_hidden], dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(1 / n_hidden * paddle.rand([n_hidden]))
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal = TruncatedNormal(mean=0.0, std=0.02)
            trunc_normal(m.weight)
            if m.bias is not None:
                constant = Constant(value=0.0)
                constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            constant = Constant(value=0.0)
            constant(m.bias)
            constant = Constant(value=1.0)
            constant(m.weight)

    def get_grid(self):
        # 获取网格位置信息
        h = paddle.arange(0, self.H, dtype='float32')
        w = paddle.arange(0, self.W, dtype='float32')
        grid = paddle.meshgrid(h, w)
        grid = paddle.stack(grid, axis=-1)
        grid = grid.reshape([1, -1, 2])
        return grid

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = paddle.to_tensor(data=np.linspace(0, 1, size_x), dtype=
        'float32')
        gridx = gridx.reshape(1, size_x, 1, 1).tile(repeat_times=[batchsize,
                                                                  1, size_y, 1])
        gridy = paddle.to_tensor(data=np.linspace(0, 1, size_y), dtype=
        'float32')
        gridy = gridy.reshape(1, 1, size_y, 1).tile(repeat_times=[batchsize,
                                                                  size_x, 1, 1])
        grid = paddle.concat(x=(gridx, gridy), axis=-1).cuda(blocking=True)
        gridx = paddle.to_tensor(data=np.linspace(0, 1, self.ref), dtype=
        'float32')
        gridx = gridx.reshape(1, self.ref, 1, 1).tile(repeat_times=[
            batchsize, 1, self.ref, 1])
        gridy = paddle.to_tensor(data=np.linspace(0, 1, self.ref), dtype=
        'float32')
        gridy = gridy.reshape(1, 1, self.ref, 1).tile(repeat_times=[
            batchsize, self.ref, 1, 1])
        grid_ref = paddle.concat(x=(gridx, gridy), axis=-1).cuda(blocking=True)
        pos = paddle.sqrt(x=paddle.sum(x=(grid[:, :, :, None, None, :] -
                                          grid_ref[:, None, None, :, :, :]) ** 2, axis=-1)).reshape(batchsize
                                                                                                    , size_x, size_y,
                                                                                                    self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = self.pos.tile(repeat_times=[tuple(x.shape)[0], 1, 1, 1]
                              ).reshape(tuple(x.shape)[0], self.H * self.W, self.ref *
                                        self.ref)
        if fx is not None:
            fx = paddle.concat(x=(x, fx), axis=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]
        if T is not None:
            Time_emb = paddle.tile(timestep_embedding(T, self.n_hidden), repeat_times=[1, x.shape[1], 1])
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
        for block in self.blocks:
            fx = block(fx)
        return fx
