import numpy as np
import paddle
import utils.paddle_aux  # NOQA
from einops import rearrange

import ppsci

ACTIVATION = {
    "gelu": paddle.nn.GELU,
    "tanh": paddle.nn.Tanh,
    "sigmoid": paddle.nn.Sigmoid,
    "relu": paddle.nn.ReLU,
    "leaky_relu": paddle.nn.LeakyReLU(negative_slope=0.1),
    "softplus": paddle.nn.Softplus,
    "ELU": paddle.nn.ELU,
    "silu": paddle.nn.Silu,
}


class Physics_Attention_1D(paddle.nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        out_0 = paddle.create_parameter(
            shape=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).shape,
            dtype=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones(shape=[1, heads, 1, 1]) * 0.5
            ),
        )
        out_0.stop_gradient = not True
        self.temperature = out_0
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_slice = paddle.nn.Linear(
            in_features=dim_head, out_features=slice_num
        )
        for ly in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(ly.weight)
        self.to_q = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_k = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_v = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            paddle.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        B, N, C = tuple(x.shape)
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(perm=[0, 2, 1, 3])
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(perm=[0, 2, 1, 3])
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].repeat(
            1, 1, 1, self.dim_head
        )
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        x = k_slice_token
        perm_0 = list(range(x.ndim))
        perm_0[-1] = -2
        perm_0[-2] = -1
        dots = paddle.matmul(x=q_slice_token, y=x.transpose(perm=perm_0)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class MLP(paddle.nn.Layer):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
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
        self.linear_pre = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_input, out_features=n_hidden), act()
        )
        self.linear_post = paddle.nn.Linear(in_features=n_hidden, out_features=n_output)
        self.linears = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Sequential(
                    paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden), act()
                )
                for _ in range(n_layers)
            ]
        )

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

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.Attn = Physics_Attention_1D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = paddle.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            fx = self.mlp2(self.ln_3(fx))
        return fx


class Model(paddle.nn.Layer):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
    ):
        super(Model, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref * self.ref,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=_ == n_layers - 1,
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        param = 1 / n_hidden * paddle.rand(shape=(n_hidden,), dtype="float32")
        out_1 = paddle.create_parameter(
            shape=param.shape,
            dtype=param.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(param),
        )
        out_1.stop_gradient = not True
        self.placeholder = out_1

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            m.weight = ppsci.utils.initializer.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def get_grid(self, my_pos):
        batchsize = tuple(my_pos.shape)[0]
        gridx = paddle.to_tensor(data=np.linspace(-1.5, 1.5, self.ref), dtype="float32")
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
            [batchsize, 1, self.ref, self.ref, 1]
        )
        gridy = paddle.to_tensor(data=np.linspace(0, 2, self.ref), dtype="float32")
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
            [batchsize, self.ref, 1, self.ref, 1]
        )
        gridz = paddle.to_tensor(data=np.linspace(-4, 4, self.ref), dtype="float32")
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
            [batchsize, self.ref, self.ref, 1, 1]
        )
        grid_ref = (
            paddle.concat(x=(gridx, gridy, gridz), axis=-1)
            .cuda(blocking=True)
            .reshape(batchsize, self.ref**3, 3)
        )
        pos = paddle.sqrt(
            x=paddle.sum(
                x=(my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, axis=-1
            )
        ).reshape(batchsize, tuple(my_pos.shape)[1], self.ref * self.ref * self.ref)
        return pos

    def forward(self, data):
        cfd_data, geom_data = data
        x, fx, _ = cfd_data.x, None, None
        x = x[None, :, :]
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = paddle.concat(x=(x, new_pos), axis=-1)
        if fx is not None:
            fx = paddle.concat(x=(x, fx), axis=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx[0]
