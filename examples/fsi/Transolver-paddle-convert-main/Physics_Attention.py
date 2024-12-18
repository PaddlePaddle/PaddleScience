import sys
# sys.path.append('../../utils')
from utils import paddle_aux
import paddle
from einops import rearrange, repeat


class Physics_Attention_Irregular_Mesh(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=
            inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features
            =inner_dim)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head
            ).transpose(perm=[0, 2, 1, 3]).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head
            ).transpose(perm=[0, 2, 1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.
            temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
        H=101, W=31, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.in_project_x = paddle.nn.Conv2D(in_channels=dim, out_channels=
            inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_fx = paddle.nn.Conv2D(in_channels=dim, out_channels
            =inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        x = x.reshape(B, self.H, self.W, C).contiguous().transpose(perm=[0,
            3, 1, 2]).contiguous()
        fx_mid = self.in_project_fx(x).transpose(perm=[0, 2, 3, 1]).contiguous(
            ).reshape(B, N, self.heads, self.dim_head).transpose(perm=[0, 2,
            1, 3]).contiguous()
        x_mid = self.in_project_x(x).transpose(perm=[0, 2, 3, 1]).contiguous(
            ).reshape(B, N, self.heads, self.dim_head).transpose(perm=[0, 2,
            1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / paddle.
            clip(x=self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=32,
        H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D
        self.in_project_x = paddle.nn.Conv3D(in_channels=dim, out_channels=
            inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_fx = paddle.nn.Conv3D(in_channels=dim, out_channels
            =inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().transpose(perm
            =[0, 4, 1, 2, 3]).contiguous()
        fx_mid = self.in_project_fx(x).transpose(perm=[0, 2, 3, 4, 1]
            ).contiguous().reshape(B, N, self.heads, self.dim_head).transpose(
            perm=[0, 2, 1, 3]).contiguous()
        x_mid = self.in_project_x(x).transpose(perm=[0, 2, 3, 4, 1]
            ).contiguous().reshape(B, N, self.heads, self.dim_head).transpose(
            perm=[0, 2, 1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / paddle.
            clip(x=self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
