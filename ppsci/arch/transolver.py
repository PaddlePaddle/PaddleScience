"""
Reference: https://github.com/thuml/Transolver
"""
from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple

import einops
import paddle
from paddle import nn

from ppsci.arch import activation
from ppsci.arch import base


class Physics_Attention_Irregular_Mesh(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = nn.Parameter(paddle.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_fx = nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_slice = nn.Linear(in_features=dim_head, out_features=slice_num)
        for l in [self.in_project_slice]:
            nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_k = nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_v = nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_out = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        B, N, C = x.shape
        fx_mid = (
            self.in_project_fx(x)
            .reshape([B, N, self.heads, self.dim_head])
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B, N, H, D] -> [B, H, N, D]
        x_mid = (
            self.in_project_x(x)
            .reshape([B, N, self.heads, self.dim_head])
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B, H, N, D]
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # [B, H, N, G]
        slice_norm = slice_weights.sum(2)  # [B, H, G]

        slice_token = paddle.matmul(slice_weights, fx_mid, transpose_x=True)
        slice_token = (
            slice_token / (slice_norm + 1e-05)[:, :, :, None]
        )  # [B, H, G, D] / ([B, H, G, 1] -> [B, H, G, D])

        q_slice_token = self.to_q(slice_token)  # [B, H, G, D']
        k_slice_token = self.to_k(slice_token)  # [B, H, G, D']
        v_slice_token = self.to_v(slice_token)  # [B, H, G, D']

        dots = (
            paddle.matmul(q_slice_token, k_slice_token, transpose_y=True)
            * self.scale  # [B, H, G, D'] x [B, H, D', G] -> [B, H, G, G]
        )
        attn = self.softmax(dots)  # [B, H, G, G]
        attn = self.dropout(attn)  # [B, H, G, G]
        out_slice_token = paddle.matmul(
            attn, v_slice_token
        )  # [B, H, G, G] x [B, H, G, D'] -> [B, H, G, D']

        # out_slice_token = F.scaled_dot_product_attention(
        #     q_slice_token,          # [B, H, G, D']
        #     k_slice_token,          # [B, H, G, D']
        #     v_slice_token,          # [B, H, G, D']
        #     dropout_p=self.dropout,
        #     is_causal=False,
        # )
        # out_x = paddle.einsum(
        #     "bhgc,bhng->bhnc", out_slice_token, slice_weights
        # )  # [B, H, N, G] x [B, H, G, D'] -> [B, H, N, D']
        out_x = paddle.matmul(slice_weights, out_slice_token)
        out_x = einops.rearrange(out_x, "b h n d -> b n (h d)")  # [B, N, HD']
        return self.to_out(out_x)  # [B, N, C]


class MLP(nn.Layer):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=n_hidden),
            activation.get_activation(act),
        )
        self.linear_post = nn.Linear(in_features=n_hidden, out_features=n_output)
        self.linears = nn.LayerList(
            sublayers=[
                nn.Sequential(
                    nn.Linear(in_features=n_hidden, out_features=n_hidden),
                    activation.get_activation(act),
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


class TransolverBlock(nn.Layer):
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
        self.ln_1 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        space_dim: int = 1,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: int = 0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 1,
        fun_dim: int = 1,
        out_dim: Optional[int, List[int]] = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        if isinstance(out_dim, int):
            out_dim = [out_dim]
        out_dim = list(out_dim)
        self.out_dim = out_dim
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
        self.blocks = nn.LayerList(
            sublayers=[
                TransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=sum(out_dim),
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            1 / n_hidden * paddle.rand(shape=[n_hidden], dtype=paddle.float32)
        )
        x = paddle.linspace(-1.5, 1.5, self.ref)
        y = paddle.linspace(0, 2, self.ref)
        z = paddle.linspace(-4, 4, self.ref)
        gridx, gridy, gridz = paddle.meshgrid(x, y, z, indexing="ij")
        grid_ref = paddle.stack([gridx, gridy, gridz], axis=-1).reshape(
            [1, self.ref**3, 3]
        )
        self.register_buffer("grid_ref", grid_ref)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1D)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def get_grid(self, my_pos):
        pos = paddle.sqrt(
            # [B, X, 1, 3] - [1, 1, R**3, 3] --> [B, X, R**3, 3]
            paddle.sum(
                (my_pos[:, :, None, :] - self.grid_ref[:, None, :, :]) ** 2, axis=-1
            )
        )

        return pos  # [B, X, R³]

    def forward(self, input_dict):
        # [B, N, C]
        x = input_dict[self.input_keys[0]]

        if self.unified_pos:
            # [B, N, C]
            pos = input_dict[self.input_keys[1]]
            # [B, N, R³]
            new_pos = self.get_grid(pos)
            # [B, N, C+R³]
            x = paddle.concat((x, new_pos), axis=-1)

        # [B, N, C]
        y = self.preprocess(x)
        y = y + self.placeholder[None, None, :]

        for block in self.blocks:
            y: paddle.Tensor = block(y)

        # [B, N, C]
        outputs = y.split(self.out_dim, axis=-1)
        return {k: v for k, v in zip(self.output_keys, outputs)}
