from __future__ import annotations

from typing import Optional
from typing import Tuple

import paddle

import ppsci

# from ppsci.arch import activation as act_mod
from ppsci.arch import base

# from ppsci.arch.kan import KAN_Fourier
# from ppsci.arch.kan import KAN_Laplace
# from ppsci.arch.kan import KAN_Legendre
from ppsci.arch.kan import PiraKanNet

"""
if f of x and g of u are periodic functions, set periods=True in PIRateNet can get better convergence speed and accuracy.
"""


def sinusoidal_positional_encoding(pos: paddle.Tensor, embed_dim: int) -> paddle.Tensor:
    """
    Args:
        pos: 输入位置张量，形状为 [*, 1]
        embed_dim: 嵌入维度（最终输出维度的一半）
    Returns:
        位置编码张量，形状为 [*, 2 * embed_dim]
    """
    # 生成 i 从 0 到 embed_dim-1
    i = paddle.arange(embed_dim, dtype=paddle.get_default_dtype())  # [embed_dim]
    # print(i.shape)

    # 计算频率因子 (10^(i/embed_dim))
    div_term = paddle.pow(paddle.full([], 10.0), i / embed_dim)  # [embed_dim]

    # 计算角度：pos / div_term
    div_term_shape = [1] * (pos.ndim - 1) + [-1]
    angles = pos / div_term.reshape(div_term_shape)  # [*, embed_dim]

    # 分别计算正弦和余弦
    sin_enc = paddle.sin(angles)  # [*, embed_dim]
    cos_enc = paddle.cos(angles)  # [*, embed_dim]

    # 拼接结果
    encoding = paddle.concat([sin_enc, cos_enc], axis=-1)  # [*, 2*embed_dim]
    return encoding


class DeepONet(base.Arch):
    def __init__(
        self,
        trunk_input_keys: Tuple[str, ...] = ("x", "u"),
        branch_a_input_keys: Tuple[str, ...] = ("a",),
        output_keys: Tuple[str, ...] = ("f_r", "f_i", "g_r", "g_i"),
        trunk_num_layers: int = 3,
        trunk_hidden_size: int = 256,
        hidden_out_size: int = 64,
        branch_input_dim: int = 128,
        branch_num_layers: int = 3,
        branch_hidden_size: int = 256,
        trunk_activation: str = "tanh",
        branch_activation: str = "tanh",
        branch_n_dim: int | None = None,
        branch_m_dim: int | None = None,
        # use_bias: bool = True,
    ):
        super().__init__()
        self.trunk_input_keys = trunk_input_keys
        self.branch_a_input_keys = branch_a_input_keys
        self.input_keys = self.trunk_input_keys + self.branch_a_input_keys
        self.output_keys = output_keys
        self.hidden_out_size = hidden_out_size

        self.x_trunk_net = ppsci.arch.MLP(
            input_keys=self.trunk_input_keys[0:1],
            output_keys=("tx",),
            num_layers=trunk_num_layers,
            hidden_size=trunk_hidden_size,
            activation=trunk_activation,
            input_dim=1,
            output_dim=hidden_out_size,
        )

        self.u_trunk_net = ppsci.arch.MLP(
            input_keys=self.trunk_input_keys[1:2],
            output_keys=("tu",),
            num_layers=trunk_num_layers,
            hidden_size=trunk_hidden_size,
            activation=trunk_activation,
            input_dim=1,
            output_dim=hidden_out_size,
        )

        self.a1_branch_net = ppsci.arch.MLP(
            input_keys=self.branch_a_input_keys,
            output_keys=("bax",),
            num_layers=branch_num_layers,
            hidden_size=branch_hidden_size,
            activation=branch_activation,
            input_dim=branch_input_dim,
            output_dim=hidden_out_size,
        )

        self.a2_branch_net = ppsci.arch.MLP(
            input_keys=self.branch_a_input_keys,
            output_keys=("bau",),
            num_layers=branch_num_layers,
            hidden_size=branch_hidden_size,
            activation=branch_activation,
            input_dim=branch_input_dim,
            output_dim=hidden_out_size,
        )

        self.branch_n_dim = branch_n_dim
        if self.branch_n_dim:
            self.n1_branch_net = ppsci.arch.MLP(
                input_keys=("n_pos_embed",),
                output_keys=("n_embed",),
                num_layers=branch_num_layers,
                hidden_size=branch_hidden_size,
                activation=branch_activation,
                input_dim=branch_n_dim,
                output_dim=hidden_out_size,
            )
            self.n2_branch_net = ppsci.arch.MLP(
                input_keys=("n_pos_embed",),
                output_keys=("n_embed",),
                num_layers=branch_num_layers,
                hidden_size=branch_hidden_size,
                activation=branch_activation,
                input_dim=branch_n_dim,
                output_dim=hidden_out_size,
            )

        self.branch_m_dim = branch_m_dim
        if self.branch_m_dim:
            self.m1_branch_net = ppsci.arch.MLP(
                input_keys=("m",),
                output_keys=("bmx",),
                num_layers=branch_num_layers,
                hidden_size=branch_hidden_size,
                activation=branch_activation,
                input_dim=branch_input_dim,
                output_dim=hidden_out_size,
            )
            self.m2_branch_net = ppsci.arch.MLP(
                input_keys=("m",),
                output_keys=("bmu",),
                num_layers=branch_num_layers,
                hidden_size=branch_hidden_size,
                activation=branch_activation,
                input_dim=branch_input_dim,
                output_dim=hidden_out_size,
            )

        # self.use_bias = use_bias
        # if use_bias:
        #     # register bias to parameter for updating in optimizer and storage
        #     self.b = self.create_parameter(
        #         shape=(1,),
        #         attr=nn.initializer.Constant(0.0),
        #     )

    def forward(self, x):
        # Trunk nets hidden output for f(x) and g(u)

        x_features = self.x_trunk_net(x)[self.x_trunk_net.output_keys[0]]
        u_features = self.u_trunk_net(x)[self.u_trunk_net.output_keys[0]]
        # print(f"hidden out x {x_features.shape}")
        # Branch net hidden output for parameters: a, etc.
        a1_features = self.a1_branch_net(x)[self.a1_branch_net.output_keys[0]]
        a2_features = self.a2_branch_net(x)[self.a2_branch_net.output_keys[0]]
        if self.branch_n_dim:
            n_pos_embed = sinusoidal_positional_encoding(
                x["n"],
                self.branch_n_dim // 2,
            )  # [*, 2*embed_dim]
            n1_features = self.n1_branch_net({"n_pos_embed": n_pos_embed})[
                self.n1_branch_net.output_keys[0]
            ]  # [*, 2*embed_dim] --> [*, hidden_out_size]
            n2_features = self.n2_branch_net({"n_pos_embed": n_pos_embed})[
                self.n2_branch_net.output_keys[0]
            ]  # [*, 2*embed_dim] --> [*, hidden_out_size]

        if self.branch_m_dim:
            # m_embed = sinusoidal_positional_encoding(
            #     x["m"],
            #     self.branch_m_dim // 2,
            # )  # [*, 2*embed_dim]
            m1_features = self.m1_branch_net(x)[self.m1_branch_net.output_keys[0]]
            m2_features = self.m2_branch_net(x)[self.m2_branch_net.output_keys[0]]

        split_ind = int(self.hidden_out_size / 2)
        f_r_tmp = x_features[:, :split_ind] * a1_features[:, :split_ind]
        if self.branch_n_dim:
            f_r_tmp *= n1_features[:, :split_ind]
        if self.branch_m_dim:
            f_r_tmp *= m1_features[:, :split_ind]
        f_r = paddle.sum(
            f_r_tmp,
            axis=-1,
            keepdim=True,
        )
        # print(f"f r output shape {f_r.shape}")
        f_i_tmp = x_features[:, split_ind:] * a1_features[:, split_ind:]
        if self.branch_n_dim:
            f_i_tmp *= n1_features[:, split_ind:]
        if self.branch_m_dim:
            f_i_tmp *= m1_features[:, split_ind:]
        f_i = paddle.sum(
            f_i_tmp,
            axis=-1,
            keepdim=True,
        )
        # print(f"f i outpu shape {f_i.shape}")
        g_r_tmp = u_features[:, :split_ind] * a2_features[:, :split_ind]
        if self.branch_n_dim:
            g_r_tmp *= n2_features[:, :split_ind]
        if self.branch_m_dim:
            g_r_tmp *= m2_features[:, :split_ind]
        g_r = paddle.sum(
            g_r_tmp,
            axis=-1,
            keepdim=True,
        )
        g_i_tmp = u_features[:, split_ind:] * a2_features[:, split_ind:]
        if self.branch_n_dim:
            g_i_tmp *= n2_features[:, split_ind:]
        if self.branch_m_dim:
            g_i_tmp *= m2_features[:, split_ind:]
        g_i = paddle.sum(
            g_i_tmp,
            axis=-1,
            keepdim=True,
        )

        # Hard constraint of BC at x = 1 and u = 1
        x_in = x[self.trunk_input_keys[0]]
        c = paddle.exp(x_in - 1) - 1
        f_new_real = c * f_r + 1
        f_new_imag = c * f_i
        # print(f"f new real output {f_new_real.shape}")
        # f_new_real = paddle.reshape(f_new_real, [-1, 1])
        # print(f"reshape f real output {f_new_real.shape}")
        # f_new_imag = paddle.reshape(f_new_imag, [-1, 1])

        u_in = x[self.trunk_input_keys[1]]
        c_g = paddle.exp(u_in + 1) - 1
        g_new_real = c_g * g_r + 1
        g_new_imag = c_g * g_i
        # g_new_real = paddle.reshape(g_new_real, [-1, 1])
        # g_new_imag = paddle.reshape(g_new_imag, [-1, 1])

        return {
            self.output_keys[0]: f_new_real,
            self.output_keys[1]: f_new_imag,
            self.output_keys[2]: g_new_real,
            self.output_keys[3]: g_new_imag,
        }


class DeepOPirate(base.Arch):
    def __init__(
        self,
        trunk_input_keys: Tuple[str, ...] = ("x", "u"),
        branch_a_input_keys: Tuple[str, ...] = ("a",),
        output_keys: Tuple[str, ...] = ("f_r", "f_i", "g_r", "g_i"),
        trunk_num_blocks: int = 3,
        trunk_hidden_size: int = 256,
        hidden_out_size: int = 64,
        branch_input_dim: int = 128,
        branch_num_blocks: int = 3,
        branch_hidden_size: int = 256,
        trunk_activation: str = "tanh",
        branch_activation: str = "tanh",
        trunk_fourier: Optional[dict] = None,
        branch_fourier: Optional[dict] = None,
        random_weight: Optional[dict] = None,
        branch_n_dim: int | None = None,
        # use_bias: bool = True,
    ):
        super().__init__()
        self.trunk_input_keys = trunk_input_keys
        self.branch_a_input_keys = branch_a_input_keys
        self.input_keys = self.trunk_input_keys + self.branch_a_input_keys
        self.output_keys = output_keys
        self.hidden_out_size = hidden_out_size

        self.x_trunk_net = ppsci.arch.PirateNet(
            input_keys=self.trunk_input_keys[0:1],
            output_keys=("tx",),
            num_blocks=trunk_num_blocks,
            hidden_size=trunk_hidden_size,
            activation=trunk_activation,
            fourier=trunk_fourier,
            random_weight=random_weight,
            input_dim=1,
            output_dim=hidden_out_size,
        )

        self.u_trunk_net = ppsci.arch.PirateNet(
            input_keys=self.trunk_input_keys[1:2],
            output_keys=("tu",),
            num_blocks=trunk_num_blocks,
            hidden_size=trunk_hidden_size,
            activation=trunk_activation,
            fourier=trunk_fourier,
            random_weight=random_weight,
            input_dim=1,
            output_dim=hidden_out_size,
        )

        self.a1_branch_net = ppsci.arch.PirateNet(
            input_keys=self.branch_a_input_keys,
            output_keys=("bax",),
            num_blocks=branch_num_blocks,
            hidden_size=branch_hidden_size,
            activation=branch_activation,
            fourier=branch_fourier,
            random_weight=random_weight,
            input_dim=branch_input_dim,
            output_dim=hidden_out_size,
        )

        self.a2_branch_net = ppsci.arch.PirateNet(
            input_keys=self.branch_a_input_keys,
            output_keys=("bau",),
            num_blocks=branch_num_blocks,
            hidden_size=branch_hidden_size,
            activation=branch_activation,
            fourier=branch_fourier,
            random_weight=random_weight,
            input_dim=branch_input_dim,
            output_dim=hidden_out_size,
        )
        self.branch_n_dim = branch_n_dim
        if self.branch_n_dim:
            self.n1_branch_net = ppsci.arch.PirateNet(
                input_keys=("n_pos_embed",),
                output_keys=("n_embed",),
                num_blocks=branch_num_blocks,
                hidden_size=branch_hidden_size,
                activation=branch_activation,
                fourier=branch_fourier,
                random_weight=random_weight,
                input_dim=self.branch_n_dim,
                output_dim=hidden_out_size,
            )
            self.n2_branch_net = ppsci.arch.PirateNet(
                input_keys=("n_pos_embed",),
                output_keys=("n_embed",),
                num_blocks=branch_num_blocks,
                hidden_size=branch_hidden_size,
                activation=branch_activation,
                fourier=branch_fourier,
                random_weight=random_weight,
                input_dim=self.branch_n_dim,
                output_dim=hidden_out_size,
            )

        # self.use_bias = use_bias
        # if use_bias:
        #     # register bias to parameter for updating in optimizer and storage
        #     self.b = self.create_parameter(
        #         shape=(1,),
        #         attr=nn.initializer.Constant(0.0),
        #     )

    def forward(self, x):
        # Trunk nets hidden output for f(x) and g(u)

        x_features = self.x_trunk_net(x)[self.x_trunk_net.output_keys[0]]
        u_features = self.u_trunk_net(x)[self.u_trunk_net.output_keys[0]]
        # print(f"hidden out x {x_features.shape}")
        # Branch net hidden output for parameters: a, etc.
        a1_features = self.a1_branch_net(x)[self.a1_branch_net.output_keys[0]]
        a2_features = self.a2_branch_net(x)[self.a2_branch_net.output_keys[0]]
        # print(f"hidden out a {a1_features.shape}")

        if self.branch_n_dim:
            n_pos_embed = sinusoidal_positional_encoding(
                x["n"],
                self.branch_n_dim // 2,
            )
            n1_features = self.n1_branch_net({"n_pos_embed": n_pos_embed})[
                self.n1_branch_net.output_keys[0]
            ]  # [*, 2*embed_dim] --> [*, hidden_out_size]
            n2_features = self.n2_branch_net({"n_pos_embed": n_pos_embed})[
                self.n2_branch_net.output_keys[0]
            ]  # [*, 2*embed_dim] --> [*, hidden_out_size]

        # Dot product for output f and g, the former part of hidden output is used for real part of f and g
        # the latter part is used for imagine part of f and g
        split_ind = int(self.hidden_out_size / 2)
        f_r_tmp = x_features[:, :split_ind] * a1_features[:, :split_ind]
        if self.branch_n_dim:
            f_r_tmp *= n1_features[:, :split_ind]
        f_r = paddle.sum(
            f_r_tmp,
            axis=-1,
            keepdim=True,
        )
        # print(f"f r output shape {f_r.shape}")
        f_i_tmp = x_features[:, split_ind:] * a1_features[:, split_ind:]
        if self.branch_n_dim:
            f_i_tmp *= n1_features[:, split_ind:]
        f_i = paddle.sum(
            f_i_tmp,
            axis=-1,
            keepdim=True,
        )
        # print(f"f i outpu shape {f_i.shape}")
        g_r_tmp = u_features[:, :split_ind] * a2_features[:, :split_ind]
        if self.branch_n_dim:
            g_r_tmp *= n2_features[:, :split_ind]
        g_r = paddle.sum(
            g_r_tmp,
            axis=-1,
            keepdim=True,
        )
        g_i_tmp = u_features[:, split_ind:] * a2_features[:, split_ind:]
        if self.branch_n_dim:
            g_i_tmp *= n2_features[:, split_ind:]
        g_i = paddle.sum(
            g_i_tmp,
            axis=-1,
            keepdim=True,
        )

        # Hard constraint of BC at x = 1 and u = 1
        x_in = x[self.trunk_input_keys[0]]
        c = paddle.exp(x_in - 1) - 1
        f_new_real = c * f_r + 1
        f_new_imag = c * f_i
        # print(f"f new real output {f_new_real.shape}")
        # f_new_real = paddle.reshape(f_new_real, [-1, 1])
        # print(f"reshape f real output {f_new_real.shape}")
        # f_new_imag = paddle.reshape(f_new_imag, [-1, 1])

        u_in = x[self.trunk_input_keys[1]]
        c_g = paddle.exp(u_in + 1) - 1
        g_new_real = c_g * g_r + 1
        g_new_imag = c_g * g_i
        # g_new_real = paddle.reshape(g_new_real, [-1, 1])
        # g_new_imag = paddle.reshape(g_new_imag, [-1, 1])

        return {
            self.output_keys[0]: f_new_real,
            self.output_keys[1]: f_new_imag,
            self.output_keys[2]: g_new_real,
            self.output_keys[3]: g_new_imag,
        }


class DeepOPirakan(base.Arch):
    def __init__(
        self,
        trunk_input_keys: Tuple[str, ...] = ("x", "u"),
        branch_a_input_keys: Tuple[str, ...] = ("a",),
        output_keys: Tuple[str, ...] = ("f_r", "f_i", "g_r", "g_i"),
        trunk_num_blocks: int = 3,
        trunk_hidden_size: int = 32,
        trunk_kan_grid_size: int = 1,
        hidden_out_size: int = 32,
        branch_input_dim: int = 1,
        branch_num_blocks: int = 2,
        branch_hidden_size: int = 32,
        branch_kan_grid_size: int = 1,
        trunk_activation: str = "tanh",
        branch_activation: str = "tanh",
        trunk_fourier: Optional[dict] = None,
        branch_fourier: Optional[dict] = None,
        random_weight: Optional[dict] = None,
        alpha_init: float = 0.0,
        branch_n_dim: int | None = None,
        branch_m_dim: int | None = None,
        # use_bias: bool = True,
    ):
        super().__init__()
        self.trunk_input_keys = trunk_input_keys
        self.branch_a_input_keys = branch_a_input_keys
        self.input_keys = self.trunk_input_keys + self.branch_a_input_keys
        self.output_keys = output_keys
        self.hidden_out_size = hidden_out_size

        self.x_trunk_net = PiraKanNet(
            input_keys=self.trunk_input_keys[0:1],
            output_keys=("tx",),
            num_blocks=trunk_num_blocks,
            hidden_size=trunk_hidden_size,
            kan_grid_size=trunk_kan_grid_size,
            activation=trunk_activation,
            fourier=trunk_fourier,
            random_weight=random_weight,
            alpha_init=alpha_init,
            input_dim=1,
            output_dim=hidden_out_size,
        )

        self.u_trunk_net = PiraKanNet(
            input_keys=self.trunk_input_keys[1:2],
            output_keys=("tu",),
            num_blocks=trunk_num_blocks,
            hidden_size=trunk_hidden_size,
            kan_grid_size=trunk_kan_grid_size,
            activation=trunk_activation,
            fourier=trunk_fourier,
            random_weight=random_weight,
            alpha_init=alpha_init,
            input_dim=1,
            output_dim=hidden_out_size,
        )

        self.a1_branch_net = PiraKanNet(
            input_keys=self.branch_a_input_keys,
            output_keys=("bax",),
            num_blocks=branch_num_blocks,
            hidden_size=branch_hidden_size,
            kan_grid_size=branch_kan_grid_size,
            activation=branch_activation,
            fourier=branch_fourier,
            random_weight=random_weight,
            alpha_init=alpha_init,
            input_dim=branch_input_dim,
            output_dim=hidden_out_size,
        )

        self.a2_branch_net = PiraKanNet(
            input_keys=self.branch_a_input_keys,
            output_keys=("bau",),
            num_blocks=branch_num_blocks,
            hidden_size=branch_hidden_size,
            kan_grid_size=branch_kan_grid_size,
            activation=branch_activation,
            fourier=branch_fourier,
            random_weight=random_weight,
            alpha_init=alpha_init,
            input_dim=branch_input_dim,
            output_dim=hidden_out_size,
        )
        self.branch_n_dim = branch_n_dim
        if self.branch_n_dim:
            # self.n_branch_net = ppsci.arch.MLP(
            #     input_keys=("n_pos_embed",),
            #     output_keys=("n_embed",),
            #     num_layers=2,
            #     hidden_size=branch_hidden_size,
            #     activation=branch_activation,
            #     input_dim=self.branch_n_dim,
            #     output_dim=hidden_out_size,
            # )
            self.n1_branch_net = PiraKanNet(
                input_keys=("n_pos_embed",),
                output_keys=("n_embed",),
                num_blocks=1,
                hidden_size=branch_hidden_size,
                kan_grid_size=branch_kan_grid_size,
                activation=branch_activation,
                fourier=branch_fourier,
                random_weight=random_weight,
                alpha_init=alpha_init,
                input_dim=self.branch_n_dim,
                output_dim=hidden_out_size,
            )
            self.n2_branch_net = PiraKanNet(
                input_keys=("n_pos_embed",),
                output_keys=("n_embed",),
                num_blocks=1,
                hidden_size=branch_hidden_size,
                kan_grid_size=branch_kan_grid_size,
                activation=branch_activation,
                fourier=branch_fourier,
                random_weight=random_weight,
                alpha_init=alpha_init,
                input_dim=self.branch_n_dim,
                output_dim=hidden_out_size,
            )

        self.branch_m_dim = branch_m_dim
        if self.branch_m_dim:
            self.m_branch_net = PiraKanNet(
                input_keys=("m",),
                output_keys=("bm",),
                num_blocks=1,
                hidden_size=branch_hidden_size,
                kan_grid_size=branch_kan_grid_size,
                activation=branch_activation,
                fourier=branch_fourier,
                random_weight=random_weight,
                alpha_init=alpha_init,
                input_dim=branch_input_dim,
                output_dim=hidden_out_size,
            )
            # self.m2_branch_net = PiraKanNet(
            #     input_keys=("m",),
            #     output_keys=("bm",),
            #     num_blocks=1,
            #     hidden_size=branch_hidden_size,
            #     kan_grid_size=branch_kan_grid_size,
            #     activation=branch_activation,
            #     fourier=branch_fourier,
            #     random_weight=random_weight,
            #     alpha_init=alpha_init,
            #     input_dim=branch_input_dim,
            #     output_dim=hidden_out_size,
            # )

        # self.use_bias = use_bias
        # if use_bias:
        #     # register bias to parameter for updating in optimizer and storage
        #     self.b = self.create_parameter(
        #         shape=(1,),
        #         attr=nn.initializer.Constant(0.0),
        #     )

    def forward(self, x):
        # Trunk nets hidden output for f(x) and g(u)

        x_features = self.x_trunk_net(x)[self.x_trunk_net.output_keys[0]]
        u_features = self.u_trunk_net(x)[self.u_trunk_net.output_keys[0]]
        # print(f"hidden out x {x_features.shape}")
        # Branch net hidden output for parameters: a, etc.
        a1_features = self.a1_branch_net(x)[self.a1_branch_net.output_keys[0]]
        a2_features = self.a2_branch_net(x)[self.a2_branch_net.output_keys[0]]
        # print(f"hidden out a {a1_features.shape}")
        # Dot product for output f and g, the former part of hidden output is used for real part of f and g
        # the latter part is used for imagine part of f and g
        if self.branch_n_dim:
            n_pos_embed = sinusoidal_positional_encoding(
                x["n"],
                self.branch_n_dim // 2,
            )  # [*, 2*embed_dim]
            n1_features = self.n1_branch_net({"n_pos_embed": n_pos_embed})[
                self.n1_branch_net.output_keys[0]
            ]  # [*, 2*embed_dim] --> [*, hidden_out_size]
            n2_features = self.n2_branch_net({"n_pos_embed": n_pos_embed})[
                self.n2_branch_net.output_keys[0]
            ]  # [*, 2*embed_dim] --> [*, hidden_out_size]

        if self.branch_m_dim:
            # m_embed = sinusoidal_positional_encoding(
            #     x["m"],
            #     self.branch_m_dim // 2,
            # )  # [*, 2*embed_dim]
            m_features = self.m_branch_net(x)[self.m_branch_net.output_keys[0]]
            # m2_features = self.m2_branch_net(x)[self.m2_branch_net.output_keys[0]]

        split_ind = int(self.hidden_out_size / 2)
        f_r_tmp = x_features[:, :split_ind] * a1_features[:, :split_ind]
        if self.branch_n_dim:
            f_r_tmp *= n1_features[:, :split_ind]
        if self.branch_m_dim:
            f_r_tmp *= m_features[:, :split_ind]
        f_r = paddle.sum(
            f_r_tmp,
            axis=-1,
            keepdim=True,
        )
        # print(f"f r output shape {f_r.shape}")
        f_i_tmp = x_features[:, split_ind:] * a1_features[:, split_ind:]
        if self.branch_n_dim:
            f_i_tmp *= n1_features[:, split_ind:]
        if self.branch_m_dim:
            f_i_tmp *= m_features[:, split_ind:]
        f_i = paddle.sum(
            f_i_tmp,
            axis=-1,
            keepdim=True,
        )
        # print(f"f i outpu shape {f_i.shape}")
        g_r_tmp = u_features[:, :split_ind] * a2_features[:, :split_ind]
        if self.branch_n_dim:
            g_r_tmp *= n2_features[:, :split_ind]
        if self.branch_m_dim:
            g_r_tmp *= m_features[:, :split_ind]
        g_r = paddle.sum(
            g_r_tmp,
            axis=-1,
            keepdim=True,
        )
        g_i_tmp = u_features[:, split_ind:] * a2_features[:, split_ind:]
        if self.branch_n_dim:
            g_i_tmp *= n2_features[:, split_ind:]
        if self.branch_m_dim:
            g_i_tmp *= m_features[:, split_ind:]
        g_i = paddle.sum(
            g_i_tmp,
            axis=-1,
            keepdim=True,
        )

        # Hard constraint of BC at x = 1 and u = 1
        x_in = x[self.trunk_input_keys[0]]
        c = paddle.exp(x_in - 1) - 1
        f_new_real = c * f_r + 1
        f_new_imag = c * f_i
        # print(f"f new real output {f_new_real.shape}")
        # f_new_real = paddle.reshape(f_new_real, [-1, 1])
        # print(f"reshape f real output {f_new_real.shape}")
        # f_new_imag = paddle.reshape(f_new_imag, [-1, 1])

        u_in = x[self.trunk_input_keys[1]]
        c_g = paddle.exp(u_in + 1) - 1
        g_new_real = c_g * g_r + 1
        g_new_imag = c_g * g_i
        # g_new_real = paddle.reshape(g_new_real, [-1, 1])
        # g_new_imag = paddle.reshape(g_new_imag, [-1, 1])

        return {
            self.output_keys[0]: f_new_real,
            self.output_keys[1]: f_new_imag,
            self.output_keys[2]: g_new_real,
            self.output_keys[3]: g_new_imag,
        }
