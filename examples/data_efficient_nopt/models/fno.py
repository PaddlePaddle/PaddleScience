""" from original FNO repo """
# from pdb import set_trace as bp
# from turtle import forward

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from timm.models.layers import DropPath

# from timm.models.vision_transformer import Mlp
from .basics import SpectralConv2dV2
from .basics import _get_act


# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L45
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (B, num_windows, C, window_size, window_size)
    """
    if len(x.shape) == 4:
        B, C, H, W = x.shape
        x = x.view([B, C, H // window_size, window_size, W // window_size, window_size])
        windows = x.transpose([0, 2, 4, 1, 3, 5]).view(
            [B, -1, C, window_size, window_size]
        )  # B, n_win*n_win, C, win_s, win_s
    elif len(x.shape) == 5:
        B, J, C, H, W = x.shape
        # x = x.view(B*J, C, H, W)
        x = x.view(
            [B, J, C, H // window_size, window_size, W // window_size, window_size]
        )
        windows = x.transpose([0, 1, 3, 5, 2, 4, 6]).view(
            [B, J, -1, C, window_size, window_size]
        )  # B, J, n_win*n_win, C, win_s, win_s
    return windows


# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L60
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B, num_windows, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = windows.shape[0]
    if len(windows.shape) == 4:
        # B, n_win_sq, self.win_s, self.win_s
        x = windows.view(
            [B, H // window_size, W // window_size, window_size, window_size]
        )
        x = x.transpose([0, 1, 3, 2, 4]).view([B, H, W])
    if len(windows.shape) == 5:
        # B, n_win_sq, C, self.win_s, self.win_s
        x = windows.view(
            [B, H // window_size, W // window_size, -1, window_size, window_size]
        )
        x = x.transpose([0, 3, 1, 4, 2, 5]).contiguous().view([B, -1, H, W])
    elif len(windows.shape) == 6:
        # B, J, n_win_sq, C, self.win_s, self.win_s
        J = windows.shape[1]
        x = windows.view(
            [B, J, H // window_size, W // window_size, -1, window_size, window_size]
        )
        x = x.transpose([0, 1, 4, 2, 5, 3, 6]).contiguous().view([B, J, -1, H, W])
    return x


# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L26
class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FNN2d_Backbone(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        layers=None,
        in_dim=3,
        dropout=0,
        activation="tanh",
    ):
        super(FNN2d_Backbone, self).__init__()

        """
        The backbone network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the feature
        output shape: (batchsize, c=width, x=s, y=s)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.LayerList(
            [
                SpectralConv2dV2(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2
                )
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        self.ws = nn.LayerList(
            [
                nn.Conv1D(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.activation = _get_act(activation)

    def forward(self, x):
        """
        (b,c,h,w) -> (b,1,h,w)
        """
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.transpose([0, 2, 3, 1])
        x = self.fc0(x)  # project

        x = x.transpose([0, 3, 1, 2])

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view([batchsize, self.layers[i], -1])).view(
                [batchsize, self.layers[i + 1], size_x, size_y]
            )
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
            x = self.dropout(x)

        return x


class FNN2d(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
    ):
        super(FNN2d, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """

        self.backbone = FNN2d_Backbone(
            modes1, modes2, width, layers, in_dim, dropout, activation
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x):
        """
        (b,c,h,w) -> (b,1,h,w)
        """
        x = self.backbone(x)
        x = x.transpose([0, 2, 3, 1])
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.transpose([0, 3, 1, 2])

        if self.mean_constraint:
            x = x - paddle.mean(x, axis=(-2, -1), keepdim=True)

        return x

    def forward_icl(self, x, demo_xs, demo_ys, use_tqdm=False):
        """
        x: B, C, H, W
        demo_xs: J, C, H, W
        demo_ys: J, H, W
        """
        C_out = 1
        B, C, H, W = x.shape

        # repeat = 20; p = 0.05; sigma_range = [0, 0]
        repeat = 1
        p = 0.0
        sigma_range = [0, 0]
        x_aug = []
        demo_xs_aug = []
        for _ in range(repeat):
            if sum(sigma_range) > 0:
                import random

                sigma = random.uniform(*sigma_range)
                # https://github.com/scipy/scipy/blob/v1.11.4/scipy/ndimage/_filters.py#L232
                _kernel = min(
                    int((sigma * 4 + 1) / 2) * 2 + 1, (x.shape[1] // 2) * 2 - 1
                )
            mask = paddle.nn.functional.dropout(paddle.ones([1, C, H, W]), p=p)
            ######
            from .gaussian_blur import gaussian_blur

            if sum(sigma_range) > 0:
                _x_aug = gaussian_blur(
                    x.clone(), kernel_size=[_kernel, _kernel], sigma=sigma
                )
            else:
                _x_aug = x.clone()
            _x_aug = _x_aug * mask
            x_aug.append(_x_aug)
            ######
            _demo_xs_aug = []
            if sum(sigma_range) > 0:
                _demo_xs_aug = gaussian_blur(
                    demo_xs.clone(), kernel_size=[_kernel, _kernel], sigma=sigma
                )
            else:
                _demo_xs_aug = demo_xs.clone()
            _demo_xs_aug = _demo_xs_aug * mask
            demo_xs_aug.append(_demo_xs_aug)
        x_aug = paddle.stack(x_aug, axis=0)
        demo_xs_aug = paddle.stack(demo_xs_aug, axis=0)

        J = demo_xs.shape[0]
        # pred = self.forward(x) # B, H, W, T, 1
        ######################
        pred0 = self.forward(x)  # B, H, W, T, 1
        pred = paddle.stack([self.forward(_x) for _x in x_aug], axis=-1)  # B, 1, H, W
        C = pred.shape[-1]
        # pred = torch.stack([self.backbone(_x) for _x in x_aug], dim=-1) # B, 1, H, W TODO
        # C = pred0.shape[-1]
        ######################
        demo_pred = []
        idx = 0
        for _demo_xs_aug in demo_xs_aug:
            idx = 0
            _demo_pred = []
            while idx < _demo_xs_aug.shape[0]:
                _x = _demo_xs_aug[idx : idx + B]
                ##########
                _pred = self.forward(_x)
                # _pred = self.backbone(_x) # TODO
                ##########
                _demo_pred.append(_pred)
                idx += _x.shape[0]
            demo_pred.append(paddle.concat(_demo_pred, axis=0))
        demo_pred = paddle.stack(demo_pred, axis=-1)

        demo_pred_flat = demo_pred.view([1, -1, C])
        y_nn = paddle.zeros([B, C_out, H, W])
        stds_nn = paddle.zeros([B, 1, H, W])
        batch_b = 1
        _b = 0
        batch_h = 64
        _h = 0
        batch_w = 64
        _w = 0

        topk = int(20 * (J**0.5))  # TODO:
        pbar = None
        while _b < B:
            _h = 0
            while _h < H:
                _w = 0
                while _w < W:
                    if pbar is not None:
                        pbar.set_description("_b %d, _h %d, _w %d" % (_b, _h, _w))
                        pbar.update(1)
                    pred_flat = pred[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ]
                    __b, _, __h, __w, _ = pred_flat.shape
                    pred_flat = pred_flat.view([-1, 1, C])

                    # gap = (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2)
                    gap = paddle.linalg.norm(
                        (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2), axis=-1
                    )
                    # cos = 1 - pairwise_cosine_similarity(pred_flat.squeeze(1), demo_pred_flat.squeeze(0))
                    # gap = gap * cos
                    gap_re = gap.view([__b, __h, __w, -1])
                    index = paddle.argsort(paddle.abs(gap_re), -1)[
                        :, :, :, :topk
                    ]  # TODO: spatial index of ascending sort by pred gap
                    _y_nn = paddle.stack(
                        [
                            paddle.take_along_axis(
                                demo_ys.view([-1, C_out]),
                                index[:, :, :, _k].view([-1, 1]),
                                axis=0,
                            ).view([__b, C_out, __h, __w])
                            for _k in range(topk)
                        ],
                        -1,
                    )

                    y_nn[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ] = _y_nn.mean(-1)
                    stds_nn[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ] = paddle.abs(_y_nn.std(-1) / _y_nn.mean(-1))

                    # np.set_printoptions(precision=5)
                    # print(_h, _w, sum(pred[_b, :2, _h+1, _w+1]-y_nn[_b, :2, _h+1, _w+1]).item(), np.round(pred[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist(), np.round(y_nn[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist())

                    _w += batch_w
                _h += batch_h
            _b += batch_b
        # bp()
        # print(y_nn.mean(), demo_ys.mean())
        # return y_nn
        mask = (stds_nn < stds_nn.mean()).astype(paddle.float32)  # TODO:
        return mask * y_nn + (1 - mask) * pred0


# channel-wise concatenating X_demo and Y_demo
class FNN2d_FewShot_Baseline(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
        n_demos=7,
    ):
        super(FNN2d_FewShot_Baseline, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.backbone = FNN2d_Backbone(
            modes1, modes2, width, layers, in_dim, dropout, activation
        )
        self.dropout = nn.Dropout(p=dropout)
        #########################
        self.num_heads = 8
        self.fc1 = nn.Linear(
            layers[-1] * (n_demos + 1) + out_dim * n_demos * self.num_heads, fc_dim
        )
        self.fc2 = nn.Linear(
            fc_dim, out_dim
        )  # TODO compare fc_dim vs out_dim*n_demos*self.num_heads (demo's solutions)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.n_demos = n_demos

    def forward(self, demo_XY_query_x):
        """
        demo_XY_query_x: (b, J*c + J*1 + c, h, w)
        """
        demo_X, demo_Y, query_x = [], [], None
        B = len(demo_XY_query_x)
        C = self.in_dim
        J = (demo_XY_query_x.shape[1] - C) // (C + 1)
        H, W = demo_XY_query_x.shape[-2:]
        query_x = demo_XY_query_x[:, -C:]
        demo_X = demo_XY_query_x[:, : J * C]
        demo_Y = demo_XY_query_x[:, J * C : -C]
        """
        demo_X: (b, J*c, h, w)
        demo_Y: (b, J*1, h, w)
        query_x: (b, c, h, w)
        -> (b,1,h,w)
        """
        query_features = self.backbone(query_x).transpose(
            [0, 2, 3, 1]
        )  # B, C_fno, H, W
        # B, J*C, H, W = demo_X.shape()
        demo_features = (
            self.backbone(demo_X.view([B, J, C, H, W]).view([B * J, C, H, W]))
            .view([B, J * self.C_fno, H, W])
            .transpose([0, 2, 3, 1])
        )

        x = paddle.stack(
            [
                paddle.concat([query_features[_b], demo_features[_b]], axis=-1)
                for _b in range(B)
            ],
            axis=0,
        )
        x = paddle.stack(
            [
                paddle.concat(
                    [
                        x[_b],
                        demo_Y[_b].repeat([self.num_heads, 1, 1]).transpose([1, 2, 0]),
                    ],
                    axis=-1,
                )
                for _b in range(B)
            ],
            axis=0,
        )  # b, h, w, (1+J)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)  # b, h, w, 1

        x = x.transpose([0, 3, 1, 2])

        if self.mean_constraint:
            x = x - paddle.mean(x, axis=(-2, -1), keepdim=True)

        return x


# https://github.com/facebookresearch/deit/blob/main/models_v2.py#L42
class TransformerBlock(nn.Layer):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=nn.MultiHeadAttention,
        Mlp_block=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        self.attn = Attention_block(
            dim,
            num_heads,
            dropout=attn_drop,
            bias=True,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, query, key=None, value=None):
        """
        x: B, J+1, C
        """
        if key is None:
            key = query
        if value is None:
            value = query
        query = self.norm1(query)
        # x = x + self.drop_path(self.attn(x, x, x)[0])
        _query, attn_mat = self.attn(
            query, key, value, need_weights=True, average_attn_weights=True
        )  # attn_mat: B,L,L
        query = query + self.drop_path(_query)
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        return query, attn_mat


def simple_attention(query, key=None):
    if key is None:
        key = query
    attn = paddle.einsum("blc,bsc->bls", query, key)
    attn = attn.softmax(dim=-1)
    return attn


class DownSample(nn.Layer):
    def __init__(self, C_in, C_2d, C_out, shape, k=7, s=4, p=2):
        super(DownSample, self).__init__()
        # 7x7 2D conv and reduce the dimension down to 16x16x16, then flatten this and run it through a 1D conv
        self.conv2d = nn.Conv2D(C_in, C_2d, k, stride=s, padding=p)
        self.fc = nn.Linear(C_2d * (shape[0] // s) * (shape[1] // s), C_out)

    def forward(self, x):
        # B, C_fno, H, W
        # B, J, C_fno, H, W
        # query_features_down = self.query_downsample(query_features.view(B, -1)).view(B, 1, -1)
        J = 1
        if len(x.shape) == 5:
            B, J, C, H, W = x.shape
            x = x.view(-1, C, H, W)
        else:
            B, C, H, W = x.shape
        x = self.conv2d(x)
        x = x.view(B * J, -1)
        x = self.fc(x)
        return x.view(B, J, -1)


class UpSample(nn.Layer):
    def __init__(self, C_in, shape):
        super(UpSample, self).__init__()
        self.fc = nn.Linear(C_in, np.prod(shape))
        self.H, self.W = shape

    def forward(self, x):
        # x: B, C' (just for the query)
        B, C = x.shape
        x = self.fc(x).view(B, 1, self.H, self.W)
        return x


class FNN2d_FewShot_Spatial(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
        n_demos=7,
        l_attn=1,
        # input_shape=(64, 64),
        down=1,
        win_s=8,
        c_attn_hidden=1024,
        skip_backbone=False,
    ):
        super(FNN2d_FewShot_Spatial, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.down = down
        self.win_s = win_s
        self.skip_backbone = skip_backbone
        if not skip_backbone:
            self.backbone = FNN2d_Backbone(
                modes1, modes2, width, layers, in_dim, dropout, activation
            )
        else:
            self.in_dim = self.C_fno
        self.dropout = nn.Dropout(p=dropout)
        self.num_heads = 8
        self.l_attn = l_attn
        self.fc1 = nn.Linear(self.C_fno, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.n_demos = n_demos

    def forward(self, demo_XY_query_x):
        """
        demo_XY_query_x: (b, J*c + J*1 + c, h, w)
        """
        demo_X, demo_Y, query_x = [], [], None
        B = len(demo_XY_query_x)
        C = self.in_dim
        J = (demo_XY_query_x.shape[1] - C) // (C + 1)
        H, W = demo_XY_query_x.shape[-2:]
        query_x = demo_XY_query_x[:, :C]
        demo_X = demo_XY_query_x[:, C : (J + 1) * C]
        demo_Y = demo_XY_query_x[:, (J + 1) * C :]
        """
        demo_X: (b, J*c, h, w)
        demo_Y: (b, J*1, h, w)
        query_x: (b, c, h, w)
        -> (b,1,h,w)
        """
        # TODO:Ablation
        # demo_X[:, :C] = query_x
        # demo_Y[:, :1] = self.targets
        # TODO:Ablation

        if not self.skip_backbone:
            query_features = self.backbone(query_x)  # B, C_fno, H, W
            demo_features = self.backbone(
                demo_X.view([B, J, C, H, W]).view([B * J, C, H, W])
            ).view(B, J, self.C_fno, H, W)
        else:
            query_features = query_x
            demo_features = demo_X.view([B, J, self.C_fno, H, W])

        # TODO: downsample query_x, demo_X
        query_features_down = F.interpolate(
            query_features,
            size=(int(H) // self.down, int(W) // self.down),
            mode="bilinear",
            align_corners=True,
        )
        demo_features_down = paddle.stack(
            [
                F.interpolate(
                    demo_features[_b],
                    size=(int(H) // self.down, int(W) // self.down),
                    mode="bilinear",
                    align_corners=True,
                )
                for _b in range(B)
            ],
            axis=0,
        )
        # query_demo_features_down = torch.concat([query_features_down.unsqueeze(1), demo_features_down], dim=1)
        # TODO: chunk Xs into windows
        # windows = window_partition(query_demo_features_down, self.win_s) # B, J+1, n_win*n_win, C, win_s, win_s
        # n_win = int(windows.shape[2] ** 0.5)
        # sequence = windows.permute(0, 1, 2, 4, 5, 3).view(B*(J+1)*n_win*n_win, self.win_s**2, self.C_fno) # N, L, C; N = B*(J+1)*n_win*n_win; L = win_s**2
        query_windows = window_partition(
            query_features_down, self.win_s
        )  # B, n_win*n_win, C, win_s, win_s
        demo_windows = window_partition(
            demo_features_down, self.win_s
        )  # B, J, n_win*n_win, C, win_s, win_s
        n_win = int(query_windows.shape[1] ** 0.5)
        query_windows = query_windows.transpose([0, 1, 3, 4, 2]).view(
            [B * n_win**2, self.win_s**2, self.C_fno]
        )  # B*n_win*n_win, win_s*win_s, C
        demo_windows = demo_windows.transpose([0, 2, 1, 4, 5, 3]).view(
            [B * n_win**2, J * self.win_s**2, self.C_fno]
        )  # B*n_win*n_win, J*win_s*win_s, C

        self._attn_mats = []
        # # TODO: add position embedding
        # for _l in range(self.l_attn):
        #     # sequence, attn_mat = self.attns[_l](sequence)
        #     # cross-attention
        #     query_windows, attn_mat = self.attns[_l](query_windows, demo_windows)
        #     self._attn_mats.append(attn_mat.detach().cpu().numpy()) # N, L, S; N = B*n_win*n_win; L = win_s**2; S = J*win_s**2

        # TODO: simple attention
        attn_mat = simple_attention(query_windows, demo_windows)
        self._attn_mats.append(
            attn_mat.detach().cpu().numpy()
        )  # N, L, S; N = B*n_win*n_win; L = win_s**2; S = J*win_s**2

        demo_Y_down = F.interpolate(
            demo_Y,
            size=(int(H) // self.down, int(W) // self.down),
            mode="bilinear",
            align_corners=True,
        )
        # B, J, n_win*n_win, 1, win_s, win_s => B, J, n_win*n_win, win_s, win_s
        windows_Y = (
            window_partition(demo_Y_down.unsqueeze(2), self.win_s)
            .squeeze(3)
            .transpose([0, 2, 1, 3, 4])
            .view([B * n_win**2, J * self.win_s**2])
        )
        demo_Y_reweighted = paddle.einsum("bls,bs->bl", attn_mat, windows_Y).view(
            [B, n_win**2, 1, self.win_s, self.win_s]
        )
        demo_Y_reweighted = window_reverse(
            demo_Y_reweighted, self.win_s, int(H) // self.down, int(W) // self.down
        )  # B 1, H_down, W_down
        demo_Y_reweighted = F.interpolate(
            demo_Y_reweighted, size=(H, W), mode="bilinear", align_corners=True
        )  # B, 1, H, W
        self.query_score = demo_Y_reweighted[:, 0].detach().cpu().numpy()

        # query_demo_features = self.upsample(sequence[:, 0]).view(B, 1, H, W).transpose([0, 2, 3, 1]) # B, 1, H, W => B, H, W, 1
        # self.query_score = query_demo_features[:, :, :, 0].detach().cpu().numpy()

        # query_features = query_features.transpose(0, 2, 3, 1) # B, C_fno, H, W => B, H, W, C_fno
        y = self.fc1(
            query_features.transpose([0, 2, 3, 1])
        )  # B, C_fno, H, W => B, H, W, C_fno
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)  # b, h, w, 1

        y = y.transpose([0, 3, 1, 2])
        if self.mean_constraint:
            y = y - paddle.mean(y, axis=(-2, -1), keepdim=True)

        y = (y + demo_Y_reweighted) / 2  # TODO:

        return y


class FNN2d_FewShot_Spatial_v2(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
        n_demos=7,
        l_attn=1,
        # input_shape=(64, 64),
        down=1,
        win_s=8,
        c_attn_hidden=1024,
        skip_backbone=False,
    ):
        super(FNN2d_FewShot_Spatial_v2, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        1. Backbone:
            1) Lift the input to the desire channel dimension by self.fc0 .
            2) 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
        2. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=1, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.down = down
        self.win_s = win_s
        self.skip_backbone = skip_backbone
        if not skip_backbone:
            self.backbone = FNN2d_Backbone(
                modes1, modes2, width, layers, in_dim, dropout, activation
            )
        else:
            self.in_dim = self.C_fno
        self.dropout = nn.Dropout(p=dropout)
        self.num_heads = 8
        self.l_attn = l_attn
        self.fc1 = nn.Linear(self.C_fno, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.n_demos = n_demos

    def forward(self, demo_XY_query_x):
        """
        demo_XY_query_x: (b, J*c + J*1 + c, h, w)
        """
        demo_X, demo_Y, query_x = [], [], None
        B = len(demo_XY_query_x)
        C = self.in_dim
        J = (demo_XY_query_x.shape[1] - C) // (C + 1)
        H, W = demo_XY_query_x.shape[-2:]
        query_x = demo_XY_query_x[:, :C]
        demo_X = demo_XY_query_x[:, C : (J + 1) * C]
        demo_Y = demo_XY_query_x[:, (J + 1) * C :]
        """
        demo_X: (b, J*c, h, w)
        demo_Y: (b, J*1, h, w)
        query_x: (b, c, h, w)
        -> (b,1,h,w)
        """
        # TODO:Ablation
        # demo_X[:, :C] = query_x
        # demo_Y[:, :1] = self.targets
        # TODO:Ablation

        if not self.skip_backbone:
            query_features = self.backbone(query_x)  # B, C_fno, H, W
            demo_features = self.backbone(
                demo_X.view([B, J, C, H, W]).view([B * J, C, H, W])
            ).view(B, J, self.C_fno, H, W)
        else:
            query_features = query_x
            demo_features = demo_X.view([B, J, self.C_fno, H, W])

        self.query_score = None
        self._attn_mats = [None]

        y = self.fc1(
            query_features.transpose([0, 2, 3, 1])
        )  # B, C_fno, H, W => B, H, W, C_fno
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)  # b, h, w, 1
        y = y.transpose([0, 3, 1, 2])  # b, 1, h, w
        if self.mean_constraint:
            y = y - paddle.mean(y, axis=(-2, -1), keepdim=True)

        y_demo = self.fc1(
            demo_features.transpose([0, 1, 3, 4, 2])
        )  # B, J, C_fno, H, W => B, J, H, W, C_fno
        y_demo = self.activation(y_demo)
        y_demo = self.dropout(y_demo)
        y_demo = self.fc2(y_demo)
        y_demo = self.dropout(y_demo)  # b, j, h, w, 1
        y_demo = y_demo.transpose([0, 1, 4, 2, 3])  # b, j, 1, h, w
        if self.mean_constraint:
            y_demo = y_demo - paddle.mean(y_demo, axis=(-2, -1), keepdim=True)

        B, C, H, W = y.shape

        # # #########################################################
        y_flat = y.view([-1, 1])
        y_demo_flat = y_demo.view([1, -1])
        gap = y_flat - y_demo_flat
        gap_re = gap.view([B, C, H, W, -1])

        index = paddle.argsort(paddle.abs(gap_re), -1)

        topk = 100
        y_nn = 0
        for _k in range(topk):
            y_nn += paddle.take(demo_Y.view([-1, 1]), index[:, :, :, :, _k])
        y_nn /= topk
        y = (y + y_nn) / 2  # TODO:
        return y_nn


def build_fno(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut] * len(params.modes1)
        params.modes2 = [params.mode_cut] * len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut] * len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    if params.n_demos == 0:
        return FNN2d(
            params.modes1,
            params.modes2,
            layers=params.layers,
            fc_dim=params.fc_dim,
            in_dim=input_dim,
            out_dim=params.out_dim,
            dropout=params.dropout,
            activation="gelu",
            mean_constraint=(params.loss_func == "pde"),
        )
    else:
        if hasattr(params, "baseline") and params.baseline:
            return FNN2d_FewShot_Baseline(
                params.modes1,
                params.modes2,
                layers=params.layers,
                fc_dim=params.fc_dim,
                in_dim=input_dim,
                out_dim=params.out_dim,
                dropout=params.dropout,
                activation="gelu",
                mean_constraint=(params.loss_func == "pde"),
                n_demos=params.n_demos,
            )
        elif hasattr(params, "spatial") and params.spatial:
            return FNN2d_FewShot_Spatial_v2(
                params.modes1,
                params.modes2,
                layers=params.layers,
                fc_dim=params.fc_dim,
                in_dim=input_dim,
                out_dim=params.out_dim,
                dropout=params.dropout,
                activation="gelu",
                mean_constraint=(params.loss_func == "pde"),
                n_demos=params.n_demos,
                l_attn=params.l_attn,
                c_attn_hidden=params.c_attn_hidden,
                down=params.down,
                win_s=params.win_s,
                skip_backbone=(
                    params.train_path.endswith("npy")
                    and ("feature_data" in params.train_path)
                ),
            )
        # else:
        #     return FNN2d_FewShot(params.modes1, params.modes2, layers=params.layers, fc_dim=params.fc_dim,
        #                 in_dim=input_dim, out_dim=params.out_dim, dropout=params.dropout,
        #                 activation='gelu', mean_constraint=(params.loss_func == 'pde'), n_demos=params.n_demos, l_attn=params.l_attn,
        #                 input_shape=(params.nx, params.ny), k_conv2d=params.k_conv2d, s_conv2d=params.s_conv2d, c_conv2d=params.c_conv2d, c_attn_hidden=params.c_attn_hidden,
        #                 skip_backbone=(params.train_path.endswith("npy") and ("feature_data" in params.train_path))
        #                 )


class FNN2d_MAE(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        dropout=0,
        activation="tanh",
        mean_constraint=False,
    ):
        super(FNN2d_MAE, self).__init__()

        """
        The overall network. The backbone contains 4 layers of the Fourier layer.
        Backbone:
          1) Lift the input to the desire channel dimension by self.fc0 .
          2) 4 layers of the integral operators u' = (W + K)(u).
              W defined by self.w; K defined by self.conv .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=3, x=s, y=s)
        """
        self.in_dim = in_dim
        self.C_fno = layers[-1]
        self.fc_dim = fc_dim
        self.out_dim = out_dim
        self.encoder = FNN2d_Backbone(
            modes1, modes2, width, layers, in_dim, dropout, activation
        )
        self.decoder = FNN2d_Backbone(
            modes1,
            modes2,
            width,
            layers[:-1] + [in_dim],
            self.C_fno,
            dropout,
            activation,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_to_decoder = nn.Linear(self.C_fno, self.C_fno)
        #########################
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x, mask=None):
        """
        x: (b, c, h, w)
        """
        # import
        # B, C, H, W = x.shape
        if mask is None:
            x_enc = self.encoder(x)
        else:
            x_enc = self.encoder(x * mask)
        x_enc = self.encoder_to_decoder(x_enc.transpose([0, 2, 3, 1])).transpose(
            [0, 3, 1, 2]
        )
        x_dec = self.decoder(x_enc)
        return x_dec


def fno_pretrain(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut] * len(params.modes1)
        params.modes2 = [params.mode_cut] * len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut] * len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    return FNN2d_MAE(
        params.modes1,
        params.modes2,
        layers=params.layers,
        fc_dim=params.fc_dim,
        in_dim=input_dim,
        out_dim=params.out_dim,
        dropout=params.dropout,
        activation="gelu",
        mean_constraint=(params.loss_func == "pde"),
    )
