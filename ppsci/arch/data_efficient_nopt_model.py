# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# refs: https://github.com/delta-lab-ai/data_efficient_nopt

from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as Tensor


def _get_act(activation):
    if activation == "tanh":
        func = F.tanh
    elif activation == "gelu":
        func = F.gelu
    elif activation == "relu":
        func = F.relu_
    elif activation == "elu":
        func = F.elu_
    elif activation == "leaky_relu":
        func = F.leaky_relu_
    else:
        raise ValueError(f"{activation} is not supported")
    return func


def compl_mul2d_v2(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    tmp = paddle.einsum("bixys,ioxyt->stboxy", a, b)
    return paddle.stack(
        [
            tmp[0, 0, :, :, :, :] - tmp[1, 1, :, :, :, :],
            tmp[1, 0, :, :, :, :] + tmp[0, 1, :, :, :, :],
        ],
        axis=-1,
    )


class SpectralConv2dV2(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand([in_channels, out_channels, self.modes1, self.modes2, 2])
        )
        self.weights2 = paddle.base.framework.EagerParamBase.from_tensor(
            self.scale
            * paddle.rand([in_channels, out_channels, self.modes1, self.modes2, 2])
        )

    def forward(self, x: paddle.Tensor):
        size_0 = x.shape[-2]
        size_1 = x.shape[-1]
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfft2(x.astype(paddle.float32), axes=(-2, -1), norm="ortho")
        x_ft = paddle.as_real(x_ft)

        out_ft = paddle.zeros(
            [batchsize, self.out_channels, size_0, size_1 // 2 + 1, 2]
        )
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d_v2(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d_v2(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        out_ft = paddle.as_complex(out_ft)

        x = paddle.fft.irfft2(out_ft, axes=(-2, -1), norm="ortho", s=(size_0, size_1))

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
        x = self.fc0(x)

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

        repeat = 1
        p = 0.0
        sigma_range = [0, 0]
        x_aug = []
        demo_xs_aug = []
        for _ in range(repeat):
            if sum(sigma_range) > 0:
                import random

                sigma = random.uniform(*sigma_range)
                _kernel = min(
                    int((sigma * 4 + 1) / 2) * 2 + 1, (x.shape[1] // 2) * 2 - 1
                )
            mask = paddle.nn.functional.dropout(paddle.ones([1, C, H, W]), p=p)
            if sum(sigma_range) > 0:
                _x_aug = gaussian_blur(
                    x.clone(), kernel_size=[_kernel, _kernel], sigma=sigma
                )
            else:
                _x_aug = x.clone()
            _x_aug = _x_aug * mask
            x_aug.append(_x_aug)
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
        pred0 = self.forward(x)
        pred = paddle.stack([self.forward(_x) for _x in x_aug], axis=-1)
        C = pred.shape[-1]
        demo_pred = []
        idx = 0
        for _demo_xs_aug in demo_xs_aug:
            idx = 0
            _demo_pred = []
            while idx < _demo_xs_aug.shape[0]:
                _x = _demo_xs_aug[idx : idx + B]
                _pred = self.forward(_x)
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

        topk = int(20 * (J**0.5))
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

                    gap = paddle.linalg.norm(
                        (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2), axis=-1
                    )
                    gap_re = gap.view([__b, __h, __w, -1])
                    index = paddle.argsort(paddle.abs(gap_re), -1)[:, :, :, :topk]
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

                    _w += batch_w
                _h += batch_h
            _b += batch_b

        mask = (stds_nn < stds_nn.mean()).astype(paddle.float32)
        return mask * y_nn + (1 - mask) * pred0


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
        self.num_heads = 8
        self.fc1 = nn.Linear(
            layers[-1] * (n_demos + 1) + out_dim * n_demos * self.num_heads, fc_dim
        )
        self.fc2 = nn.Linear(fc_dim, out_dim)
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
        query_features = self.backbone(query_x).transpose([0, 2, 3, 1])
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
        )
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        x = x.transpose([0, 3, 1, 2])

        if self.mean_constraint:
            x = x - paddle.mean(x, axis=(-2, -1), keepdim=True)

        return x


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

        if not self.skip_backbone:
            query_features = self.backbone(query_x)
            demo_features = self.backbone(
                demo_X.view([B, J, C, H, W]).view([B * J, C, H, W])
            ).view(B, J, self.C_fno, H, W)
        else:
            query_features = query_x
            demo_features = demo_X.view([B, J, self.C_fno, H, W])

        self.query_score = None
        self._attn_mats = [None]

        y = self.fc1(query_features.transpose([0, 2, 3, 1]))
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y.transpose([0, 3, 1, 2])
        if self.mean_constraint:
            y = y - paddle.mean(y, axis=(-2, -1), keepdim=True)

        y_demo = self.fc1(demo_features.transpose([0, 1, 3, 4, 2]))
        y_demo = self.activation(y_demo)
        y_demo = self.dropout(y_demo)
        y_demo = self.fc2(y_demo)
        y_demo = self.dropout(y_demo)
        y_demo = y_demo.transpose([0, 1, 4, 2, 3])
        if self.mean_constraint:
            y_demo = y_demo - paddle.mean(y_demo, axis=(-2, -1), keepdim=True)

        B, C, H, W = y.shape

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
        y = (y + y_nn) / 2
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
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x, mask=None):
        """
        x: (b, c, h, w)
        """
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


def _cast_squeeze_in(img: Tensor, req_dtypes: List[paddle.dtype]):
    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(axis=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _get_gaussian_kernel1d(
    kernel_size: int, sigma: float, dtype: paddle.dtype
) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = paddle.linspace(-ksize_half, ksize_half, num=kernel_size, dtype=dtype)
    pdf = paddle.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: paddle.dtype
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype)
    kernel2d = paddle.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: paddle.dtype
) -> Tensor:
    if need_squeeze:
        img = img.squeeze(axis=0)

    if need_cast:
        if out_dtype in (
            paddle.uint8,
            paddle.int8,
            paddle.int16,
            paddle.int32,
            paddle.int64,
        ):
            img = paddle.round(img)
        img = img.to(out_dtype)

    return img


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(
            f"sigma should be either float or sequence of floats. Got {type(sigma)}"
        )
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(
            f"If sigma is a sequence, its length should be 2. Got {len(sigma)}"
        )
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype)
    kernel = kernel.expand([img.shape[-3], 1, kernel.shape[0], kernel.shape[1]])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    img = paddle.nn.functional.pad(img, padding, mode="reflect")
    img = paddle.nn.functional.conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


class YParams:
    """Yaml file parser"""

    def __init__(self, yaml_params, config_name, mode):
        self._config_name = config_name
        self.params = {}
        self.mode = mode

        for key, val in yaml_params[config_name].items():
            if val == "None":
                val = None

            self.params[key] = val
            self.__setattr__(key, val)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)
