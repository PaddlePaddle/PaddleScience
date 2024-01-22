# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from typing import Tuple

import paddle

from ppsci.arch import base


class NowcastNet(base.Arch):
    """The NowcastNet model.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        input_length (int, optional): Input length. Defaults to 9.
        total_length (int, optional): Total length. Defaults to 29.
        image_height (int, optional): Image height. Defaults to 512.
        image_width (int, optional): Image width. Defaults to 512.
        image_ch (int, optional): Image channel. Defaults to 2.
        ngf (int, optional): Noise Projector input length. Defaults to 32.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.NowcastNet(("input", ), ("output", ))
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        input_length: int = 9,
        total_length: int = 29,
        image_height: int = 512,
        image_width: int = 512,
        image_ch: int = 2,
        ngf: int = 32,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.input_length = input_length
        self.total_length = total_length
        self.image_height = image_height
        self.image_width = image_width
        self.image_ch = image_ch
        self.ngf = ngf

        configs = collections.namedtuple(
            "Object", ["ngf", "evo_ic", "gen_oc", "ic_feature"]
        )
        configs.ngf = self.ngf
        configs.evo_ic = self.total_length - self.input_length
        configs.gen_oc = self.total_length - self.input_length
        configs.ic_feature = self.ngf * 10

        self.pred_length = self.total_length - self.input_length
        self.evo_net = Evolution_Network(self.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.total_length, base_c=self.ngf)
        self.gen_dec = Generative_Decoder(configs)
        self.proj = Noise_Projector(self.ngf)
        sample_tensor = paddle.zeros(shape=[1, 1, self.image_height, self.image_width])
        self.grid = make_grid(sample_tensor)

    @staticmethod
    def split_to_dict(data_tensors: Tuple[paddle.Tensor, ...], keys: Tuple[str, ...]):
        return {key: data_tensors[i] for i, key in enumerate(keys)}

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        x_tensor = self.concat_to_tensor(x, self.input_keys)

        y = []
        out = self.forward_tensor(x_tensor)
        y.append(out)
        y = self.split_to_dict(y, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y

    def forward_tensor(self, x):
        all_frames = x[:, :, :, :, :1]
        frames = all_frames.transpose(perm=[0, 1, 4, 2, 3])
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        # Input Frames
        input_frames = frames[:, : self.input_length]
        input_frames = input_frames.reshape((batch, self.input_length, height, width))
        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape((batch, self.pred_length, 2, height, width))
        intensity_ = intensity.reshape((batch, self.pred_length, 1, height, width))
        series = []
        last_frames = all_frames[:, self.input_length - 1 : self.input_length, :, :, 0]
        grid = self.grid.tile((batch, 1, 1, 1))
        for i in range(self.pred_length):
            last_frames = warp(
                last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border"
            )
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = paddle.concat(x=series, axis=1)
        evo_result = evo_result / 128
        # Generative Network
        evo_feature = self.gen_enc(paddle.concat(x=[input_frames, evo_result], axis=1))
        noise = paddle.randn(shape=[batch, self.ngf, height // 32, width // 32])
        noise_feature = (
            self.proj(noise)
            .reshape((batch, -1, 4, 4, 8, 8))
            .transpose(perm=[0, 1, 4, 5, 2, 3])
            .reshape((batch, -1, height // 8, width // 8))
        )
        feature = paddle.concat(x=[evo_feature, noise_feature], axis=1)
        gen_result = self.gen_dec(feature, evo_result)
        return gen_result.unsqueeze(axis=-1)


class Evolution_Network(paddle.nn.Layer):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear)
        self.outc = OutConv(base_c * 1, n_classes)
        param1 = paddle.zeros(shape=[1, n_classes, 1, 1])
        gamma = self.create_parameter(
            shape=param1.shape,
            dtype=param1.dtype,
            default_initializer=paddle.nn.initializer.Assign(param1),
        )
        gamma.stop_gradient = False
        self.gamma = gamma
        self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
        self.outc_v = OutConv(base_c * 1, n_classes * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) * self.gamma
        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        v = self.outc_v(v)
        return x, v


class DoubleConv(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = paddle.nn.Sequential(
            paddle.nn.BatchNorm2D(num_features=in_channels),
            paddle.nn.ReLU(),
            paddle.nn.utils.spectral_norm(
                layer=paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
            ),
            paddle.nn.BatchNorm2D(num_features=mid_channels),
            paddle.nn.ReLU(),
            paddle.nn.utils.spectral_norm(
                layer=paddle.nn.Conv2D(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
            ),
        )
        self.single_conv = paddle.nn.Sequential(
            paddle.nn.BatchNorm2D(num_features=in_channels),
            paddle.nn.utils.spectral_norm(
                layer=paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
            ),
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x


class Down(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2),
            DoubleConv(in_channels, out_channels, kernel),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = paddle.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2
            )
        else:
            self.up = paddle.nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = paddle.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = paddle.concat(x=[x2, x1], axis=1)
        return self.conv(x)


class Up_S(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = paddle.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, kernel=kernel, mid_channels=in_channels
            )
        else:
            self.up = paddle.nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


class Generative_Encoder(paddle.nn.Layer):
    def __init__(self, n_channels, base_c=64):
        super().__init__()
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c, kernel=3)
        self.down1 = Down(base_c * 1, base_c * 2, 3)
        self.down2 = Down(base_c * 2, base_c * 4, 3)
        self.down3 = Down(base_c * 4, base_c * 8, 3)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class Generative_Decoder(paddle.nn.Layer):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        ic = opt.ic_feature
        self.fc = paddle.nn.Conv2D(
            in_channels=ic, out_channels=8 * nf, kernel_size=3, padding=1
        )
        self.head_0 = GenBlock(8 * nf, 8 * nf, opt)
        self.G_middle_0 = GenBlock(8 * nf, 4 * nf, opt, double_conv=True)
        self.G_middle_1 = GenBlock(4 * nf, 4 * nf, opt, double_conv=True)
        self.up_0 = GenBlock(4 * nf, 2 * nf, opt)
        self.up_1 = GenBlock(2 * nf, 1 * nf, opt, double_conv=True)
        self.up_2 = GenBlock(1 * nf, 1 * nf, opt, double_conv=True)
        final_nc = nf * 1
        self.conv_img = paddle.nn.Conv2D(
            in_channels=final_nc, out_channels=self.opt.gen_oc, kernel_size=3, padding=1
        )
        self.up = paddle.nn.Upsample(scale_factor=2)

    def forward(self, x, evo):
        x = self.fc(x)
        x = self.head_0(x, evo)
        x = self.up(x)
        x = self.G_middle_0(x, evo)
        x = self.G_middle_1(x, evo)
        x = self.up(x)
        x = self.up_0(x, evo)
        x = self.up(x)
        x = self.up_1(x, evo)
        x = self.up_2(x, evo)
        x = self.conv_img(paddle.nn.functional.leaky_relu(x=x, negative_slope=0.2))
        return x


class GenBlock(paddle.nn.Layer):
    def __init__(self, fin, fout, opt, use_se=False, dilation=1, double_conv=False):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        self.opt = opt
        self.double_conv = double_conv
        self.pad = paddle.nn.Pad2D(padding=dilation, mode="reflect")
        self.conv_0 = paddle.nn.Conv2D(
            in_channels=fin,
            out_channels=fmiddle,
            kernel_size=3,
            padding=0,
            dilation=dilation,
        )
        self.conv_1 = paddle.nn.Conv2D(
            in_channels=fmiddle,
            out_channels=fout,
            kernel_size=3,
            padding=0,
            dilation=dilation,
        )
        if self.learned_shortcut:
            self.conv_s = paddle.nn.Conv2D(
                in_channels=fin, out_channels=fout, kernel_size=1, bias_attr=False
            )
        self.conv_0 = paddle.nn.utils.spectral_norm(layer=self.conv_0)
        self.conv_1 = paddle.nn.utils.spectral_norm(layer=self.conv_1)
        if self.learned_shortcut:
            self.conv_s = paddle.nn.utils.spectral_norm(layer=self.conv_s)
        ic = opt.evo_ic
        self.norm_0 = SPADE(fin, ic)
        self.norm_1 = SPADE(fmiddle, ic)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, ic)

    def forward(self, x, evo):
        x_s = self.shortcut(x, evo)
        dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, evo))))
        if self.double_conv:
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, evo))))
        out = x_s + dx
        return out

    def shortcut(self, x, evo):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, evo))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return paddle.nn.functional.leaky_relu(x=x, negative_slope=0.2)


class SPADE(paddle.nn.Layer):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        self.param_free_norm = paddle.nn.InstanceNorm2D(
            num_features=norm_nc, weight_attr=False, bias_attr=False, momentum=1 - 0.1
        )
        nhidden = 64
        ks = 3
        pw = ks // 2
        self.mlp_shared = paddle.nn.Sequential(
            paddle.nn.Pad2D(padding=pw, mode="reflect"),
            paddle.nn.Conv2D(
                in_channels=label_nc, out_channels=nhidden, kernel_size=ks, padding=0
            ),
            paddle.nn.ReLU(),
        )
        self.pad = paddle.nn.Pad2D(padding=pw, mode="reflect")
        self.mlp_gamma = paddle.nn.Conv2D(
            in_channels=nhidden, out_channels=norm_nc, kernel_size=ks, padding=0
        )
        self.mlp_beta = paddle.nn.Conv2D(
            in_channels=nhidden, out_channels=norm_nc, kernel_size=ks, padding=0
        )

    def forward(self, x, evo):
        normalized = self.param_free_norm(x)
        evo = paddle.nn.functional.adaptive_avg_pool2d(x=evo, output_size=x.shape[2:])
        actv = self.mlp_shared(evo)
        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))
        out = normalized * (1 + gamma) + beta
        return out


class Noise_Projector(paddle.nn.Layer):
    def __init__(self, input_length):
        super().__init__()
        self.input_length = input_length
        self.conv_first = spectral_norm(
            paddle.nn.Conv2D(
                in_channels=self.input_length,
                out_channels=self.input_length * 2,
                kernel_size=3,
                padding=1,
            )
        )
        self.L1 = ProjBlock(self.input_length * 2, self.input_length * 4)
        self.L2 = ProjBlock(self.input_length * 4, self.input_length * 8)
        self.L3 = ProjBlock(self.input_length * 8, self.input_length * 16)
        self.L4 = ProjBlock(self.input_length * 16, self.input_length * 32)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        return x


class ProjBlock(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.one_conv = spectral_norm(
            paddle.nn.Conv2D(
                in_channels=in_channel,
                out_channels=out_channel - in_channel,
                kernel_size=1,
                padding=0,
            )
        )
        self.double_conv = paddle.nn.Sequential(
            spectral_norm(
                paddle.nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    padding=1,
                )
            ),
            paddle.nn.ReLU(),
            spectral_norm(
                paddle.nn.Conv2D(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    padding=1,
                )
            ),
        )

    def forward(self, x):
        x1 = paddle.concat(x=[x, self.one_conv(x)], axis=1)
        x2 = self.double_conv(x)
        output = x1 + x2
        return output


def make_grid(input):
    B, C, H, W = input.shape
    xx = paddle.arange(start=0, end=W).reshape((1, -1)).tile((H, 1))
    yy = paddle.arange(start=0, end=H).reshape((-1, 1)).tile((1, W))
    xx = xx.reshape((1, 1, H, W)).tile((B, 1, 1, 1))
    yy = yy.reshape((1, 1, H, W)).tile((B, 1, 1, 1))
    grid = paddle.concat(x=(xx, yy), axis=1).astype(dtype=paddle.get_default_dtype())
    return grid


def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):
    B, C, H, W = input.shape
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.transpose(perm=[0, 2, 3, 1])
    output = paddle.nn.functional.grid_sample(
        x=input.cpu(),
        grid=vgrid.cpu(),
        padding_mode=padding_mode,
        mode=mode,
        align_corners=True,
    )
    return output.cuda()


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class spectral_norm(paddle.nn.Layer):
    def __init__(self, module, name="weight", power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.detach().shape[0]
        for _ in range(self.power_iterations):
            v = l2normalize(
                paddle.mv(
                    x=paddle.t(input=w.reshape((height, -1)).detach()), vec=u.detach()
                )
            )
            u = l2normalize(
                paddle.mv(x=w.reshape((height, -1)).detach(), vec=v.detach())
            )
        sigma = u.dot(y=w.reshape((height, -1)).mv(vec=v))
        setattr(self.module, self.name, w / sigma.expand_as(y=w))

    def _made_params(self):
        try:
            _ = getattr(self.module, self.name + "_u")
            _ = getattr(self.module, self.name + "_v")
            _ = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.detach().shape[0]
        width = w.reshape((height, -1)).detach().shape[1]

        tmp_w = paddle.normal(shape=[height])
        out_0 = paddle.create_parameter(
            shape=tmp_w.shape,
            dtype=tmp_w.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(tmp_w),
        )
        out_0.stop_gradient = True
        u = out_0

        tmp_w = paddle.normal(shape=[width])
        out_1 = paddle.create_parameter(
            shape=tmp_w.shape,
            dtype=tmp_w.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(tmp_w),
        )
        out_1.stop_gradient = True
        v = out_1
        u = l2normalize(u)
        v = l2normalize(v)
        tmp_w = w.detach()
        out_2 = paddle.create_parameter(
            shape=tmp_w.shape,
            dtype=tmp_w.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(tmp_w),
        )
        out_2.stop_gradient = False
        w_bar = out_2
        del self.module._parameters[self.name]

        u = create_param(u)
        v = create_param(v)
        self.module.add_parameter(name=self.name + "_u", parameter=u)
        self.module.add_parameter(name=self.name + "_v", parameter=v)
        self.module.add_parameter(name=self.name + "_bar", parameter=w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def create_param(x):
    param = paddle.create_parameter(
        shape=x.shape,
        dtype=x.dtype,
        default_initializer=paddle.nn.initializer.Assign(x),
    )
    param.stop_gradient = x.stop_gradient
    return param
