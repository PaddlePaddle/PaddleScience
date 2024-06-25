import math

import paddle

from ppsci.arch import base


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = paddle.exp(x=paddle.arange(dtype="float32", end=half_dim) * -emb)
    emb = emb.to(device="gpu")
    emb = timesteps.astype(dtype="float32")[:, (None)] * emb[(None), :]
    emb = paddle.concat(x=[paddle.sin(x=emb), paddle.cos(x=emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = paddle.nn.functional.pad(emb, pad=(0, 1, 0, 0), mode="constant", value=0)
    return emb


def nonlinearity(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def Normalize(in_channels):
    return paddle.nn.GroupNorm(
        num_groups=8,
        num_channels=in_channels,
        epsilon=1e-06,
        weight_attr=True,
        bias_attr=True,
    )


class Upsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            )

    def forward(self, x):
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = paddle.nn.functional.pad(x, pad=pad, mode="circular")
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(
                kernel_size=2, stride=2, x=x, exclusive=False
            )
        return x


class ResnetBlock(paddle.nn.Layer):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        self.temb_proj = paddle.nn.Linear(
            in_features=temb_channels, out_features=out_channels
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="circular",
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, (None), (None)]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape([b, c, h * w])
        q = q.transpose(perm=[0, 2, 1])
        k = k.reshape([b, c, h * w])
        w_ = paddle.bmm(x=q, y=k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = v.reshape([b, c, h * w])
        w_ = w_.transpose(perm=[0, 2, 1])
        h_ = paddle.bmm(x=v, y=w_)
        h_ = h_.reshape([b, c, h, w])
        h_ = self.proj_out(h_)
        return x + h_


class DDPM(base.Arch):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.config = config
        ch, out_ch, ch_mult = (
            config.model.ch,
            config.model.out_ch,
            tuple(config.model.ch_mult),
        )
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        if config.model.type == "bayesian":
            out_3 = paddle.create_parameter(
                shape=paddle.zeros(shape=num_timesteps).shape,
                dtype=paddle.zeros(shape=num_timesteps).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.zeros(shape=num_timesteps)
                ),
            )
            out_3.stop_gradient = not True
            self.logvar = out_3
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temb = paddle.nn.Layer()
        self.temb.dense = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(in_features=self.ch, out_features=self.temb_ch),
                paddle.nn.Linear(in_features=self.temb_ch, out_features=self.temb_ch),
            ]
        )
        self.emb_conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=self.ch,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            paddle.nn.GELU(),
            paddle.nn.Conv2D(
                in_channels=self.ch,
                out_channels=self.ch,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
        )
        self.conv_in = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        self.combine_conv = paddle.nn.Conv2D(
            in_channels=self.ch * 2,
            out_channels=self.ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = paddle.nn.LayerList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) if len(self.up) != 0 else self.up.append(up)
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, x, t=None, dx=None):
        if self._input_transform is not None:
            x, t, dx, e = self._input_transform(x, self.config)
        assert x.shape[2] == x.shape[3] == self.resolution
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        x = self.conv_in(x)
        if dx is not None:
            cond_emb = self.emb_conv(dx)
        else:
            cond_emb = paddle.zeros_like(x=x)
        x = paddle.concat(x=(x, cond_emb), axis=1)
        hs = [self.combine_conv(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    paddle.concat(x=[h, hs.pop()], axis=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self._output_transform is not None:
            outputs = (h, e)
            result_dict = self._output_transform(outputs)
            return result_dict
        return h