import math
from inspect import isfunction

import paddle


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PositionalEncoding(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = paddle.arange(dtype=noise_level.dtype, end=count) / count
        encoding = noise_level.unsqueeze(axis=1) * paddle.exp(
            x=-math.log(10000.0) * step.unsqueeze(axis=0)
        )
        encoding = paddle.concat(
            x=[paddle.sin(x=encoding), paddle.cos(x=encoding)], axis=-1
        )
        return encoding


class FeatureWiseAffine(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=in_channels,
                out_features=out_channels * (1 + self.use_affine_level),
            )
        )

    def forward(self, x, noise_embed):
        batch = tuple(x.shape)[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embed)
                .view([batch, -1, 1, 1])
                .chunk(chunks=2, axis=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).reshape([batch, -1, 1, 1])
        return x


class Swish(paddle.nn.Layer):
    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(x=x)


class Upsample(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.up = paddle.nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = paddle.nn.Conv2D(
            in_channels=dim, out_channels=dim, kernel_size=3, padding=1
        )

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Block(paddle.nn.Layer):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = paddle.nn.Sequential(
            paddle.nn.GroupNorm(num_groups=groups, num_channels=dim),
            Swish(),
            paddle.nn.Dropout(p=dropout) if dropout != 0 else paddle.nn.Identity(),
            paddle.nn.Conv2D(
                in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        dim_out,
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level
        )
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = (
            paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=1)
            if dim != dim_out
            else paddle.nn.Identity()
        )

    def forward(self, x, time_emb):
        b, c, h, w = tuple(x.shape)
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(paddle.nn.Layer):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        self.norm = paddle.nn.GroupNorm(num_groups=norm_groups, num_channels=in_channel)
        self.qkv = paddle.nn.Conv2D(
            in_channels=in_channel,
            out_channels=in_channel * 3,
            kernel_size=1,
            bias_attr=False,
        )
        self.out = paddle.nn.Conv2D(
            in_channels=in_channel, out_channels=in_channel, kernel_size=1
        )

    def forward(self, input):
        batch, channel, height, width = tuple(input.shape)
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).reshape([batch, n_head, head_dim * 3, height, width])
        query, key, value = qkv.chunk(chunks=3, axis=2)
        attn = paddle.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.reshape([batch, n_head, height, width, -1])
        attn = paddle.nn.functional.softmax(x=attn, axis=-1)
        attn = attn.reshape([batch, n_head, height, width, height, width])
        out = paddle.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.reshape([batch, channel, height, width]))
        return out + input


class ResnetBlocWithAttn(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False
    ):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


class UNet(paddle.nn.Layer):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
    ):
        super().__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = paddle.nn.Sequential(
                PositionalEncoding(inner_channel),
                paddle.nn.Linear(
                    in_features=inner_channel, out_features=inner_channel * 4
                ),
                Swish(),
                paddle.nn.Linear(
                    in_features=inner_channel * 4, out_features=inner_channel
                ),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        downs = [
            paddle.nn.Conv2D(
                in_channels=in_channel,
                out_channels=inner_channel,
                kernel_size=3,
                padding=1,
            )
        ]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=False,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
        self.downs = paddle.nn.LayerList(sublayers=downs)
        self.mid = paddle.nn.LayerList(
            sublayers=[
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                ),
            ]
        )
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=False,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
        self.ups = paddle.nn.LayerList(sublayers=ups)
        self.final_conv = Block(
            pre_channel, default(out_channel, in_channel), groups=norm_groups
        )

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(paddle.concat(x=(x, feats.pop()), axis=1), t)
            else:
                x = layer(x)
        return self.final_conv(x)
