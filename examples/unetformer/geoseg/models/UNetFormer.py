import einops
import paddle
from paddle_utils import add_tensor_methods
from paddle_utils import dim2perm

add_tensor_methods()


class DropPath(paddle.nn.Layer):
    """DropPath class"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype="float32")
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()  # mask
        output = (
            inputs.divide(keep_prob) * random_tensor
        )  # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class ConvBNReLU(paddle.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=paddle.nn.BatchNorm2D,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias_attr=bias,
                dilation=dilation,
                stride=stride,
                padding=(stride - 1 + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            paddle.nn.ReLU6(),
        )


class ConvBN(paddle.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=paddle.nn.BatchNorm2D,
        bias=False,
    ):
        super(ConvBN, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias_attr=bias,
                dilation=dilation,
                stride=stride,
                padding=(stride - 1 + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )


class Conv(paddle.nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False
    ):
        super(Conv, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias_attr=bias,
                dilation=dilation,
                stride=stride,
                padding=(stride - 1 + dilation * (kernel_size - 1)) // 2,
            )
        )


class SeparableConvBNReLU(paddle.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=paddle.nn.BatchNorm2D,
    ):
        super(SeparableConvBNReLU, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=(stride - 1 + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias_attr=False,
            ),
            norm_layer(out_channels),
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias_attr=False,
            ),
            paddle.nn.ReLU6(),
        )


class SeparableConvBN(paddle.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=paddle.nn.BatchNorm2D,
    ):
        super(SeparableConvBN, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=(stride - 1 + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias_attr=False,
            ),
            norm_layer(out_channels),
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias_attr=False,
            ),
        )


class SeparableConv(paddle.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=(stride - 1 + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias_attr=False,
            ),
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias_attr=False,
            ),
        )


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.ReLU6,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Conv2D(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True,
        )
        self.act = act_layer()
        self.fc2 = paddle.nn.Conv2D(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True,
        )
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(paddle.nn.Layer):
    def __init__(
        self,
        dim=256,
        num_heads=16,
        qkv_bias=False,
        window_size=8,
        relative_pos_embedding=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.ws = window_size
        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        self.attn_x = paddle.nn.AvgPool2D(
            kernel_size=(window_size, 1),
            stride=1,
            padding=(window_size // 2 - 1, 0),
            exclusive=False,
        )
        self.attn_y = paddle.nn.AvgPool2D(
            kernel_size=(1, window_size),
            stride=1,
            padding=(0, window_size // 2 - 1),
            exclusive=False,
        )
        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            self.relative_position_bias_table = (
                paddle.base.framework.EagerParamBase.from_tensor(
                    tensor=paddle.zeros(
                        shape=[(2 * window_size - 1) * (2 * window_size - 1), num_heads]
                    )
                )
            )
            coords_h = paddle.arange(end=self.ws)
            coords_w = paddle.arange(end=self.ws)
            coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = (
                coords_flatten[:, :, (None)] - coords_flatten[:, (None), :]
            )
            relative_coords = relative_coords.transpose(perm=[1, 2, 0]).contiguous()
            relative_coords[:, :, (0)] += self.ws - 1
            relative_coords[:, :, (1)] += self.ws - 1
            relative_coords[:, :, (0)] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(axis=-1)
            self.register_buffer(
                name="relative_position_index", tensor=relative_position_index
            )
            init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
            init_TruncatedNormal(self.relative_position_bias_table)

    def pad(self, x, ps):
        _, _, H, W = tuple(x.shape)
        if W % ps != 0:
            x = paddle.nn.functional.pad(x=x, pad=(0, ps - W % ps), mode="reflect")
        if H % ps != 0:
            x = paddle.nn.functional.pad(
                x=x,
                pad=(0, 0, 0, ps - H % ps),
                mode="reflect",
                pad_from_left_axis=False,
            )
        return x

    def pad_out(self, x):
        x = paddle.nn.functional.pad(x=x, pad=(0, 1, 0, 1), mode="reflect")
        return x

    def forward(self, x):
        B, C, H, W = tuple(x.shape)
        local = self.local2(x) + self.local1(x)
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = tuple(x.shape)
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv,
            "b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            qkv=3,
            ws1=self.ws,
            ws2=self.ws,
        )
        dots = q @ k.transpose(perm=dim2perm(k.ndim, -2, -1)) * self.scale
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.transpose(
                perm=[2, 0, 1]
            ).contiguous()
            dots += relative_position_bias.unsqueeze(axis=0)
        attn = paddle.nn.functional.softmax(dots, axis=-1)
        attn = attn @ v
        attn = einops.rearrange(
            attn,
            "(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )
        attn = attn[:, :, :H, :W]
        out = self.attn_x(
            paddle.nn.functional.pad(x=attn, pad=(0, 0, 0, 1), mode="reflect")
        ) + self.attn_y(
            paddle.nn.functional.pad(x=attn, pad=(0, 1, 0, 0), mode="reflect")
        )
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return out


class Block(paddle.nn.Layer):
    def __init__(
        self,
        dim=256,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=paddle.nn.ReLU6,
        norm_layer=paddle.nn.BatchNorm2D,
        window_size=8,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(paddle.nn.Layer):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-08):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[2], dtype="float32"), trainable=True
        )
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = paddle.nn.functional.interpolate(
            x=x, scale_factor=2, mode="bilinear", align_corners=False
        )
        weights = paddle.nn.ReLU()(self.weights)
        fuse_weights = weights / (paddle.sum(x=weights, axis=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(paddle.nn.Layer):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[2], dtype="float32"), trainable=True
        )
        self.eps = 1e-08
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=decode_channels,
                out_channels=decode_channels,
                kernel_size=3,
                padding=1,
                groups=decode_channels,
            ),
            paddle.nn.Sigmoid(),
        )
        self.ca = paddle.nn.Sequential(
            paddle.nn.AdaptiveAvgPool2D(output_size=1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            paddle.nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            paddle.nn.Sigmoid(),
        )
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = paddle.nn.ReLU6()

    def forward(self, x, res):
        x = paddle.nn.functional.interpolate(
            x=x, scale_factor=2, mode="bilinear", align_corners=False
        )
        weights = paddle.nn.ReLU()(self.weights)
        fuse_weights = weights / (paddle.sum(x=weights, axis=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x


class AuxHead(paddle.nn.Layer):
    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = paddle.nn.Dropout(p=0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = paddle.nn.functional.interpolate(
            x=feat, size=(h, w), mode="bilinear", align_corners=False
        )
        return feat


class Decoder(paddle.nn.Layer):
    def __init__(
        self,
        encoder_channels=(64, 128, 256, 512),
        decode_channels=64,
        dropout=0.1,
        window_size=8,
        num_classes=6,
    ):
        super(Decoder, self).__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        if self.training:
            self.up4 = paddle.nn.UpsamplingBilinear2D(scale_factor=4)
            self.up3 = paddle.nn.UpsamplingBilinear2D(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)
        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)
        self.segmentation_head = paddle.nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            paddle.nn.Dropout2D(p=dropout),
            Conv(decode_channels, num_classes, kernel_size=1),
        )
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)
            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)
            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = paddle.nn.functional.interpolate(
                x=x, size=(h, w), mode="bilinear", align_corners=False
            )
            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)
            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)
            x = self.p2(x, res2)
            x = self.b2(x)
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = paddle.nn.functional.interpolate(
                x=x, size=(h, w), mode="bilinear", align_corners=False
            )
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, paddle.nn.Conv2D):
                init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
                    negative_slope=1, nonlinearity="leaky_relu"
                )
                init_KaimingNormal(m.weight)
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)


class create_model(paddle.nn.Layer):
    def __init__(self, pretrained=True):
        super(create_model, self).__init__()
        resnet = paddle.vision.models.resnet18(pretrained=pretrained)
        self.stage0 = paddle.nn.Sequential(
            resnet.conv1, resnet.bn1, paddle.nn.ReLU(), resnet.maxpool
        )
        self.stage1 = resnet.layer1
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4
        self.outputs = []
        self.feature_info = type(
            "FeatureInfo",
            (object,),
            {
                "channels": lambda self: [64, 128, 256, 512],
                "reduction": lambda self: [4, 8, 16, 32],
            },
        )()

    def forward(self, x):
        self.outputs = []
        x = self.stage0(x)
        x = self.stage1(x)
        self.outputs.append(x)
        x = self.stage2(x)
        self.outputs.append(x)
        x = self.stage3(x)
        self.outputs.append(x)
        x = self.stage4(x)
        self.outputs.append(x)
        return self.outputs


class UNetFormer(paddle.nn.Layer):
    def __init__(
        self,
        decode_channels=64,
        dropout=0.1,
        backbone_name="swsl_resnet18",
        pretrained=True,
        window_size=8,
        num_classes=6,
    ):
        super().__init__()
        self.backbone = create_model(
            pretrained=pretrained,
        )
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(
            encoder_channels, decode_channels, dropout, window_size, num_classes
        )

    def forward(self, x):
        h, w = tuple(x.shape)[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x
