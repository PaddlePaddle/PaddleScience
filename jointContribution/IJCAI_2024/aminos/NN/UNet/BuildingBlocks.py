import paddle


class SCA3D(paddle.nn.Layer):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = paddle.nn.AdaptiveAvgPool3D(output_size=1)
        self.channel_excitation = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=channel, out_features=int(channel // reduction)
            ),
            paddle.nn.ReLU(),
            paddle.nn.Linear(
                in_features=int(channel // reduction), out_features=channel
            ),
        )
        self.spatial_se = paddle.nn.Conv3D(
            in_channels=channel,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )

    def forward(self, x):
        bahs, chs, _, _, _ = tuple(x.shape)
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = paddle.nn.functional.sigmoid(
            x=self.channel_excitation(chn_se).view(bahs, chs, 1, 1, 1)
        )
        chn_se = paddle.multiply(x=x, y=paddle.to_tensor(chn_se))
        spa_se = paddle.nn.functional.sigmoid(x=self.spatial_se(x))
        spa_se = paddle.multiply(x=x, y=paddle.to_tensor(spa_se))
        net_out = spa_se + x + chn_se
        return net_out


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return paddle.nn.Conv3D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias_attr=bias,
    )


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input
    Return:
        list of tuple (name, module)
    """
    assert "c" in order, "Conv layer MUST be present"
    assert (
        order[0] not in "rle"
    ), "Non-linearity cannot be the first operation in the layer"
    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", paddle.nn.ReLU()))
        elif char == "l":
            modules.append(("LeakyReLU", paddle.nn.LeakyReLU(negative_slope=0.1)))
        elif char == "e":
            modules.append(("ELU", paddle.nn.ELU()))
        elif char == "c":
            bias = not ("g" in order or "b" in order)
            modules.append(
                (
                    "conv",
                    conv3d(
                        in_channels, out_channels, kernel_size, bias, padding=padding
                    ),
                )
            )
        elif char == "g":
            is_before_conv = i < order.index("c")
            assert not is_before_conv, "GroupNorm MUST go after the Conv3d"
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(
                (
                    "groupnorm",
                    paddle.nn.GroupNorm(
                        num_groups=num_groups, num_channels=out_channels
                    ),
                )
            )
        elif char == "b":
            is_before_conv = i < order.index("c")
            if is_before_conv:
                modules.append(
                    ("batchnorm", paddle.nn.BatchNorm3D(num_features=in_channels))
                )
            else:
                modules.append(
                    ("batchnorm", paddle.nn.BatchNorm3D(num_features=out_channels))
                )
        else:
            raise ValueError(
                "Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
            )
    return modules


class SingleConv(paddle.nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="crg",
        num_groups=8,
        padding=1,
    ):
        super(SingleConv, self).__init__()
        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding=padding
        ):
            self.add_sublayer(name=name, sublayer=module)


class DoubleConv(paddle.nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        kernel_size=3,
        order="crg",
        num_groups=8,
    ):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = (conv1_out_channels, out_channels)
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.add_sublayer(
            name="SingleConv1",
            sublayer=SingleConv(
                conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups
            ),
        )
        self.add_sublayer(
            name="SingleConv2",
            sublayer=SingleConv(
                conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups
            ),
        )


class ExtResNetBlock(paddle.nn.Layer):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf
    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="cge",
        num_groups=8,
        **kwargs
    ):
        super(ExtResNetBlock, self).__init__()
        self.conv1 = SingleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
        )
        self.conv2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
        )
        n_order = order
        for c in "rel":
            n_order = n_order.replace(c, "")
        self.conv3 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=n_order,
            num_groups=num_groups,
        )
        if "l" in order:
            self.non_linearity = paddle.nn.LeakyReLU(negative_slope=0.1)
        elif "e" in order:
            self.non_linearity = paddle.nn.ELU()
        else:
            self.non_linearity = paddle.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out


class Encoder(paddle.nn.Layer):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=(2, 2, 2),
        pool_type="max",
        basic_module=DoubleConv,
        conv_layer_order="crg",
        num_groups=8,
    ):
        super(Encoder, self).__init__()
        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                self.pooling = paddle.nn.MaxPool3D(kernel_size=pool_kernel_size)
            else:
                self.pooling = paddle.nn.AvgPool3D(
                    kernel_size=pool_kernel_size, exclusive=False
                )
        else:
            self.pooling = None
        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(paddle.nn.Layer):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        scale_factor=(2, 2, 2),
        basic_module=DoubleConv,
        conv_layer_order="crg",
        num_groups=8,
    ):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            self.upsample = None
        else:
            self.upsample = paddle.nn.Conv3DTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=1,
                output_padding=1,
            )
            in_channels = out_channels
        self.scse = SCA3D(in_channels)
        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
        )

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = tuple(encoder_features.shape)[2:]
            x = paddle.nn.functional.interpolate(
                x=x, size=output_size, mode="nearest", data_format="NCDHW"
            )
            x = paddle.concat(x=(encoder_features, x), axis=1)
        else:
            x = self.upsample(x)
            x += encoder_features
        x = self.scse(x)
        x = self.basic_module(x)
        return x


class FinalConv(paddle.nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, order="crg", num_groups=8
    ):
        super(FinalConv, self).__init__()
        self.add_sublayer(
            name="SingleConv",
            sublayer=SingleConv(
                in_channels, in_channels, kernel_size, order, num_groups
            ),
        )
        final_conv = paddle.nn.Conv3D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.add_sublayer(name="final_conv", sublayer=final_conv)
