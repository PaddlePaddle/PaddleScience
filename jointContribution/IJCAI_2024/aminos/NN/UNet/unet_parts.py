import paddle
import paddle_aux

""" Parts of the U-Net model """


class DoubleConv(paddle.nn.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = paddle.nn.Sequential(
            paddle.nn.Conv3D(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm3D(num_features=mid_channels),
            paddle.nn.ReLU(),
            paddle.nn.Conv3D(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm3D(num_features=out_channels),
            paddle.nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(paddle.nn.Layer):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_dim = out_channels
        self.maxpool_conv = paddle.nn.Sequential(
            paddle.nn.MaxPool3D(kernel_size=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(paddle.nn.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.out_dim = out_channels
        if bilinear:
            self.up = paddle.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = paddle.nn.Conv3DTranspose(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = tuple(x2.shape)[2] - tuple(x1.shape)[2]
        diffY = tuple(x2.shape)[3] - tuple(x1.shape)[3]
        diffX = tuple(x2.shape)[4] - tuple(x1.shape)[4]
        x1 = paddle_aux._FUNCTIONAL_PAD(
            pad=[
                diffZ // 2,
                diffZ - diffZ // 2,
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ],
            x=x1,
        )
        x = paddle.concat(x=[x2, x1], axis=1)
        return self.conv(x)


class OutConv(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = paddle.nn.Conv3D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)
