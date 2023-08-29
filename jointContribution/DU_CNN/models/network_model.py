# https://github.com/viktor-ktorvi/1d-convolutional-neural-networks

import paddle.nn as nn
from utils.pixelshuffle1d import PixelShuffle1D
from utils.pixelshuffle1d import PixelUnshuffle1D
from utils.sizes import Conv1dLayerSizes


class Network2(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network2, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network3(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network3, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network5(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network5, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network7(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network7, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network9(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network9, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network11(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network11, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            #  nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network13(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network13, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #   nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #   nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network15(nn.Layer):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network15, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(
            in_len=signal_len,
            in_ch=downsample,
            out_ch=channeln,
            kernel=kerneln,
            padding=kerneln // 2,
        )

        self.convint = nn.Sequential(
            nn.Conv1D(
                in_channels=convint_sizes.in_ch,
                out_channels=convint_sizes.out_ch,
                kernel_size=convint_sizes.kernel_size,
                padding=convint_sizes.padding,
            ),
            nn.ReLU(),
        )

        conv2_sizes = Conv1dLayerSizes(
            in_len=convint_sizes.out_len,
            in_ch=convint_sizes.out_ch,
            out_ch=convint_sizes.out_ch,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #  nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            # nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv1D(
                in_channels=conv2_sizes.in_ch,
                out_channels=conv2_sizes.out_ch,
                kernel_size=conv2_sizes.kernel_size,
                padding=conv2_sizes.padding,
            ),
            #   nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        convout_sizes = Conv1dLayerSizes(
            in_len=conv2_sizes.out_len,
            in_ch=conv2_sizes.out_ch,
            out_ch=downsample,
            kernel=kerneln,
            padding=kerneln // 2,
        )
        self.convout = nn.Sequential(
            nn.Conv1D(
                in_channels=convout_sizes.in_ch,
                out_channels=convout_sizes.out_ch,
                kernel_size=convout_sizes.kernel_size,
                padding=convout_sizes.padding,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.convout(x)
        x = self.up(x)
        return x
