import math
from typing import Tuple

import paddle
from paddle import nn

from ppsci.arch import Arch


class ConvMeanPool(nn.Layer):
    """
    A convolutional layer followed by average pooling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        he_init: bool = True,
        biases: bool = True,
    ):
        super().__init__()
        self.conv2D = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias_attr=biases,
        )
        self.AvgPool2d = nn.AvgPool2D(kernel_size=2, stride=2)

        if he_init:
            xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
            xavier_uniform(self.conv2D.weight)
        else:
            xavier_uniform = nn.initializer.XavierUniform()
            xavier_uniform(self.conv2D.weight)

    def forward(self, x):
        x = self.conv2D(x)
        x = self.AvgPool2d(x)
        return x


class MeanPoolConv(nn.Layer):
    """
    Average pooling followed by a convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        he_init: bool = True,
        biases: bool = True,
    ):
        super().__init__()
        self.conv2D = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias_attr=biases,
        )
        self.avgpool2d = nn.AvgPool2D(kernel_size=2, stride=2)
        if he_init:
            xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
            xavier_uniform(self.conv2D.weight)
        else:
            xavier_uniform = nn.initializer.XavierUniform()
            xavier_uniform(self.conv2D.weight)

    def forward(self, x):
        x = self.avgpool2d(x)
        x = self.conv2D(x)
        return x


class UpsampleConv(nn.Layer):
    """
    A PixelShuffle layer followed by a convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        he_init: bool = True,
        biases: bool = True,
    ):
        super().__init__()
        self.PixelShuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv2D = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias_attr=biases,
        )
        if he_init:
            xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
            xavier_uniform(self.conv2D.weight)
        else:
            xavier_uniform = nn.initializer.XavierUniform()
            xavier_uniform(self.conv2D.weight)

    def forward(self, x):
        x = paddle.concat([x, x, x, x], axis=1)
        x = self.PixelShuffle(x)
        x = self.conv2D(x)
        return x


class ConditionalBatchNorm(nn.Layer):
    """
    Conditional batch normalization layer
    """

    def __init__(
        self, n_labels: int, channels: int, eps: int = 1e-5, momentum: int = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.scale = self.create_parameter(
            shape=[n_labels, channels],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.offset = self.create_parameter(
            shape=[n_labels, channels],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.register_buffer("mean", paddle.zeros([channels], dtype="float32"))
        self.register_buffer("var", paddle.ones([channels], dtype="float32"))
        self.momentum = momentum

    def forward(self, x, labels):
        if self.training:
            mean = paddle.mean(x, axis=[0, 2, 3], keepdim=True)
            var = paddle.sum(paddle.square(x - mean), [0, 2, 3], keepdim=True) / (
                x.shape[0] * x.shape[2] * x.shape[3] - 1
            )
            self.mean = (
                1 - self.momentum
            ) * self.mean + self.momentum * mean.squeeze().numpy()
            self.var = (
                1 - self.momentum
            ) * self.var + self.momentum * var.squeeze().numpy()
        else:
            mean = self.mean.reshape([1, -1, 1, 1])
            var = self.var.reshape([1, -1, 1, 1])

        label_scale = paddle.gather(self.scale, labels, axis=0)
        label_offset = paddle.gather(self.offset, labels, axis=0)

        label_scale = paddle.unsqueeze(paddle.unsqueeze(label_scale, -1), -1)
        label_offset = paddle.unsqueeze(paddle.unsqueeze(label_offset, -1), -1)

        outputs = (x - mean) / paddle.sqrt(var + self.eps)
        outputs = outputs * label_scale + label_offset

        return outputs


class LayerNorm(nn.Layer):
    """
    Layer normalization layer
    """

    def __init__(self, eps: int = 1e-5, momentum: int = 0.1):
        super().__init__()
        self.eps = eps
        self.scale = self.create_parameter(shape=[1], dtype="float32")
        self.offset = self.create_parameter(shape=[1], dtype="float32")
        self.register_buffer("mean", paddle.zeros([1], dtype="float32"))
        self.register_buffer("var", paddle.ones([1], dtype="float32"))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = paddle.mean(x, axis=[1, 2, 3], keepdim=True)
            var = paddle.sum(paddle.square(x - mean), [1, 2, 3], keepdim=True) / (
                x.shape[1] * x.shape[2] * x.shape[3] - 1
            )
            self.mean = (
                1 - self.momentum
            ) * self.mean + self.momentum * mean.mean().squeeze().squeeze().numpy()
            self.var = (
                1 - self.momentum
            ) * self.var + self.momentum * var.mean().squeeze().squeeze().numpy()
        else:
            mean = self.mean
            var = self.var
        outputs = (x - mean) / paddle.sqrt(var + self.eps)
        outputs = outputs * self.scale + self.offset
        return outputs


class Normalize(nn.Layer):
    """
    Normalization layer
    """

    def __init__(
        self, channels: int, mode: str, label_num: int, use_label: bool = True
    ):
        super().__init__()
        if mode == "Generator":
            if use_label:
                self.normalize = ConditionalBatchNorm(label_num, channels)
            else:
                self.normalize = nn.BatchNorm(channels)
        elif mode == "Discriminator":
            self.normalize = LayerNorm()
        else:
            self.normalize = nn.Identity()

    def forward(self, x, labels=None):
        if isinstance(self.normalize, ConditionalBatchNorm):
            return self.normalize(x, labels=labels)
        else:
            return self.normalize(x)


class ResidualBlock(nn.Layer):
    """
    Residual block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalize_mode: str,
        label_num: int,
        resample: str | None = None,
        use_label: bool = True,
    ):
        super().__init__()

        if resample == "down":
            self.conv_shortcut = ConvMeanPool(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                he_init=False,
            )
            self.conv_1 = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            )
            self.conv_2 = ConvMeanPool(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
            channel1 = in_channels
            channel2 = out_channels
        elif resample == "up":
            self.conv_shortcut = UpsampleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                he_init=False,
            )
            self.conv_1 = UpsampleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
            self.conv_2 = nn.Conv2D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            )
            channel1 = in_channels
            channel2 = out_channels
        elif resample is None:
            self.conv_shortcut = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
            self.conv_1 = nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
            )
            self.conv_2 = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            )
            channel1 = in_channels
            channel2 = in_channels
        else:
            raise Exception("invalid resample value")
        self.normalize1 = Normalize(
            channels=channel1,
            mode=normalize_mode,
            use_label=use_label,
            label_num=label_num,
        )
        self.normalize2 = Normalize(
            channels=channel2,
            mode=normalize_mode,
            use_label=use_label,
            label_num=label_num,
        )
        self.relu = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample
        xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
        if isinstance(self.conv_1, nn.Conv2D):
            xavier_uniform(self.conv_1.weight)
        if isinstance(self.conv_2, nn.Conv2D):
            xavier_uniform(self.conv_2.weight)
        if isinstance(self.conv_shortcut, nn.Conv2D):
            xavier_uniform = nn.initializer.XavierUniform()
            xavier_uniform(self.conv_shortcut.weight)

    def forward(self, x, labels=None):
        if self.out_channels == self.in_channels and self.resample is None:
            shortcut = x  # Identity skip-connection
        else:
            shortcut = self.conv_shortcut(x)

        x = self.normalize1(x=x, labels=labels)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.normalize2(x=x, labels=labels)
        x = self.relu(x)
        x = self.conv_2(x)

        return shortcut + x


class OptimizedResBlockDisc1(nn.Layer):
    """
    Optimized residual block
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_1 = nn.Conv2D(
            in_channels=3, out_channels=dim, kernel_size=3, padding="same"
        )
        self.conv_2 = ConvMeanPool(in_channels=dim, out_channels=dim, kernel_size=3)
        self.conv_shortcut = MeanPoolConv(
            in_channels=3, out_channels=dim, kernel_size=1, he_init=False, biases=True
        )
        self.relu = nn.ReLU()
        xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
        xavier_uniform(self.conv_1.weight)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        return shortcut + x


class Cifar10Generator(nn.Layer):
    """
    A generator for Cifar10

    Args
        channels: the number of channels in the intermediate features
        output_dim: the number of channels in the output image
        use_label: whether to use label

    """

    def __init__(
        self,
        channels: int,
        output_dim: int,
        label_num: int,
        use_label: bool = True,
    ):
        super().__init__()
        self.linear1 = nn.Linear(128, 4 * 4 * channels)
        self.ResidualBlocks = nn.LayerList(
            [
                ResidualBlock(
                    channels,
                    channels,
                    3,
                    resample="up",
                    normalize_mode="Generator",
                    use_label=use_label,
                    label_num=label_num,
                ),
                ResidualBlock(
                    channels,
                    channels,
                    3,
                    resample="up",
                    normalize_mode="Generator",
                    use_label=use_label,
                    label_num=label_num,
                ),
                ResidualBlock(
                    channels,
                    channels,
                    3,
                    resample="up",
                    normalize_mode="Generator",
                    use_label=use_label,
                    label_num=label_num,
                ),
            ]
        )
        self.Normalize = Normalize(
            channels=channels, mode="Generator", use_label=False, label_num=label_num
        )
        self.relu = nn.ReLU()
        self.conv = nn.Conv2D(
            in_channels=channels, out_channels=3, kernel_size=3, padding="same"
        )
        self.tanh = nn.Tanh()
        self.channels = channels
        self.output_dim = output_dim
        xavier_uniform = nn.initializer.XavierUniform()
        xavier_uniform(self.conv.weight)
        xavier_uniform(self.linear1.weight)

    def forward(self, labels, noise=None):
        if noise is None:
            noise = paddle.randn([labels.shape[0], 128], dtype=paddle.float32)
        x = self.linear1(noise)
        x = paddle.reshape(x, [-1, self.channels, 4, 4])
        for ResidualBlock_ in self.ResidualBlocks:
            x = ResidualBlock_(x, labels)
        x = self.Normalize(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
        return paddle.reshape(x, [-1, self.output_dim])


class Cifar10Discriminator(nn.Layer):
    """
    A discriminator for Cifar10
    Args
        dim: the number of channels in the intermediate features
        use_label: whether to use label
    """

    def __init__(self, dim: int, label_num: int, use_label: bool = True):
        super().__init__()
        self.optimized_resblock_disc1 = OptimizedResBlockDisc1(dim)
        self.ResidualBlocks = nn.LayerList(
            [
                ResidualBlock(
                    dim,
                    dim,
                    3,
                    resample="down",
                    normalize_mode="Discriminator",
                    use_label=use_label,
                    label_num=label_num,
                ),
                ResidualBlock(
                    dim,
                    dim,
                    3,
                    resample=None,
                    normalize_mode="Discriminator",
                    use_label=use_label,
                    label_num=label_num,
                ),
                ResidualBlock(
                    dim,
                    dim,
                    3,
                    resample=None,
                    normalize_mode="Discriminator",
                    use_label=use_label,
                    label_num=label_num,
                ),
            ]
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(dim, 1)
        self.linear2 = nn.Linear(dim, label_num)
        self.use_label = use_label
        xavier_uniform = nn.initializer.XavierUniform()
        xavier_uniform(self.linear.weight)
        xavier_uniform(self.linear2.weight)

    def forward(self, x, labels):
        x = paddle.reshape(x, [-1, 3, 32, 32])
        x = self.optimized_resblock_disc1(x)
        for ResidualBlock_ in self.ResidualBlocks:
            x = ResidualBlock_(x, labels)
        x = self.relu(x)
        x = paddle.mean(x, axis=[2, 3])
        x_wgan = self.linear(x)
        x_wgan = paddle.reshape(x_wgan, [-1])
        if self.use_label:
            x_acgan = self.linear2(x)
            return x_wgan, x_acgan
        else:
            return x_wgan, None


class WganGpCifar10Generator(Arch):
    """
    The Generator Of WGANGP for Cifar10.
    Args
        input_keys: the input keys
        output_keys: the output keys
        dim: the number of channels in the intermediate features
        output_dim: the number of channels in the output image
        use_label: whether to use label

    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        dim: int,
        output_dim: int,
        label_num: int,
        use_label: bool = True,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.generator = Cifar10Generator(
            dim,
            output_dim,
            use_label=use_label,
            label_num=label_num,
        )
        self.use_label = use_label

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)
        labels = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.generator(labels)
        y = self.split_to_dict(y, self.output_keys, axis=-1)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class WganGpCifar10Discriminator(Arch):
    """
    The Discriminator Of WGANGP for Cifar10.
    Args
        input_keys: the input keys
        output_keys: the output keys
        dim: the number of channels in the intermediate features
        use_label: whether to use label

    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        dim: int,
        label_num: int,
        use_label: bool = True,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.discriminator = Cifar10Discriminator(
            dim, use_label=use_label, label_num=label_num
        )
        self.use_label = use_label

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(
            x, self.input_keys[: int(len(self.input_keys) // 2)], axis=0
        )
        labels = self.concat_to_tensor(
            x, self.input_keys[int(len(self.input_keys) // 2) :], axis=0
        )
        y, y_acgan = self.discriminator(y, labels)
        y = self.split_to_dict(
            y, self.output_keys[: len(self.output_keys) // 2], axis=0
        )
        y_acgan = self.split_to_dict(
            y_acgan, self.output_keys[len(self.output_keys) // 2 :], axis=0
        )
        y.update(y_acgan)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
