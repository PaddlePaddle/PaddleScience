import math

import paddle
from paddle import nn

from ppsci.arch import Arch


class MnistGenerator(nn.Layer):
    """
    The generator of WGAN GP for mnist data
    """

    def __init__(self, dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(128, 4 * 4 * 4 * dim)
        self.relu1 = nn.ReLU()
        self.conv2d_transpose1 = nn.Conv2DTranspose(4 * dim, 2 * dim, 5, [2, 2], "same")
        self.relu2 = nn.ReLU()
        self.conv2d_transpose2 = nn.Conv2DTranspose(
            2 * dim, dim, 5, [2, 2], "same", output_padding=1
        )
        self.relu3 = nn.ReLU()
        self.conv2d_transpose3 = nn.Conv2DTranspose(dim, 1, 5, [2, 2], "same")
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        self.output_dim = output_dim
        xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
        xavier_uniform(self.conv2d_transpose1.weight)
        xavier_uniform(self.conv2d_transpose2.weight)
        xavier_uniform(self.conv2d_transpose3.weight)
        xavier_uniform = nn.initializer.XavierUniform()
        xavier_uniform(self.linear1.weight)

    def forward(self, x):
        y = paddle.randn([x.shape[0], 128])
        y = self.linear1(y)
        y = self.relu1(y)
        y = paddle.reshape(y, [-1, 4 * self.dim, 4, 4])
        y = self.conv2d_transpose1(y)
        y = self.relu2(y)
        y = y[:, :, :7, :7]
        y = self.conv2d_transpose2(y)
        y = self.relu3(y)
        y = self.conv2d_transpose3(y)
        y = self.sigmoid(y)
        return paddle.reshape(y, [-1, self.output_dim])


class MnistDiscriminator(nn.Layer):
    """
    The discriminator of WGAN GP for mnist data

    """

    def __init__(self, dim):
        super().__init__()

        self.conv2d_1 = nn.Conv2D(1, dim, kernel_size=5, padding="same", stride=[2, 2])
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2d_2 = nn.Conv2D(
            dim, 2 * dim, kernel_size=5, padding="same", stride=[2, 2]
        )
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv2d_3 = nn.Conv2D(
            2 * dim, 4 * dim, kernel_size=5, padding="same", stride=[2, 2]
        )
        self.leaky_relu3 = nn.LeakyReLU()
        self.linear = nn.Linear(4 * 4 * 4 * dim, 1)
        self.dim = dim
        xavier_uniform = nn.initializer.XavierUniform(gain=math.sqrt(2))
        xavier_uniform(self.conv2d_1.weight)
        xavier_uniform(self.conv2d_2.weight)
        xavier_uniform(self.conv2d_3.weight)
        xavier_uniform = nn.initializer.XavierUniform()
        xavier_uniform(self.linear.weight)

    def forward(self, x):
        x = paddle.reshape(x, [-1, 1, 28, 28])
        x = self.conv2d_1(x)
        x = self.leaky_relu1(x)
        x = self.conv2d_2(x)
        x = self.leaky_relu2(x)
        x = self.conv2d_3(x)
        x = self.leaky_relu3(x)
        x = paddle.reshape(x, [-1, 4 * 4 * 4 * self.dim])
        x = self.linear(x)
        return paddle.reshape(x, [-1])


class WganGpMnistGenerator(Arch):
    """
    The generator of WGAN GP for mnist data
    Args
        output_keys: the output keys of the generator
        batch_size: the batch size of the generator
        dim: the dimension of the generator
        output_dim: the output dimension of the generator
    """

    def __init__(self, output_keys, dim, output_dim):
        super().__init__()
        self.output_keys = output_keys
        self.dim = dim
        self.generator = MnistGenerator(dim, output_dim)

    def forward(self, x):
        y = self.generator(next(iter(x.values())))
        y = self.split_to_dict(y, self.output_keys, axis=0)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class WganGpMnistDiscriminator(Arch):
    """
    The discriminator of WGAN GP for mnist data
    Args
        input_keys: the input keys of the discriminator
        output_keys: the output keys of the discriminator
        dim: the dimension of the discriminator

    """

    def __init__(self, input_keys, output_keys, dim):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.discriminator = MnistDiscriminator(dim)

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)
        y = self.concat_to_tensor(x, self.input_keys, axis=0)
        y = self.discriminator(y)
        y = self.split_to_dict(y, self.output_keys, axis=0)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
