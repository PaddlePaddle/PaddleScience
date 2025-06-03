import math
from typing import Tuple

import paddle
from paddle import nn

from ppsci.arch import Arch


class RuLULayer(nn.Layer):
    """
    A linear layer with ReLU activation
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        init_uniform = paddle.nn.initializer.Uniform(
            low=-(math.sqrt(6.0 / input_dim)), high=(math.sqrt(6.0 / input_dim))
        )
        init_uniform(self.linear.weight)

    def forward(self, x):
        return self.relu(self.linear(x))


class ToyGenerator(nn.Layer):
    """
    The generator of WGAN GP for toy data
    """

    def __init__(self, dim):
        super().__init__()
        self.reluLayer1 = RuLULayer(2, dim)
        self.reluLayer2 = RuLULayer(dim, dim)
        self.reluLayer3 = RuLULayer(dim, dim)
        self.linear = nn.Linear(dim, 2)
        xavier_uniform = nn.initializer.XavierUniform()
        xavier_uniform(self.linear.weight)

    def forward(self, batch_size):
        noise = paddle.randn([batch_size, 2])
        x = self.reluLayer1(noise)
        x = self.reluLayer2(x)
        x = self.reluLayer3(x)
        x = self.linear(x)
        return x


class ToyDiscriminator(nn.Layer):
    """
    The discriminator of WGAN GP for toy data
    """

    def __init__(self, dim):
        super().__init__()
        self.reluLayer1 = RuLULayer(2, dim)
        self.reluLayer2 = RuLULayer(dim, dim)
        self.reluLayer3 = RuLULayer(dim, dim)
        self.linear = nn.Linear(dim, 1)
        xavier_uniform = nn.initializer.XavierUniform()
        xavier_uniform(self.linear.weight)

    def forward(self, x):
        x = self.reluLayer1(x)
        x = self.reluLayer2(x)
        x = self.reluLayer3(x)
        x = self.linear(x)
        return x.reshape([-1])


class WganGpToyGenerator(Arch):
    """
    The Generator Of WGANGP for Toy.
    Args
        input_keys: the input keys
        output_keys: the output keys
        dim: the number of channels in the intermediate features

    """

    def __init__(self, output_keys: Tuple[str, ...], dim: int):
        super().__init__()
        self.output_keys = output_keys
        self.generator = ToyGenerator(dim)

    def forward(self, x):
        y = self.generator(next(iter(x.values())).shape[0])
        y = self.split_to_dict(y, self.output_keys, axis=-1)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class WganGpToyDiscriminator(Arch):
    """
    The Discriminator Of WGANGP for Toy.
    Args
        input_keys: the input keys
        output_keys: the output keys
        dim: the number of channels in the intermediate features

    """

    def __init__(
        self, input_keys: Tuple[str, ...], output_keys: Tuple[str, ...], dim: int
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.discriminator = ToyDiscriminator(dim)

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)
        y = self.concat_to_tensor(x, self.input_keys, axis=0)
        y = self.discriminator(y)
        y = self.split_to_dict(y, self.output_keys, axis=0)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
