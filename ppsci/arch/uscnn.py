from typing import Tuple

import numpy as np
from paddle import nn

import ppsci
from ppsci.arch import base


class USCNN(base.Arch):
    """Physics-informed convolutional neural networks.

    Args:
       input_keys (Tuple[str, ...]): Name of input keys, such as ("x", "y", "z").
       output_keys (Tuple[str, ...]): Name of output keys, such as ("u", "v", "w").
       h float: the spatial step
       nx int:
       ny int:
       nVarIn int:
       nVarOut int:
       padSingleSide int:
       k int:
       s int:
       p int:
    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.PhyCRNet(
                input_channels=2,
                hidden_channels=[8, 32, 128, 128],
                input_kernel_size=[4, 4, 4, 3],
                input_stride=[2, 2, 2, 1],
                input_padding=[1, 1, 1, 1],
                dt=0.002,
                num_layers=[3, 1],
                upscale_factor=8
            )
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        h: float,
        nx: int,
        ny: int,
        nVarIn: int = 1,
        nVarOut: int = 1,
        padSingleSide: int = 1,
        k=5,
        s=1,
        p=2,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = s
        self.p = p
        self.deltaX = h
        self.nx = nx
        self.ny = ny
        self.padSingleSide = padSingleSide
        """
        Define net
        """
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode="bicubic")
        self.conv1 = nn.Conv2D(self.nVarIn, 16, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2D(32, 16, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2D(16, self.nVarOut, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle = nn.PixelShuffle(1)
        self.apply(self.__init_weights)
        self.udfpad = nn.Pad2D(
            [padSingleSide, padSingleSide, padSingleSide, padSingleSide], value=0
        )

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            bound = 1 / np.sqrt(np.prod(m.weight.shape[1:]))
            ppsci.utils.initializer.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                ppsci.utils.initializer.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        # train
        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.US(y)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.pixel_shuffle(self.conv4(y))

        y = self.udfpad(y)
        y = y[:, 0, :, :].reshape([y.shape[0], 1, y.shape[2], y.shape[3]])
        y = self.split_to_dict(y, self.output_keys)
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
