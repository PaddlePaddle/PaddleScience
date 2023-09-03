# https://github.com/viktor-ktorvi/1d-convolutional-neural-networks

import paddle.nn as nn
from models import basicblock
from utils import pixelshuffle1d


class Networkn(nn.Layer):
    """Init function.

    Args:
        nb (int): Total number of conv layers.
        nc (int): Channel number.
        downsample (int): Downsample number.
        kerneln (int): Kernel number.
        in_nc (int, optional): Channel number of input. Defaults to 1.
        out_nc (int, optional): Channel number of output. Defaults to 1.
        act_mode (str, optional): Batch norm + activation function; 'BR' means BN+ReLU. Defaults to 'BR'.

    Raises:
        ValueError: Examples of activation function: R, L, BR, BL, IR, IL.
    """

    def __init__(self, nb, downsample, kerneln, nc, in_nc=1, out_nc=1, act_mode="BR"):
        super(Networkn, self).__init__()

        # encoder
        self.down = pixelshuffle1d.PixelUnshuffle1D(downsample)
        self.up = pixelshuffle1d.PixelShuffle1D(downsample)

        if "R" not in act_mode and "L" not in act_mode:
            raise ValueError("Examples of activation function: R, L, BR, BL, IR, IL.")
        bias = True

        m_head = basicblock.conv1(
            in_nc * downsample,
            nc,
            kerneln,
            padding=kerneln // 2,
            mode="C" + act_mode[-1],
            bias=bias,
        )
        m_body = [
            basicblock.conv1(
                nc, nc, kerneln, padding=kerneln // 2, mode="C" + act_mode, bias=bias
            )
            for _ in range(nb - 2)
        ]
        m_tail = basicblock.conv1(
            nc, out_nc * downsample, kerneln, padding=kerneln // 2, mode="C", bias=bias
        )

        self.model = basicblock.to_sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.down(x)
        x = self.model(x)
        x = self.up(x)
        return x
