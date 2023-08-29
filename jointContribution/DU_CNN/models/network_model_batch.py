# https://github.com/viktor-ktorvi/1d-convolutional-neural-networks

import models.basicblock as B
import paddle.nn as nn
from utils.pixelshuffle1d import PixelShuffle1D
from utils.pixelshuffle1d import PixelUnshuffle1D


class Networkn(nn.Layer):
    def __init__(self, nb, downsample, kerneln, nc, in_nc=1, out_nc=1, act_mode="BR"):

        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """

        super(Networkn, self).__init__()

        # encoder
        self.down = PixelUnshuffle1D(downsample)
        self.up = PixelShuffle1D(downsample)

        assert (
            "R" in act_mode or "L" in act_mode
        ), "Examples of activation function: R, L, BR, BL, IR, IL"
        bias = True

        m_head = B.conv1(
            in_nc * downsample,
            nc,
            kerneln,
            padding=kerneln // 2,
            mode="C" + act_mode[-1],
            bias=bias,
        )
        m_body = [
            B.conv1(
                nc, nc, kerneln, padding=kerneln // 2, mode="C" + act_mode, bias=bias
            )
            for _ in range(nb - 2)
        ]
        m_tail = B.conv1(
            nc, out_nc * downsample, kerneln, padding=kerneln // 2, mode="C", bias=bias
        )

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.down(x)
        x = self.model(x)
        x = self.up(x)
        return x
