import pdb
import numpy as np
import paddle
from paddle import nn
from ppsci.arch import base


# torch.manual_seed(123)

def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(shape=tensor.shape, dtype=tensor.dtype, min=a, max=b)
        )
        return tensor


def uniform_(tensor: paddle.Tensor, a: float, b: float) -> paddle.Tensor:
    """Modify tensor inplace using uniform_.

    Args:
        tensor (paddle.Tensor): Paddle Tensor.
        a (float): min value.
        b (float): max value.

    Returns:
        paddle.Tensor: Initialized tensor.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> param = paddle.empty((128, 256), "float32")
        >>> param = ppsci.utils.initializer.uniform_(param, -1, 1)
    """
    return _no_grad_uniform_(tensor, a, b)


class USCNN(base.Arch):
    def __init__(self, h: float, nx: int, ny: int, nVarIn: int = 1, nVarOut: int = 1, padSingleSide: int = 1,
                 initWay=None, k=5, s=1, p=2):
        super().__init__()
        """
        Extract basic information
        """
        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = 1
        self.p = 2
        self.deltaX = h
        self.nx = nx
        self.ny = ny

        """
        Define net
        """
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.conv1 = nn.Conv2D(self.nVarIn, 16, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2D(32, 16, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2D(16, self.nVarOut, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle = nn.PixelShuffle(1)
        self.apply(self.__init_weights)
        self.padSingleSide = padSingleSide
        self.udfpad = nn.Pad2D([padSingleSide, padSingleSide, padSingleSide, padSingleSide], value=0)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            bound = 1 / np.sqrt(np.prod(m.weight.shape[1:]))
            uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                uniform_(m.bias, -bound, bound)

    def forward(self, x):
        y=x['coords']
        y = self.US(y)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.pixel_shuffle(self.conv4(y))

        ## train
        y = self.udfpad(y)
        y = y[:, 0, :, :].reshape([y.shape[0], 1, y.shape[2], y.shape[3]])

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
