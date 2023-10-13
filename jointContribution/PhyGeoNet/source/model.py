import pdb
import numpy as np
import paddle
from paddle import nn
from ppsci.arch import base


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
	def __init__(self,h,nx,ny,nVarIn=1,nVarOut=1,initWay=None,k=5,s=1,p=2):
		super().__init__()
		"""
		Extract basic information
		"""
		self.initWay=initWay
		self.nVarIn=nVarIn
		self.nVarOut=nVarOut
		self.k=k
		self.s=1
		self.p=2
		self.deltaX=h
		self.nx=nx
		self.ny=ny
		
		"""
		Define net
		"""
		self.relu=nn.ReLU()
		self.US=nn.Upsample(size=[self.ny-2,self.nx-2],mode='bicubic')
		self.conv1=nn.Conv2D(self.nVarIn,16,kernel_size=k, stride=s, padding=p)
		self.conv2=nn.Conv2D(16,32,kernel_size=k, stride=s, padding=p)
		self.conv3=nn.Conv2D(32,16,kernel_size=k, stride=s, padding=p)
		self.conv4=nn.Conv2D(16,self.nVarOut,kernel_size=k, stride=s, padding=p)
		self.pixel_shuffle = nn.PixelShuffle(1)
		self.apply(self.__init_weights)

		
	
	def __init_weights(self, m):
		if isinstance(m, nn.Conv2D):
			bound = 1 / np.sqrt(np.prod(m.weight.shape[1:]))
			uniform_(m.weight, -bound, bound)
			if m.bias is not None:
				uniform_(m.bias, -bound, bound)

	def forward(self, x):
		x=self.US(x)
		x=self.relu(self.conv1(x))
		x=self.relu(self.conv2(x))
		x=self.relu(self.conv3(x))
		x=self.pixel_shuffle(self.conv4(x))
		return x

