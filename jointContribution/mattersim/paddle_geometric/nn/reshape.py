import paddle
from paddle import Tensor


class Reshape(paddle.nn.Layer):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """"""  # noqa: D419
        x = paddle.reshape(x, self.shape)
        return x

    def __repr__(self) -> str:
        shape = ', '.join([str(dim) for dim in self.shape])
        return f'{self.__class__.__name__}({shape})'
