from typing import Callable

import paddle

LambdaFunc = Callable[[paddle.Tensor], paddle.Tensor]


class LambdaLayer(paddle.nn.Layer):
    """Wrap any tensor operation in a network module."""

    def __init__(self, func: LambdaFunc) -> None:
        """Set callable tensor operation.

        Args:
            func: Callable tensor operation. Must be instance of ``paddle.nn.Layer``
                if it contains learnable parameters. In this case, however, the
                ``LambdaLayer`` wrapper becomes redundant. Main use is to wrap
                non-learnable Python functions.

        """
        super().__init__()
        self.func = func

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.func(x)
