import paddle
from paddle_utils import add_tensor_methods

__all__ = ["JointLoss", "WeightedLoss"]

add_tensor_methods()


class WeightedLoss(paddle.nn.Layer):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(paddle.nn.Layer):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(
        self,
        first: paddle.nn.Layer,
        second: paddle.nn.Layer,
        first_weight=1.0,
        second_weight=1.0,
    ):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)
