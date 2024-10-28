r"""Fully connected layers."""

import math
from typing import Optional

import paddle
from deepali.utils import paddle_aux  # noqa


class Linear(paddle.nn.Linear):
    r"""Fully connected layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: Optional[str] = "uniform",
    ) -> None:
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = paddle.nn.initializer
        if bias:
            bias_attr = None
        else:
            bias_attr = False
        super().__init__(in_features, out_features, bias_attr=bias_attr)

    def reset_parameters(self) -> None:
        # Initialize weights
        if self.weight_init == "uniform":
            init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
                negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
            )
            init_KaimingUniform(self.weight)
        elif self.weight_init == "constant":
            init_Constant = paddle.nn.initializer.Constant(value=0.1)
            init_Constant(self.weight)
        elif self.weight_init == "zeros":
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.weight)
        else:
            raise AssertionError(
                "Linear.reset_parameters() invalid 'init' value: {self.weight_init!r}"
            )
        # Initialize bias
        if self.bias is not None:
            if self.bias_init == "uniform":
                fan_in, _ = paddle_aux._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init_Uniform = paddle.nn.initializer.Uniform(low=-bound, high=bound)
                init_Uniform(self.bias)
            elif self.bias_init == "constant":
                init_Constant = paddle.nn.initializer.Constant(value=0.1)
                init_Constant(self.bias)
            elif self.bias_init == "zeros":
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(self.bias)
            else:
                raise AssertionError(
                    "Linear.reset_parameters() invalid 'bias' value: {self.bias_init!r}"
                )
