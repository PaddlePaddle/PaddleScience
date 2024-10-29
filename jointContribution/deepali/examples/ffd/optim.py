from typing import Sequence

import paddle
from paddle.optimizer import Optimizer


def new_optimizer(name: str, model: paddle.nn.Layer, **kwargs) -> paddle.optimizer.Optimizer:
    r"""Initialize new optimizer for parameters of given model.

    Args:
        name: Name of optimizer.
        model: Module whose parameters are to be optimized.
        kwargs: Keyword arguments for named optimizer.

    Returns:
        New optimizer instance.

    """
    import paddle

    cls = getattr(paddle.optim, name, None)
    if cls is None:
        raise ValueError(f"Unknown optimizer: {name}")
    if not issubclass(cls, Optimizer):
        raise TypeError(f"Requested type '{name}' is not a subclass of paddle.optimizer.Optimizer")
    if "learning_rate" in kwargs:
        if "lr" in kwargs:
            raise ValueError("new_optimizer() 'lr' and 'learning_rate' are mutually exclusive")
        kwargs["lr"] = kwargs.pop("learning_rate")
    return cls(model.parameters(), **kwargs)


def slope_of_least_squares_fit(values: Sequence[float]) -> float:
    r"""Compute slope of least squares fit of line to last n objective function values

    See also:
    - https://www.che.udel.edu/pdf/FittingData.pdf
    - https://en.wikipedia.org/wiki/1_%2B_2_%2B_3_%2B_4_%2B_%E2%8B%AF
    - https://proofwiki.org/wiki/Sum_of_Sequence_of_Squares

    """
    n = len(values)
    if n < 2:
        return float("nan")
    if n == 2:
        return values[1] - values[0]
    sum_x1 = (n + 1) / 2
    sum_x2 = n * (n + 1) * (2 * n + 1) / 6
    sum_y1 = sum(values)
    sum_xy = sum((x + 1) * y for x, y in enumerate(values))
    return (sum_xy - sum_x1 * sum_y1) / (sum_x2 - n * sum_x1 * sum_x1)
