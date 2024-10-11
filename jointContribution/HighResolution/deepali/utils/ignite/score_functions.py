from typing import Mapping

import paddle
from ignite.engine import Engine

from ...core import get_tensor


def negative_loss_score_function(engine: Engine, key: str = "loss") -> paddle.Tensor:
    """Get negated loss value from ``engine.state.output``."""
    output = engine.state.output
    if isinstance(output, Mapping):
        loss = get_tensor(output, key)
    elif isinstance(output, paddle.Tensor):
        loss = output
    else:
        raise ValueError(
            "negative_loss_score_function() engine output loss must be a paddle.Tensor"
        )
    loss = loss.detach().sum()
    return -float(loss)
