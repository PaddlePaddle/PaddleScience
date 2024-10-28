r"""Engine state output transformations for use as checkpoint score function."""
from typing import Mapping

import paddle
from deepali.core.collections import get_tensor
from ignite.engine import Engine


def negative_loss_score_function(engine: Engine, key: str = "loss") -> paddle.Tensor:
    r"""Get negated loss value from ``engine.state.output``."""
    output = engine.state.output
    if isinstance(output, Mapping):
        loss = get_tensor(output, key)
    elif isinstance(output, paddle.Tensor):
        loss = output
    else:
        raise ValueError("negative_loss_score_function() engine output loss must be a Tensor")
    loss = loss.detach().sum()
    return -float(loss)
