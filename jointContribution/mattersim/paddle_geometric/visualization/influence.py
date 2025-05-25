from typing import Any

import paddle
from paddle import Tensor
from paddle.autograd import grad


def influence(model: paddle.nn.Layer, src: Tensor, *args: Any) -> Tensor:
    x = src.clone()
    x.stop_gradient = False  # Enable gradient tracking
    out = model(x, *args).sum(axis=-1)

    influences = []
    for j in range(src.shape[0]):
        influence = grad(outputs=[out[j]], inputs=[x], retain_graph=True)[0].abs().sum(axis=-1)
        influences.append(influence / influence.sum())

    return paddle.stack(influences, axis=0)
