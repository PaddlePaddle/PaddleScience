import paddle


def init_weights(m):
    r"""Performs weight initialization.

    Args:
        m (paddle.nn.Layer): Paddle module

    """
    if isinstance(m, (paddle.nn.BatchNorm2D, paddle.nn.BatchNorm1D)):
        m.weight.set_value(paddle.ones_like(m.weight))
        m.bias.set_value(paddle.zeros_like(m.bias))
    elif isinstance(m, paddle.nn.Linear):
        paddle.nn.initializer.XavierUniform()(m.weight)
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
