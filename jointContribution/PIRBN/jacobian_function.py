import paddle


def flat(x, start_axis=0, stop_axis=None):
    # TODO Error if use paddle.flatten -> The Op flatten_grad doesn't have any gradop
    stop_axis = None if stop_axis is None else stop_axis + 1
    shape = x.shape

    # [3, 1] --flat--> [3]
    # [2, 2] --flat--> [4]
    temp = shape[start_axis:stop_axis]
    temp = [0 if x == 1 else x for x in temp]  # kill invalid axis
    flat_sum = sum(temp)
    head = shape[0:start_axis]
    body = [flat_sum]
    tail = [] if stop_axis is None else shape[stop_axis:]
    new_shape = head + body + tail
    x_flat = x.reshape(new_shape)
    return x_flat


def jacobian(y, x):
    J_shape = y.shape + x.shape
    J = paddle.zeros(J_shape)
    y_flat = flat(y)
    J_flat = flat(
        J, start_axis=0, stop_axis=len(y.shape) - 1
    )  # partialy flatten as y_flat
    for i, y_i in enumerate(y_flat):
        grad = paddle.grad(y_i, x, allow_unused=True)[
            0
        ]  # grad[i] == sum by j (dy[j] / dx[i])
        if grad is None:
            grad = paddle.zeros_like(x)
        J_flat[i] = grad
    return J_flat.reshape(J_shape)
