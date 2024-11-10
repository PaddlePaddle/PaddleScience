import numpy as np
import paddle


def min_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype=list(kwargs.values())[0])


setattr(paddle.Tensor, "view", view)


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def div(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    res = paddle.divide(self, y)

    if "rounding_mode" in kwargs:
        rounding_mode = kwargs["rounding_mode"]
        if rounding_mode == "trunc":
            res = paddle.trunc(res)
        elif rounding_mode == "floor":
            res = paddle.floor(res)

    return res


setattr(paddle.Tensor, "div", div)
setattr(paddle.Tensor, "divide", div)


def sqrt_complex(x):
    if x.dtype == paddle.complex64 or x.dtype == paddle.complex128:
        return paddle.to_tensor(np.sqrt(x.numpy()))
    return paddle.sqrt(x)


def zeros_like(x):
    return paddle.zeros(x.shape, dtype=x.dtype)


def bmm_fix(x, y):
    if x.is_sparse():
        x2 = x.to_dense()
        if y.dtype != x2.dtype:
            y = y.astype(x2.dtype)
        # x2.stop_gradient = True
        return paddle.bmm(x2, y)
    return paddle.bmm(x, y)


def norm_complex(x, p, axis):
    if x.dtype == paddle.complex64 or x.dtype == paddle.complex128:
        return paddle.linalg.norm(x.abs(), p, axis)
    return paddle.linalg.norm(x, p, axis)
