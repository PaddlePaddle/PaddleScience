import paddle


def bessel_i0(x: paddle.Tensor, name: str = None) -> paddle.Tensor:
    return paddle.i0(x, name)


def bessel_i0e(x: paddle.Tensor, name: str = None) -> paddle.Tensor:
    return paddle.i0e(x, name)


def bessel_i1(x: paddle.Tensor, name: str = None) -> paddle.Tensor:
    return paddle.i1(x, name)


def bessel_i1e(x: paddle.Tensor, name: str = None) -> paddle.Tensor:
    return paddle.i1e(x, name)
