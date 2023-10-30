import paddle

def expand_left(src: paddle.Tensor, dim: int, dims: int) -> paddle.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)