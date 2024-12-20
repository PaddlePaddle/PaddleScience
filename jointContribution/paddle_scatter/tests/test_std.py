import paddle
from paddle_scatter import scatter_std


def test_std():
    src = paddle.to_tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype=paddle.float32)
    src.stop_gradient = False
    index = paddle.to_tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=paddle.int64)

    out = scatter_std(src, index, dim=-1, unbiased=True)
    std = src.std(axis=-1, unbiased=True)[0]
    expected = paddle.to_tensor([[std, 0], [0, std]])
    assert paddle.allclose(out, expected)

    out.backward(paddle.randn(out.shape, out.dtype))


def test_std_out():
    src = paddle.to_tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype=paddle.float32)
    index = paddle.to_tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=paddle.int64)
    out = paddle.to_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    scatter_std(src, index, dim=-1, out=out, unbiased=True)
    std = src.std(axis=-1, unbiased=True)[0]
    expected = paddle.to_tensor([[std, 0, 0], [0, std, 0]])

    assert paddle.allclose(out, expected)
