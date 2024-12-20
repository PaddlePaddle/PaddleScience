import paddle
from paddle_scatter import scatter_log_softmax
from paddle_scatter import scatter_softmax


def test_softmax():
    src = paddle.to_tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, float("-inf")])
    src.stop_gradient = False
    index = paddle.to_tensor([0, 1, 0, 1, 1, 2, 4, 4])

    out = scatter_softmax(src, index)

    out0 = paddle.nn.functional.softmax(paddle.to_tensor([0.2, 0.2]), axis=-1)
    out1 = paddle.nn.functional.softmax(paddle.to_tensor([0, -2.1, 3.2]), axis=-1)
    out2 = paddle.nn.functional.softmax(
        paddle.to_tensor([7], dtype=paddle.float32), axis=-1
    )
    out4 = paddle.nn.functional.softmax(paddle.to_tensor([-1, float("-inf")]), axis=-1)

    expected = paddle.stack(
        [out0[0], out1[0], out0[1], out1[1], out1[2], out2[0], out4[0], out4[1]], axis=0
    )

    assert paddle.allclose(out, expected)

    out.backward(paddle.randn(out.shape, out.dtype))


def test_log_softmax():
    # do not check float("-inf") here, since F.log_softmax has bug when dealing with -inf input on cpu
    # reported here: https://github.com/PaddlePaddle/Paddle/issues/69859

    # src = paddle.to_tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, float("-inf")])

    src = paddle.to_tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, -10])
    src.stop_gradient = False
    index = paddle.to_tensor([0, 1, 0, 1, 1, 2, 4, 4])

    out = scatter_log_softmax(src, index)

    out0 = paddle.nn.functional.log_softmax(paddle.to_tensor([0.2, 0.2]), axis=-1)
    out1 = paddle.nn.functional.log_softmax(paddle.to_tensor([0, -2.1, 3.2]), axis=-1)
    out2 = paddle.nn.functional.log_softmax(
        paddle.to_tensor([7], dtype=paddle.float32), axis=-1
    )

    # out4 = paddle.nn.functional.log_softmax(
    #     paddle.to_tensor([-1.0, float("-inf")]), axis=-1
    # )

    out4 = paddle.nn.functional.log_softmax(paddle.to_tensor([-1.0, -10.0]), axis=-1)

    expected = paddle.stack(
        [out0[0], out1[0], out0[1], out1[1], out1[2], out2[0], out4[0], out4[1]], axis=0
    )

    assert paddle.allclose(out, expected, rtol=1e-3)

    out.backward(paddle.randn(out.shape, out.dtype))
