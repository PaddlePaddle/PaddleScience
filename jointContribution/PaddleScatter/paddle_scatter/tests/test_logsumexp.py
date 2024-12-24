import paddle
from paddle_scatter import scatter_logsumexp


def test_logsumexp():
    inputs = paddle.to_tensor(
        [
            0.5,
            0.5,
            0.0,
            -2.1,
            3.2,
            7.0,
            -1.0,
            -100.0,
        ]
    )
    inputs.stop_gradient = False
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2, 4, 4])
    splits = [2, 3, 1, 0, 2]

    outputs = scatter_logsumexp(inputs, index)

    for src, out in zip(inputs.split(splits), outputs.unbind()):
        if src.numel() > 0:
            assert out.numpy() == paddle.logsumexp(src, axis=0).numpy()
        else:
            assert out.item() == 0.0

    outputs.backward(paddle.randn(outputs.shape, outputs.dtype))


def test_logsumexp_out():
    src = paddle.to_tensor([-1.0, -50.0])
    index = paddle.to_tensor([0, 0])
    out = paddle.to_tensor([-10.0, -10.0])

    scatter_logsumexp(src=src, index=index, out=out)
    assert out.allclose(paddle.to_tensor([-0.9999, -10.0]), atol=1e-4)
