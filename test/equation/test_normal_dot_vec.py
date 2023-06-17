import paddle
import pytest

from ppsci import equation


def compute_func(x: tuple, y: tuple):
    z_i = paddle.zeros_like(x[0])
    for x_i, y_i in zip(x, y):
        z_i += x_i * y_i
    return z_i


def test_normal_dot_vel():
    batch_size = 13
    u = paddle.randn([batch_size, 1])
    v = paddle.randn([batch_size, 1])
    w = paddle.randn([batch_size, 1])

    normal_x = paddle.randn([batch_size, 1])
    normal_y = paddle.randn([batch_size, 1])
    normal_z = paddle.randn([batch_size, 1])

    pde = equation.NormalDotVec(("u", "v", "w"))
    out = {
        "u": u,
        "v": v,
        "w": w,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "normal_z": normal_z,
    }

    expected_result = compute_func((u, v, w), (normal_x, normal_y, normal_z))
    assert paddle.allclose(pde.equations["normal_dot_vel"](out), expected_result)


if __name__ == "__main__":
    pytest.main()
