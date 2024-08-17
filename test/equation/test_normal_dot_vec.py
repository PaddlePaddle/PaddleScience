import paddle
import pytest
import sympy as sp

import ppsci
from ppsci import arch
from ppsci import equation


def compute_func(x: tuple, y: tuple):
    z_i = paddle.zeros_like(x[0])
    for x_i, y_i in zip(x, y):
        z_i += x_i * y_i
    return z_i


def test_normal_dot_vel():
    batch_size = 13
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    z = paddle.randn([batch_size, 1])
    input_dims = ("x", "y", "z")
    output_dims = ("u", "v", "w")
    model = arch.MLP(input_dims, output_dims, 2, 16)
    output_dict = model(
        {
            "x": x,
            "y": y,
            "z": z,
        }
    )
    u = output_dict["u"]
    v = output_dict["v"]
    w = output_dict["w"]

    normal_x = paddle.randn([batch_size, 1])
    normal_y = paddle.randn([batch_size, 1])
    normal_z = paddle.randn([batch_size, 1])

    norm_doc_vec = equation.NormalDotVec(output_dims)
    for name, expr in norm_doc_vec.equations.items():
        if isinstance(expr, sp.Basic):
            norm_doc_vec.equations[name] = ppsci.lambdify(
                expr,
                model,
            )
    out = {
        "u": u,
        "v": v,
        "w": w,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "normal_z": normal_z,
    }

    expected_result = compute_func((u, v, w), (normal_x, normal_y, normal_z))
    assert paddle.allclose(
        norm_doc_vec.equations["normal_dot_vec"](out), expected_result
    )


if __name__ == "__main__":
    pytest.main()
