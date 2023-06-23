import paddle
import pytest
from paddle import nn

from ppsci import equation

__all__ = []


@pytest.mark.parametrize("dim", (2, 3))
def test_biharmonic(dim):
    """Test for biharmonic equation."""
    batch_size = 13
    input_dims = ("x", "y", "z")[:dim]
    output_dims = ("u",)

    q = -1.0
    D = 1.0

    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    input_data = paddle.concat([x, y], axis=1)
    if dim == 3:
        z = paddle.randn([batch_size, 1])
        z.stop_gradient = False
        input_data = paddle.concat([x, y, z], axis=1)

    # build NN model
    model = nn.Sequential(
        nn.Linear(len(input_dims), len(output_dims)),
        nn.Tanh(),
    )

    # manually generate output
    u = model(input_data)

    # use self-defined jacobian and hessian
    def jacobian(y: "paddle.Tensor", x: "paddle.Tensor") -> "paddle.Tensor":
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: "paddle.Tensor", x: "paddle.Tensor") -> "paddle.Tensor":
        return jacobian(jacobian(y, x), x)

    # compute expected result
    expected_result = -q / D

    # compute fourth order derivative
    vars = (x, y)
    if dim == 3:
        vars += (z,)
    for var_i in vars:
        for var_j in vars:
            expected_result += hessian(hessian(u, var_i), var_j)

    # compute result using built-in Biharmonic module
    biharmonic_equation = equation.Biharmonic(dim=dim, q=q, D=D)
    data_dict = {
        "x": x,
        "y": y,
        "u": u,
    }
    if dim == 3:
        data_dict["z"] = z
    test_result = biharmonic_equation.equations["biharmonic"](data_dict)

    # check result whether is equal
    assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
