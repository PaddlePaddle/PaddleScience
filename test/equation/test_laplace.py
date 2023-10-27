import paddle
import pytest
import sympy as sp

import ppsci
from ppsci import arch
from ppsci import equation

__all__ = []


@pytest.mark.parametrize("dim", (2, 3))
def test_l1loss_mean(dim):
    """Test for only mean."""
    batch_size = 13
    input_dims = ("x", "y", "z")[:dim]
    output_dims = ("u",)

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
    model = arch.FullyConnectedLayer(len(input_dims), len(output_dims), 2, 16)
    model_sym = arch.MLP(input_dims, output_dims, 2, 16)
    model_sym.load_dict(model.state_dict())

    # manually generate output
    u = model(input_data)

    # use self-defined jacobian and hessian
    def jacobian(y: "paddle.Tensor", x: "paddle.Tensor") -> "paddle.Tensor":
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: "paddle.Tensor", x: "paddle.Tensor") -> "paddle.Tensor":
        return jacobian(jacobian(y, x), x)

    # compute expected result
    expected_result = hessian(u, x) + hessian(u, y)
    if dim == 3:
        expected_result += hessian(u, z)

    # compute result using built-in Laplace module
    laplace_equation = equation.Laplace(dim=dim)
    for name, expr in laplace_equation.equations.items():
        if isinstance(expr, sp.Basic):
            laplace_equation.equations[name] = ppsci.lambdify(
                expr,
                model_sym,
            )

    data_dict = {
        "x": x,
        "y": y,
        "u": u,
    }
    if dim == 3:
        data_dict["z"] = z
    test_result = laplace_equation.equations["laplace"](data_dict)

    # check result whether is equal
    assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
