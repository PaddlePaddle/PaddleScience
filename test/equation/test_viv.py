import paddle
import pytest
import sympy as sp
from paddle.nn import initializer

import ppsci
from ppsci import arch
from ppsci.equation.pde import Vibration


@pytest.mark.parametrize("rho,k1,k2", [(1.0, 4.0, -1.0)])
def test_vibration(rho, k1, k2):
    """Test for Vibration equation."""
    batch_size = 13
    rho = rho
    k11 = paddle.create_parameter(
        shape=[],
        dtype=paddle.get_default_dtype(),
        name="k11",
        default_initializer=initializer.Constant(k1),
    )
    k22 = paddle.create_parameter(
        shape=[],
        name="k22",
        dtype=paddle.get_default_dtype(),
        default_initializer=initializer.Constant(k2),
    )
    # generate input data
    eta = paddle.randn([batch_size, 1])
    t_f = paddle.randn([batch_size, 1])
    eta.stop_gradient = False
    t_f.stop_gradient = False
    input_data = paddle.concat([eta, t_f], axis=1)
    input_dims = ("eta", "t_f")
    output_dims = ("f",)
    model = arch.MLP(input_dims, output_dims, 2, 16)

    # manually generate output
    eta = model.forward_tensor(input_data)

    def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return jacobian(jacobian(y, x), x)

    expected_result = (
        rho * hessian(eta, t_f)
        + paddle.exp(k11) * jacobian(eta, t_f)
        + paddle.exp(k22) * eta
    )

    # compute result using Vibration class
    vibration_equation = Vibration(rho=rho, k1=k1, k2=k2)
    for name, expr in vibration_equation.equations.items():
        if isinstance(expr, sp.Basic):
            vibration_equation.equations[name] = ppsci.lambdify(
                expr,
                model,
                vibration_equation.learnable_parameters,
            )
    data_dict = {"eta": eta, "t_f": t_f}
    test_result = vibration_equation.equations["f"](data_dict)
    # check result whether is equal
    assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
