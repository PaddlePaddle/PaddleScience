import paddle
import pytest
from paddle import nn
from paddle.nn import initializer

from ppsci.equation.pde import Vibration


@pytest.mark.parametrize("rho,k1,k2", [(1.0, 4.0, -1.0)])
def test_vibration(rho, k1, k2):
    """Test for Vibration equation."""
    batch_size = 13
    rho = rho
    k1 = paddle.create_parameter(
        shape=[],
        dtype=paddle.get_default_dtype(),
        default_initializer=initializer.Constant(k1),
    )
    k2 = paddle.create_parameter(
        shape=[],
        dtype=paddle.get_default_dtype(),
        default_initializer=initializer.Constant(k2),
    )
    # generate input data
    eta = paddle.randn([batch_size, 1])
    t_f = paddle.randn([batch_size, 1])
    eta.stop_gradient = False
    t_f.stop_gradient = False
    input_data = paddle.concat([eta, t_f], axis=1)
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Tanh(),
    )

    # manually generate output
    eta = model(input_data)

    def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return jacobian(jacobian(y, x), x)

    expected_result = (
        rho * hessian(eta, t_f)
        + paddle.exp(k1) * jacobian(eta, t_f)
        + paddle.exp(k2) * eta
    )
    # compute result using Vibration class

    vibration_equation = Vibration(rho=rho, k1=k1, k2=k2)
    data_dict = {"eta": eta, "t_f": t_f}
    test_result = vibration_equation.equations["f"](data_dict)
    # check result whether is equal
    assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
